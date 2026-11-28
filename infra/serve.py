"""FastAPI wrapper around the DLM sampler (GPU build for an AWS EC2 g4dn instance).

Loads the checkpoint once at startup, serves one generation at a time (a second concurrent request gets 503)
and refuses to start when the release and the container disagree (schedule, code revision, tokenizer ids).
DLM_DEVICE (default cuda) and DLM_DTYPE (fp32 default; fp16 only after a parity check, a T4 has no bf16).

Two endpoints:
  POST /generate              the original blocking endpoint (unchanged; gpu.sh smoke and deploy.md use it).
  POST /v1/chat/completions   OpenAI-compatible, modelled on Inception's Mercury API: `stream: true` sends Server-Sent Events
                              with the text as it becomes final; `stream: true, diffusing: true` sends the WHOLE text at every
                              denoising step (masked positions drawn as a block) for the client to overwrite.
  GET /demo                   a one-page browser demo of the diffusing stream (the text resolving out of mask blocks).
Everything for the streaming endpoint and the demo page lives in this file: the image build copies only serve.py and entrypoint.py from infra/.
"""
import json
import logging
import os
import threading
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from dlm.cleaning import clean_story
from dlm.data import HEADER
from dlm.sample import COLLAPSE_SHARE, END_NAMES, commit_plan, generate_text, load_checkpoint, plan_blocks, specials

log = logging.getLogger("uvicorn.error")


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    prompt: str = Field(default="", max_length=8192)
    mode: Literal["story", "continue"] | None = None
    max_new_tokens: int = Field(default=128, ge=1, le=480)
    steps: int | None = Field(default=None, ge=1, le=480)
    block_length: int = Field(default=32, ge=0, le=512)
    temperature: float = Field(default=0.8, ge=0, le=2)
    seed: int = Field(default=0, ge=0, le=2**63 - 1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    directory = Path(os.environ.get("DLM_MODEL_DIR", "/model"))
    device = os.environ.get("DLM_DEVICE", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DLM_DEVICE=cuda but CUDA is unavailable; check the host driver and --gpus all")
    dtype = os.environ.get("DLM_DTYPE", "fp32")
    if dtype not in {"fp32", "fp16"}:
        raise ValueError("DLM_DTYPE must be fp32 or fp16")
    if dtype == "fp16" and device != "cuda":
        raise ValueError("DLM_DTYPE=fp16 needs DLM_DEVICE=cuda")
    release = json.loads((directory / "deployment.json").read_text())
    if release["schedule"] not in {"linear", "cosine"}:
        raise ValueError("deployment.json must specify linear or cosine")
    if release["code_ref"] != os.environ["DLM_CODE_REF"]:
        raise ValueError("Container code revision differs from the model release")
    model, tokenizer = load_checkpoint(directory, device)
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match the checkpoint")
    if (tokenizer.token_to_id("[MASK]") != model.config.mask_id
            or tokenizer.token_to_id("[PAD]") != model.config.pad_id):
        raise ValueError("Tokenizer special IDs do not match the checkpoint")
    model = model.eval()
    if dtype == "fp16":
        model = model.half()
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.release = release
    app.state.device = device
    app.state.dtype = dtype
    app.state.gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    app.state.generation_lock = threading.Lock()
    yield


app = FastAPI(title="TinyStories DLM inference", lifespan=lifespan)


@app.get("/healthz")
def health(request: Request):
    state = request.app.state
    return {"status": "ready", "model_version": state.release["model_version"],
            "code_ref": state.release["code_ref"], "schedule": state.release["schedule"],
            "device": state.device, "dtype": state.dtype, "gpu": state.gpu,
            "context_length": state.model.config.seq_len}


@app.post("/generate")
def generate(body: GenerateRequest, request: Request):
    state = request.app.state
    if not state.generation_lock.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Model busy; retry shortly", headers={"Retry-After": "1"})
    try:
        started = time.perf_counter()
        result = generate_text(state.model, state.tokenizer,
                               schedule=state.release["schedule"], **body.model_dump())
        return {"text": result["text"], "model_version": state.release["model_version"],
                "schedule": state.release["schedule"], "seed": body.seed,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "forward_calls": result["forward_calls"], "truncated": result["truncated"],
                "warnings": result["warnings"]}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        state.generation_lock.release()


# ================================================================ streaming, OpenAI-compatible (Mercury-style)
MASK_GLYPH = "█"  # drawn for every still-masked token in `diffusing` frames
MAX_MESSAGES = 64
MAX_PROMPT_CHARS = 8192


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str | list[dict] | None = None


class ChatRequest(BaseModel):
    """OpenAI chat-completions body. Unknown fields (top_p, stop, user, stream_options, ...) are ignored, as most gateways
    send them; `n` above 1 is refused. Fields after `diffusing` are extensions of this server."""
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    model: str | None = None  # ignored: the one deployed model answers, and its version is echoed back
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)
    max_tokens: int | None = Field(default=None, ge=1, le=480)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=480)  # wins over max_tokens (Mercury's name)
    temperature: float | None = Field(default=None, ge=0, le=2)  # default 0.8
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)  # default 0
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = False
    diffusing: bool = False  # with stream: every chunk is the whole text so far, masked tokens drawn as a block
    # extensions
    chunk: Literal["block", "prefix"] = "block"  # stream granularity: when a decoding block completes / as soon as the text is final
    mode: Literal["story", "continue"] | None = None
    steps: int | None = Field(default=None, ge=1, le=480)
    block_length: int = Field(default=32, ge=0, le=512)


class Slot:
    """The generation lock, taken once. release() is idempotent so that every cleanup path (end of the stream, generator close,
    the response's background task, garbage collection) may call it: a client that disconnects before the first byte never
    starts the stream generator, so a `finally` inside it alone would leak the lock and every later request would get 503."""

    def __init__(self, lock: threading.Lock):
        self._lock, self._held, self._guard = lock, True, threading.Lock()

    @classmethod
    def take(cls, lock: threading.Lock):
        return cls(lock) if lock.acquire(blocking=False) else None

    def release(self) -> None:
        with self._guard:
            if not self._held:
                return
            self._held = False
        self._lock.release()

    def __del__(self):
        self.release()


class Decode:
    """One generation, validated up front (so a bad request is a 422 before any byte is streamed). `events()` runs the same
    semi-autoregressive loop as dlm.sample.diffusion_generate (same operations, same random draws in the same order, so the
    same seed gives the same text) but yields after every model call instead of returning at the end."""

    def __init__(self, model, tok, prompt: str, mode, max_new_tokens: int, steps, block_length: int,
                 temperature: float, schedule: str, seed: int):
        self.model, self.tok, self.schedule, self.temperature, self.seed = model, tok, schedule, temperature, seed
        mode = mode or ("continue" if prompt.strip() else "story")
        if mode == "story" and prompt.strip():
            raise ValueError("mode 'story' takes no opening words; send an empty user message or use mode 'continue'")
        if mode == "continue" and not prompt.strip():
            raise ValueError("mode 'continue' needs the opening words of the Story in the last user message")
        self.sp = specials(tok)
        self.end_ids = {self.sp[n] for n in END_NAMES}
        header_ids = tok.encode(HEADER, add_special_tokens=False).ids
        self.prompt_ids = tok.encode(clean_story(prompt), add_special_tokens=False).ids if mode == "continue" else []
        self.prefix = [self.sp["[BOS]"]] + header_ids + self.prompt_ids
        seq_len = model.config.seq_len
        if len(self.prefix) >= seq_len:
            raise ValueError(f"prompt too long: {len(self.prefix)} prefix tokens, the model's window is {seq_len}")
        self.gen_len = min(max_new_tokens, seq_len - len(self.prefix))
        self.blocks, self.steps_per_block = plan_blocks(self.gen_len, block_length, steps)

    @torch.no_grad()  # the decorator re-enters no_grad on every resume; a `with` around a yield would not (grad mode is per thread)
    def events(self):
        model, sp, P = self.model, self.sp, len(self.prefix)
        mask_id, pad_id = sp["[MASK]"], sp["[PAD]"]
        device = next(model.parameters()).device
        generator = torch.Generator().manual_seed(self.seed) if self.temperature > 0 else None
        ids = torch.tensor(self.prefix + [mask_id] * self.gen_len, device=device)[None]
        forward_calls = covered = 0
        for a, b in self.blocks:
            lo, hi = P + a, P + b
            for n_commit in commit_plan(b - a, min(self.steps_per_block, b - a), self.schedule):
                if n_commit == 0:
                    continue
                logits = model(ids)[0, lo:hi].float()
                forward_calls += 1
                logits[:, mask_id] = float("-inf")
                logits[:, pad_id] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                if self.temperature > 0:
                    x0 = torch.multinomial(torch.softmax(logits / self.temperature, dim=-1).cpu(), 1, generator=generator)
                    x0 = x0.squeeze(-1).to(device)
                else:
                    x0 = logits.argmax(dim=-1)
                conf = probs.gather(-1, x0[:, None]).squeeze(-1)
                conf = torch.where(ids[0, lo:hi] == mask_id, conf, torch.full_like(conf, float("-inf")))
                top = conf.topk(n_commit).indices
                ids[0, lo + top] = x0[top]
                yield {"kind": "step", "gen": ids[0, P:P + self.gen_len].tolist(), "hi": b}
            covered = hi
            yield {"kind": "block", "gen": ids[0, P:P + self.gen_len].tolist(), "hi": b}
            if any(t in self.end_ids for t in ids[0, P:hi].tolist()):  # later blocks are never run
                break
        gen = ids[0, P:covered].tolist()
        end_index = next((i for i, t in enumerate(gen) if t in self.end_ids), None)
        story = gen if end_index is None else gen[:end_index]
        warnings = []
        if len(story) >= 20:
            top_tok, n = Counter(story).most_common(1)[0]
            if n / len(story) >= COLLAPSE_SHARE:
                warnings.append(f"collapse: {n / len(story):.0%} of the generated tokens are {self.tok.id_to_token(top_tok)!r}")
        yield {"kind": "done", "story": story, "truncated": end_index is None, "forward_calls": forward_calls,
               "warnings": warnings, "prompt_tokens": P}


class Renderer:
    """Token ids -> the text the client sees: the Story WITHOUT the opening words (a chat reply does not echo the prompt;
    `prompt + content` equals what /generate returns as `text`)."""

    def __init__(self, dec: Decode):
        self.tok, self.prompt_ids, self.mask_id, self.end_ids = dec.tok, dec.prompt_ids, dec.sp["[MASK]"], dec.end_ids
        self.base = self.tok.decode(self.prompt_ids, skip_special_tokens=True)

    def cont(self, story_ids: list[int]) -> str:
        full = self.tok.decode(self.prompt_ids + story_ids, skip_special_tokens=True)
        return full[len(self.base):] if full.startswith(self.base) else full

    def prefix(self, gen: list[int]):
        """(committed tokens from the start of the response up to the first masked one, cut at the first end token; whether an end was seen).
        Committed tokens never change, so this is always a true prefix of the final Story."""
        j = gen.index(self.mask_id) if self.mask_id in gen else len(gen)
        pre = gen[:j]
        for i, t in enumerate(pre):
            if t in self.end_ids:
                return pre[:i], True
        return pre, False

    def frame(self, gen: list[int]) -> str:
        """The whole text so far with a block for every still-masked token; nothing after the first committed end token."""
        end = next((i for i, t in enumerate(gen) if t != self.mask_id and t in self.end_ids), None)
        gen = gen if end is None else gen[:end]
        out, run = [], []
        for t in gen:
            if t == self.mask_id:
                if run:
                    out.append(self.tok.decode(run, skip_special_tokens=True))
                    run = []
                out.append(MASK_GLYPH)
            else:
                run.append(t)
        if run:
            out.append(self.tok.decode(run, skip_special_tokens=True))
        return "".join(out)


def _pieces(dec: Decode, events, chunk: str, diffusing: bool):
    """Yield ("content", text) pieces and finally ("done", info).
    appendable (default): each piece is new text to append; text is sent only once final (end of a decoding block, or as soon as
    the committed prefix grows with chunk='prefix'); a trailing U+FFFD (half a UTF-8 character) is held back until the next piece.
    diffusing: each piece is the WHOLE text so far, masked tokens as blocks; the last piece is exactly the final text."""
    rd, last, emitted, seen = Renderer(dec), None, "", 0
    for ev in events:
        if ev["kind"] == "done":
            final = rd.cont(ev["story"])
            if diffusing:
                if final != last:
                    yield "content", final
            elif final.startswith(emitted):
                if len(final) > len(emitted):
                    yield "content", final[len(emitted):]
            else:  # not expected (tests assert it never happens); what was sent cannot be taken back
                log.warning("streamed text is not a prefix of the final text; the remainder was not sent")
            yield "done", ev
            return
        if diffusing:
            frame = rd.frame(ev["gen"][:ev["hi"]])
            if frame != last:
                last = frame
                yield "content", frame
        elif ev["kind"] == "block" or chunk == "prefix":
            pre, ended = rd.prefix(ev["gen"])
            if len(pre) <= seen and not ended:
                continue
            seen = len(pre)
            text = rd.cont(pre)
            if not ended:
                text = text.rstrip("�")
            if text.startswith(emitted) and len(text) > len(emitted):
                yield "content", text[len(emitted):]
                emitted = text


def content_stream(dec: Decode, chunk: str, diffusing: bool):
    """_pieces() over a fresh generation; the decoding generator is closed as soon as the stream ends or is closed."""
    events = dec.events()
    try:
        yield from _pieces(dec, events, chunk, diffusing)
    finally:
        events.close()


def _sse(obj) -> str:
    return "data: " + json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n\n"


def _finish_reason(info) -> str:
    return "length" if info["truncated"] else "stop"


def _usage(info) -> dict:
    n = len(info["story"])
    return {"prompt_tokens": info["prompt_tokens"], "completion_tokens": n, "total_tokens": info["prompt_tokens"] + n}


def _last_user_text(body: ChatRequest) -> str:
    last = body.messages[-1]
    if last.role != "user":
        raise ValueError("the last message must have role 'user' (its text is the opening words of the Story; "
                         "this pre-trained model does not follow instructions, and earlier messages are ignored)")
    c = last.content
    if isinstance(c, list):
        c = "".join(part.get("text", "") for part in c if isinstance(part, dict) and part.get("type", "text") == "text")
    c = c or ""
    if len(c) > MAX_PROMPT_CHARS:
        raise ValueError(f"the last user message is longer than {MAX_PROMPT_CHARS} characters")
    return c


def build_decode(body: ChatRequest, model, tok, schedule: str) -> Decode:
    """Raises ValueError for anything the client got wrong (the endpoint turns it into a 422)."""
    if body.diffusing and not body.stream:
        raise ValueError("diffusing needs stream=true")
    return Decode(model, tok, _last_user_text(body), body.mode,
                  max_new_tokens=body.max_completion_tokens or body.max_tokens or 128, steps=body.steps,
                  block_length=body.block_length, temperature=0.8 if body.temperature is None else body.temperature,
                  schedule=schedule, seed=body.seed or 0)


def _guarded(slot: Slot, pieces, head: dict, started: float, release: dict):
    """The SSE byte stream. The lock is released when it ends, fails or is closed; Slot.release() is idempotent."""
    try:
        base = {"id": head["id"], "object": "chat.completion.chunk", "created": head["created"], "model": head["model"]}
        yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
        info = None
        try:
            for kind, payload in pieces:
                if kind == "content":
                    yield _sse({**base, "choices": [{"index": 0, "delta": {"content": payload}, "finish_reason": None}]})
                else:
                    info = payload
        except Exception:  # headers are already sent: report in-band, then end the stream cleanly
            log.exception("generation failed while streaming")
            yield _sse({**base, "error": {"message": "generation failed", "type": "server_error"}})
            yield "data: [DONE]\n\n"
            return
        yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": _finish_reason(info)}],
                    "usage": _usage(info),
                    "dlm": {**release, "latency_seconds": round(time.perf_counter() - started, 3),
                            "forward_calls": info["forward_calls"], "warnings": info["warnings"]}})
        yield "data: [DONE]\n\n"
    finally:
        slot.release()


@app.post("/v1/chat/completions")
def chat_completions(body: ChatRequest, request: Request):
    state = request.app.state
    schedule = state.release["schedule"]
    try:
        dec = build_decode(body, state.model, state.tokenizer, schedule)
    except ValueError as exc:  # before the lock and before any byte is streamed
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    slot = Slot.take(state.generation_lock)
    if slot is None:
        raise HTTPException(status_code=503, detail="Model busy; retry shortly", headers={"Retry-After": "1"})
    head = {"id": "chatcmpl-" + uuid.uuid4().hex[:24], "created": int(time.time()),
            "model": state.release["model_version"]}
    release = {"schedule": schedule, "seed": body.seed or 0}
    started = time.perf_counter()
    if not body.stream:
        try:
            pieces = list(content_stream(dec, body.chunk, False))
        except Exception:
            log.exception("generation failed")
            raise HTTPException(status_code=500, detail="generation failed") from None
        finally:
            slot.release()
        info = pieces[-1][1]
        text = "".join(p for k, p in pieces if k == "content")
        return JSONResponse({**{k: head[k] for k in ("id", "created", "model")}, "object": "chat.completion",
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                          "finish_reason": _finish_reason(info)}],
                             "usage": _usage(info),
                             "dlm": {**release, "latency_seconds": round(time.perf_counter() - started, 3),
                                     "forward_calls": info["forward_calls"], "warnings": info["warnings"]}})
    # `slot` is released by the generator's finally, by the background task, and (last resort) by garbage collection.
    return StreamingResponse(_guarded(slot, content_stream(dec, body.chunk, body.diffusing), head, started, release),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                             background=BackgroundTask(slot.release))


# ================================================================ /demo: watch the text denoise in a browser
DEMO_CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; "
            "img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")

DEMO_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TinyStories DLM demo</title>
<link rel="icon" href="data:,">
<style>
:root{--bg:#faf9f7;--card:#fff;--fg:#1c1917;--muted:#6b6560;--line:#e4e0db;--accent:#2457d6;--accent-fg:#fff;--mask:#cfc9c2;--new:#15803d;--newbg:#dcfce7;--prompt:#8a837c;--err:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#0e0d0c;--card:#181614;--fg:#ece8e3;--muted:#a39c94;--line:#2c2825;--accent:#6b9bff;--accent-fg:#0e0d0c;--mask:#443f3a;--new:#86efac;--newbg:#14532d;--prompt:#8f8880;--err:#f87171}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:760px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:1.5rem;margin:0 0 4px;letter-spacing:-.01em}
.sub{margin:0 0 20px;color:var(--muted);font-size:.95rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
label{display:block;font-size:.82rem;font-weight:600;color:var(--muted);margin:0 0 4px}
textarea,input[type=number],select{width:100%;font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
textarea{resize:vertical;min-height:64px}
textarea:focus-visible,input:focus-visible,select:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:12px}
.check{display:flex;align-items:center;gap:8px;margin-top:12px;font-size:.92rem}
.check input{width:18px;height:18px}
.row{display:flex;gap:10px;align-items:center;margin-top:14px;flex-wrap:wrap}
button{font:inherit;font-weight:600;border:1px solid var(--line);border-radius:8px;padding:9px 18px;cursor:pointer;background:var(--card);color:var(--fg)}
button.primary{background:var(--accent);color:var(--accent-fg);border-color:var(--accent)}
button:disabled{opacity:.5;cursor:not-allowed}
#status{color:var(--muted);font-size:.9rem}
#status.err{color:var(--err)}
#out{margin-top:16px;min-height:11rem;font:1.05rem/1.7 Georgia,"Times New Roman",serif;white-space:pre-wrap;overflow-wrap:anywhere}
#out .prompt{color:var(--prompt)}
#out .mask{font-family:ui-monospace,Menlo,Consolas,monospace;color:var(--mask);animation:pulse 1.2s ease-in-out infinite}
#out .new{color:var(--new);background:var(--newbg);border-radius:3px;animation:fade .9s ease-out forwards}
@keyframes pulse{50%{opacity:.45}}
@keyframes fade{to{color:inherit;background:transparent}}
@media (prefers-reduced-motion:reduce){#out .mask,#out .new{animation:none}}
#stats{margin-top:12px;color:var(--muted);font-size:.85rem;min-height:1.3em}
.empty{color:var(--muted);font-family:system-ui,sans-serif;font-size:.95rem}
footer{margin-top:18px;color:var(--muted);font-size:.8rem}
.sr{position:absolute;left:-9999px;width:1px;height:1px;overflow:hidden}
</style>
</head>
<body>
<main>
<h1>TinyStories DLM</h1>
<p class="sub">A small masked-diffusion language model. It does not write left to right: the whole text starts as blocks and the words are filled in where the model is most confident.</p>

<section class="card" aria-label="Controls">
  <label for="prompt">Opening words (optional; leave empty for a story from scratch)</label>
  <textarea id="prompt" maxlength="2000" placeholder="Once upon a time"></textarea>
  <div class="grid">
    <div><label for="look">Look</label>
      <select id="look"><option value="block" title="best quality">Block by block</option><option value="whole" title="all of it denoises at once; a little less coherent">Whole text</option></select></div>
    <div><label for="len">Length (tokens)</label><input id="len" type="number" min="16" max="480" step="16" value="128"></div>
    <div><label for="seed">Seed (empty = random)</label><input id="seed" type="number" min="0" placeholder="random"></div>
    <div><label for="fps" id="fpsl">Speed: 16 frames/s</label><input id="fps" type="range" min="4" max="60" value="16"></div>
  </div>
  <label class="check"><input id="final" type="checkbox"> Show the final result only (no animation)</label>
  <div class="row">
    <button id="go" class="primary" type="button">Generate</button>
    <button id="stop" type="button" disabled>Stop</button>
    <button id="copy" type="button" disabled>Copy text</button>
    <span id="status" role="status"></span>
  </div>
</section>

<section class="card" style="margin-top:16px" aria-label="Result">
  <div id="out" aria-live="off"><span class="empty">The story will appear here.</span></div>
  <div id="stats"></div>
</section>
<div id="sr" class="sr" aria-live="polite"></div>
<footer id="foot"></footer>
</main>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
const MASK = "█";
let runId = 0, ctl = null, finalText = "";

function diffRange(a, b) {                       // the characters of b that differ from a (common prefix/suffix trimmed)
  const m = Math.min(a.length, b.length);
  let i = 0; while (i < m && a[i] === b[i]) i++;
  let j = 0; while (j < m - i && a[a.length - 1 - j] === b[b.length - 1 - j]) j++;
  return [i, b.length - j];
}

function span(cls, text) { const s = document.createElement("span"); if (cls) s.className = cls; s.textContent = text; return s; }

function paint(prompt, text, prev) {             // text comes from the model: only ever set as textContent
  const out = $("out"), frag = document.createDocumentFragment();
  if (prompt) frag.appendChild(span("prompt", prompt));
  const [a, b] = prev == null ? [0, 0] : diffRange(prev, text);
  let pos = 0;
  for (const part of text.split(/(█+)/)) {
    if (!part) continue;
    if (part[0] === MASK) frag.appendChild(span("mask", part));
    else {
      const lo = Math.max(pos, a), hi = Math.min(pos + part.length, b);
      if (lo < hi) {
        frag.appendChild(document.createTextNode(part.slice(0, lo - pos)));
        frag.appendChild(span("new", part.slice(lo - pos, hi - pos)));
        frag.appendChild(document.createTextNode(part.slice(hi - pos)));
      } else frag.appendChild(document.createTextNode(part));
    }
    pos += part.length;
  }
  out.replaceChildren(frag);
}

function setStatus(msg, err) { const s = $("status"); s.textContent = msg || ""; s.className = err ? "err" : ""; }
function busy(on) { $("go").disabled = on; $("stop").disabled = !on; }

function finish(prompt, text, info, started) {
  finalText = prompt + text;
  paint(prompt, text, null);
  $("copy").disabled = !text;
  $("sr").textContent = finalText;
  if (info) {
    const d = info.dlm || {}, u = info.usage || {};
    $("stats").textContent = `${u.completion_tokens ?? "?"} tokens · ${d.forward_calls ?? "?"} model calls · ` +
      `${((performance.now() - started) / 1000).toFixed(1)} s on screen · seed ${d.seed} · ` +
      `${info.choices[0].finish_reason === "stop" ? "the story ended" : "stopped at the length limit"}`;
  }
  setStatus("Done.");
  busy(false);
}

async function generate() {
  const my = ++runId;
  if (ctl) ctl.abort();
  ctl = new AbortController();
  const prompt = $("prompt").value, finalOnly = $("final").checked;
  const seed = $("seed").value === "" ? Math.floor(Math.random() * 1e9) : Number($("seed").value);
  const body = { messages: [{ role: "user", content: prompt }], max_completion_tokens: Number($("len").value) || 128,
                 seed, stream: !finalOnly, diffusing: !finalOnly };
  if ($("look").value === "whole") { body.block_length = 0; body.steps = 32; }
  const fps = Number($("fps").value), started = performance.now();
  busy(true); $("copy").disabled = true; $("stats").textContent = ""; $("sr").textContent = "";
  setStatus("Waiting for the model…");
  try {
    const res = await fetch("/v1/chat/completions", { method: "POST", headers: { "Content-Type": "application/json" },
                                                       body: JSON.stringify(body), signal: ctl.signal });
    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      throw new Error(res.status === 503 ? "The model is busy with another request. Try again in a moment."
                      : typeof e.detail === "string" ? e.detail : `HTTP ${res.status}`);
    }
    if (finalOnly) {                              // one JSON answer, no frames
      const j = await res.json();
      return my === runId && finish(prompt, j.choices[0].message.content, j, started);
    }
    const queue = []; let info = null, done = false, shown = null, n = 0;
    const take = (ev) => {                        // one SSE event: lines "data: {...}" or "data: [DONE]"
      for (const line of ev.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        if (line.slice(6) === "[DONE]") { done = true; continue; }
        const o = JSON.parse(line.slice(6));
        if (o.error) throw new Error(o.error.message || "generation failed");
        const c = o.choices && o.choices[0];
        if (c && typeof c.delta.content === "string" && c.delta.content !== "") queue.push(c.delta.content);
        if (c && c.finish_reason) info = o;
      }
    };
    (async () => {                                // reader: fill the queue as fast as the server sends
      const reader = res.body.getReader(), dec = new TextDecoder(); let buf = "";
      for (;;) {
        const { value, done: end } = await reader.read();
        if (end) break;
        buf += dec.decode(value, { stream: true });
        let i; while ((i = buf.indexOf("\n\n")) >= 0) { take(buf.slice(0, i)); buf = buf.slice(i + 2); }
      }
      done = true;
    })().catch((e) => { if (my === runId) { done = true; fail(e); } });
    const tick = () => {                          // player: one frame per tick; a long backlog is caught up by skipping frames
      if (my !== runId) return;
      if (queue.length) {
        const frames = queue.splice(0, 1 + Math.floor(queue.length / 30));
        const next = frames[frames.length - 1];
        paint(prompt, next, shown); shown = next; n++;
        setStatus(`Denoising… frame ${n}`);
      } else if (done) {
        return finish(prompt, shown || "", info, started);   // the last frame shown is the exact final text
      }
      setTimeout(tick, 1000 / fps);
    };
    tick();
  } catch (e) { if (my === runId) fail(e); }
}

function fail(e) {
  if (e && e.name === "AbortError") return;
  setStatus(e && e.message ? e.message : String(e), true);
  busy(false);
}

$("go").addEventListener("click", generate);
$("stop").addEventListener("click", () => { runId++; if (ctl) ctl.abort(); setStatus("Stopped."); busy(false); });
$("fps").addEventListener("input", () => { $("fpsl").textContent = `Speed: ${$("fps").value} frames/s`; });
$("copy").addEventListener("click", () => { if (navigator.clipboard) navigator.clipboard.writeText(finalText).then(() => setStatus("Copied."), () => {}); });
$("prompt").addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !$("go").disabled) generate(); });
fetch("/healthz").then((r) => r.json()).then((h) => {
  $("foot").textContent = `Model ${h.model_version} · ${h.schedule} schedule · ${h.device}${h.gpu ? " (" + h.gpu + ")" : ""} · context ${h.context_length} tokens`;
}).catch(() => {});
</script>
</body>
</html>
"""


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo():
    return HTMLResponse(DEMO_HTML, headers={"Content-Security-Policy": DEMO_CSP, "Cache-Control": "no-cache"})
