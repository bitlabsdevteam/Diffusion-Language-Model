"""Tests for the streaming endpoint in infra/serve.py (POST /v1/chat/completions). Run from DLM/:

    .venv/bin/python infra/test_serve.py

A real uvicorn server runs in a thread and is called with http.client / raw sockets (no httpx needed); the model is a
random-weight "tiny" one (text is noise: only the protocol, the parity with dlm.sample and the lock handling are tested).
Stub models with scripted logits cover what random weights cannot: an end token committed in the middle of a block, a
character split across tokens, and grad mode across threads.
"""
import http.client
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "infra")]

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from dlm.cleaning import clean_story  # noqa: E402
from dlm.data import HEADER  # noqa: E402
from dlm.model import DLMConfig, DiffusionTransformerLM  # noqa: E402
from dlm.sample import diffusion_generate, generate_text, load_checkpoint  # noqa: E402

PASS = FAIL = 0


def ok(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print("  FAIL:", msg)


# ---------------------------------------------------------------- the server
tmp = Path(tempfile.mkdtemp(prefix="serve_test_"))
torch.manual_seed(0)
DiffusionTransformerLM(DLMConfig.from_preset("tiny")).save_pretrained(tmp / "model")
shutil.copytree(ROOT / "tokenizer", tmp / "model" / "tokenizer")
(tmp / "model" / "deployment.json").write_text(json.dumps({"model_version": "test-v1", "schedule": "cosine", "code_ref": "abc"}))
os.environ.update(DLM_MODEL_DIR=str(tmp / "model"), DLM_DEVICE="cpu", DLM_CODE_REF="abc")
import serve  # noqa: E402
import uvicorn  # noqa: E402

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    PORT = s.getsockname()[1]
server = uvicorn.Server(uvicorn.Config(serve.app, host="127.0.0.1", port=PORT, log_level="error"))
threading.Thread(target=server.run, daemon=True).start()
for _ in range(100):
    if server.started:
        break
    time.sleep(0.1)
assert server.started, "server did not start"
MODEL, TOK = load_checkpoint(tmp / "model", "cpu")
MODEL.eval()
CHAT = "/v1/chat/completions"


def call(path, body, timeout=60):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    return r.status, dict(r.getheaders()), r.read()


def chat(body, **kw):
    st, h, raw = call(CHAT, body, **kw)
    return st, h, json.loads(raw) if raw else None


def stream(body):
    """(status, list of parsed events, headers); the last event is the string "[DONE]" when the stream ended properly."""
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=60)
    c.request("POST", CHAT, json.dumps({**body, "stream": True}), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        return r.status, [json.loads(r.read())], dict(r.getheaders())
    events = []
    while True:
        line = r.readline()
        if not line:
            break
        line = line.decode().strip()
        if line.startswith("data: "):
            events.append("[DONE]" if line[6:] == "[DONE]" else json.loads(line[6:]))
            if events[-1] == "[DONE]":
                break
    return r.status, events, dict(r.getheaders())


def user(text):
    return [{"role": "user", "content": text}]


def deltas(events):
    return [e["choices"][0]["delta"].get("content") for e in events[:-1] if e["choices"] and "content" in e["choices"][0]["delta"]][1:]


def base_text(prompt):
    return TOK.decode(TOK.encode(clean_story(prompt), add_special_tokens=False).ids, skip_special_tokens=True)


# ---------------------------------------------------------------- 1. /generate is untouched, chat matches the sampler
print("1. parity with dlm.sample")
st, _, raw = call("/generate", {"prompt": "", "max_new_tokens": 64, "seed": 3})
old = json.loads(raw)
ok(st == 200 and old["text"] == generate_text(MODEL, TOK, "", None, seed=3, max_new_tokens=64, schedule="cosine")["text"],
   "/generate still returns what generate_text returns")
cases = [dict(prompt="", seed=0), dict(prompt="", seed=1, block_length=0), dict(prompt="Once upon a time", seed=2),
         dict(prompt="Tom had a ball", seed=3, steps=12), dict(prompt="", seed=4, temperature=0.0),
         dict(prompt="Mia saw a cat", seed=5, temperature=1.3, block_length=16)]
for c in cases:
    ref = generate_text(MODEL, TOK, c["prompt"], None, seed=c["seed"], max_new_tokens=96, schedule="cosine",
                        steps=c.get("steps"), block_length=c.get("block_length", 32), temperature=c.get("temperature", 0.8))
    body = {"messages": user(c["prompt"]), "max_tokens": 96, "seed": c["seed"], "steps": c.get("steps"),
            "block_length": c.get("block_length", 32), "temperature": c.get("temperature", 0.8)}
    st, _, r = chat(body)
    ok(st == 200 and base_text(c["prompt"]) + r["choices"][0]["message"]["content"] == ref["text"], f"non-stream parity {c}")
    ok(r["usage"]["completion_tokens"] > 0 and r["model"] == "test-v1" and r["object"] == "chat.completion", f"shape {c}")
    for chunk in ("block", "prefix"):
        st, ev, h = stream({**body, "chunk": chunk})
        ok(st == 200 and ev[-1] == "[DONE]", f"{chunk}: stream ends with [DONE] {c}")
        ok(base_text(c["prompt"]) + "".join(d for d in deltas(ev) if d) == ref["text"], f"{chunk}: deltas concatenate to the sampler text {c}")
        ok(ev[0]["choices"][0]["delta"].get("role") == "assistant" and ev[0]["object"] == "chat.completion.chunk", "first chunk carries the role")
        last = ev[-2]
        ok(last["choices"][0]["finish_reason"] in ("stop", "length") and last["usage"]["completion_tokens"] == r["usage"]["completion_tokens"]
           and "forward_calls" in last["dlm"], "last chunk: finish_reason, usage, extras")
        ok(h["content-type"].startswith("text/event-stream") and h.get("cache-control") == "no-cache", "SSE headers")
    ok(len({e["id"] for e in ev[:-1]}) == 1, "one id for the whole stream")

# prefix streaming sends at least as many pieces as block streaming, and block streaming never sends partial blocks
_, evb, _ = stream({"messages": user(""), "max_tokens": 96, "seed": 0, "chunk": "block"})
_, evp, _ = stream({"messages": user(""), "max_tokens": 96, "seed": 0, "chunk": "prefix"})
ok(len(deltas(evp)) >= len(deltas(evb)) >= 1, "prefix chunks >= block chunks >= 1")

# ---------------------------------------------------------------- 2. diffusing
print("2. diffusing")
body = {"messages": user(""), "max_tokens": 64, "seed": 7}
r = chat(body)[2]
final = r["choices"][0]["message"]["content"]
st, ev, _ = stream({**body, "diffusing": True})
frames = deltas(ev)
ok(st == 200 and frames and frames[-1] == final, "the last diffusing frame is exactly the final text")
ok(all(f != "" for f in frames) and serve.MASK_GLYPH not in frames[-1], "the final frame has no mask glyph")
ok(serve.MASK_GLYPH in frames[0], "the first frame still has masked positions")
ok(len(frames) > len(deltas(stream({**body})[1])), "more frames than block chunks (one per denoising step)")
ok(len({tuple(f) for f in frames}) == len(frames), "no repeated frame")
st, h, r = chat({**body, "diffusing": True})
ok(st == 422 and "stream" in json.dumps(r), "diffusing without stream -> 422")

# ---------------------------------------------------------------- 3. request handling (OpenAI compatibility)
print("3. OpenAI-style requests")
st, _, r = chat({"model": "mercury-3", "messages": [{"role": "system", "content": "be nice"}, {"role": "user", "content": "Once upon a time"}],
                 "max_completion_tokens": 40, "max_tokens": 200, "top_p": 0.9, "stop": ["x"], "user": "u1", "stream_options": {"include_usage": True},
                 "frequency_penalty": 0.1})
ok(st == 200 and r["model"] == "test-v1", "unknown OpenAI fields are ignored; the served model version is echoed")
ok(r["usage"]["completion_tokens"] <= 40, "max_completion_tokens wins over max_tokens")
st, _, r = chat({"messages": [{"role": "user", "content": [{"type": "text", "text": "Once upon"}, {"type": "text", "text": " a time"}]}], "max_tokens": 8})
st2, _, r2 = chat({"messages": user("Once upon a time"), "max_tokens": 8})
ok(st == 200 and r["choices"] == r2["choices"], "content parts are joined")
for bad, why in [({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]}, "last message not from the user"),
                 ({"messages": []}, "no messages"), ({"messages": user("hi"), "n": 2}, "n=2"),
                 ({"messages": user("hi"), "mode": "story"}, "story mode with opening words"),
                 ({"messages": user(""), "mode": "continue"}, "continue mode without opening words"),
                 ({"messages": user("x " * 3000)}, "prompt too long for the window"),
                 ({"messages": user("hi"), "max_tokens": 0}, "max_tokens 0"), ({"messages": user("hi"), "temperature": 3}, "temperature 3")]:
    st, _, _ = chat(bad)
    ok(st == 422, f"422 for: {why} (got {st})")
    st, evs, _ = stream(bad)
    ok(st == 422, f"422 before any byte is streamed for: {why}")
st, _, r = chat({"messages": user("hi"), "max_tokens": 4})
ok(st == 200, "no lock leaked after the 422s")

# ---------------------------------------------------------------- 4. scripted models: end token mid-block, UTF-8 split, threads
print("4. scripted model")
WORDS = TOK.encode(" the cat sat on a warm mat today", add_special_tokens=False).ids
END = TOK.token_to_id("<|end|>")


class Stub(torch.nn.Module):
    """Each response position p wants token token_at[p] (else a filler word) with a fixed confidence: later positions of a block
    first (order='rtl') or earlier ones first ('ltr'), whatever the context is."""

    def __init__(self, token_at, order="rtl", P=0):
        super().__init__()
        self.config = SimpleNamespace(seq_len=512, vocab_size=8192, mask_id=4, pad_id=0)
        self.w, self.token_at, self.order, self.P, self.grad_modes = torch.nn.Parameter(torch.zeros(1)), token_at, order, P, []

    def forward(self, ids):
        self.grad_modes.append(torch.is_grad_enabled())
        logits = torch.zeros(1, ids.shape[1], 8192)
        for p in range(self.P, ids.shape[1]):
            rel = p - self.P
            tok = self.token_at.get(rel, WORDS[rel % len(WORDS)])
            logits[0, p, tok] = 10 + (0.02 if self.order == "rtl" else -0.02) * (rel % 32)
        return logits


def decode_for(stub_tokens, order, prompt=""):
    prompt_ids = TOK.encode(clean_story(prompt), add_special_tokens=False).ids if prompt else []
    header = TOK.encode(HEADER, add_special_tokens=False).ids
    P = 1 + len(header) + len(prompt_ids)
    stub = Stub(stub_tokens, order, P)
    dec = serve.Decode(stub, TOK, prompt, None, max_new_tokens=96, steps=None, block_length=32, temperature=0.0, schedule="linear", seed=0)
    return stub, dec, header, prompt_ids


def drain(dec, chunk="prefix", diffusing=False):
    out = list(serve.content_stream(dec, chunk, diffusing))
    return [p for k, p in out if k == "content"], out[-1][1]


# 4a. an end token commits at position 5 while positions 0-4 are still masked (later positions first)
token_at = {5: END, 6: WORDS[0], 7: WORDS[1]}
for i in range(5):
    token_at[i] = WORDS[(i + 3) % len(WORDS)]
for chunk in ("block", "prefix"):
    stub, dec, header, prompt_ids = decode_for(token_at, "rtl")
    pieces, info = drain(dec, chunk)
    expect = TOK.decode([token_at[i] for i in range(5)], skip_special_tokens=True)
    ok("".join(pieces) == expect, f"{chunk}: nothing after the first end token is streamed (got {''.join(pieces)!r})")
    ok(info["truncated"] is False and len(info["story"]) == 5, f"{chunk}: story cut at the end token")
    ref = diffusion_generate(Stub(token_at, "rtl", len(dec.prefix)), TOK, header, prompt_ids, max_new_tokens=96, temperature=0.0, schedule="linear")
    ok(ref["text"] == expect, f"{chunk}: same text as dlm.sample.diffusion_generate")
    ok(info["forward_calls"] == ref["forward_calls"], f"{chunk}: same number of model calls")
# that run had the end token committed before the prefix was complete: a frame must not show anything after it
stub, dec, *_ = decode_for(token_at, "rtl")
frames, _ = drain(dec, "block", diffusing=True)
drop = next((i for i in range(1, len(frames)) if len(frames[i]) < len(frames[i - 1])), None)  # the step where the end token commits
ok(frames[-1] == expect, "diffusing: final frame is exactly the text before the end token")
ok(drop is not None and all(len(f) <= len(expect) + 5 * len(serve.MASK_GLYPH) for f in frames[drop:]),
   "diffusing: from the step the end token commits, nothing past it is shown")
ok(any(serve.MASK_GLYPH in f for f in frames[:-1]) and frames[-1].count(serve.MASK_GLYPH) == 0, "diffusing: placeholders disappear")

# 4b. a character split over several tokens is never sent half (left to right, so the prefix grows one token at a time)
emoji = TOK.encode(" ok \U0001F600 fine", add_special_tokens=False).ids
ok(len(emoji) > 2, "the emoji is split over several byte tokens")
at = {i: t for i, t in enumerate(emoji)}
at[len(emoji)] = END
stub, dec, *_ = decode_for(at, "ltr")
pieces, info = drain(dec, "prefix")
ok(pieces and all("�" not in p for p in pieces), f"no half character in any piece: {pieces!r}")
ok("".join(pieces) == " ok \U0001F600 fine" and len(pieces) >= 3, "pieces concatenate to the exact text")

# 4c. grad mode: Starlette resumes a sync generator on whatever pool thread is free, and grad mode is per thread
stub, dec, *_ = decode_for(at, "ltr")
gen = dec.events()
with ThreadPoolExecutor(max_workers=4) as ex:
    while True:
        ev = ex.submit(next, gen).result()
        if ev["kind"] == "done":
            break
gen.close()
ok(len(stub.grad_modes) >= 5 and not any(stub.grad_modes), f"forward ran without grad on every resume ({len(stub.grad_modes)} calls)")

# ---------------------------------------------------------------- 5. the lock: disconnects must never leave the server busy
print("5. lock handling")


class Slow(torch.nn.Module):
    def __init__(self, inner, delay):
        super().__init__()
        self.inner, self.config, self.delay, self.calls = inner, inner.config, delay, 0

    def forward(self, ids):
        self.calls += 1
        time.sleep(self.delay)
        return self.inner(ids)


real = serve.app.state.model
slow = Slow(real, 0.05)
serve.app.state.model = slow


def raw_request(body):
    s = socket.create_connection(("127.0.0.1", PORT), timeout=10)
    payload = json.dumps(body).encode()
    s.sendall(b"POST " + CHAT.encode() + b" HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
    return s


def recovers(seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        st, _, _ = chat({"messages": user(""), "max_tokens": 2})
        if st == 200:
            return time.time() - t0
        time.sleep(0.1)
    return None


long_body = {"messages": user(""), "max_tokens": 128, "stream": True, "seed": 1}
# 5a. client leaves right after sending the request (the stream may not have produced a byte)
slow.calls = 0
raw_request(long_body).close()
t = recovers(8)
ok(t is not None and t < 3, f"disconnect before the first byte: the next request succeeds after {t}")
time.sleep(0.5)
n_before = slow.calls
time.sleep(0.5)
ok(slow.calls == n_before and slow.calls < 100, f"generation stopped after the disconnect ({slow.calls} of 128 model calls)")
# 5b. client leaves in the middle of the stream
s = raw_request(long_body)
buf = b""
while b'"content":"' not in buf[buf.find(b"\r\n\r\n"):] and len(buf) < 100000:
    chunk = s.recv(4096)
    if not chunk:
        break
    buf += chunk
s.close()
slow.calls = 0
t = recovers(8)
ok(t is not None and t < 3, f"disconnect mid-stream: the next request succeeds after {t}")
time.sleep(0.5)
n_before = slow.calls
time.sleep(0.5)
ok(slow.calls == n_before, "no further model calls after the disconnect")
# 5c. a second request while one is streaming gets 503 with Retry-After; the first still finishes
res = {}


def run_long():
    res["long"] = stream({"messages": user(""), "max_tokens": 40, "seed": 2})


th = threading.Thread(target=run_long)
th.start()
time.sleep(0.5)
st, h, _ = chat({"messages": user(""), "max_tokens": 2})
ok(st == 503 and h.get("retry-after") == "1", f"busy -> 503 + Retry-After (got {st})")
st, h, _ = call("/generate", {"max_new_tokens": 2})
ok(st == 503, "/generate shares the same lock")
th.join()
ok(res["long"][0] == 200 and res["long"][1][-1] == "[DONE]", "the running stream was not disturbed")
ok(recovers(3) is not None, "lock free after the stream ended")
serve.app.state.model = real

# ---------------------------------------------------------------- 6. healthz
c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
c.request("GET", "/healthz")
hz = json.loads(c.getresponse().read())
ok(hz["status"] == "ready" and hz["schedule"] == "cosine", "/healthz unchanged")

server.should_exit = True
time.sleep(0.5)
import gc  # noqa: E402

gc.collect()  # close leftover generators now, not while the interpreter is tearing torch down
shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
