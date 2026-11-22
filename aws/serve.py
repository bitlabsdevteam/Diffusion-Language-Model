"""FastAPI wrapper around the DLM sampler (CPU build for AWS Fargate).

Loads the checkpoint once at startup, serves one generation at a time (a second concurrent request gets 503)
and refuses to start when the release and the container disagree (schedule, code revision, tokenizer ids).
"""
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from dlm.sample import generate_text, load_checkpoint


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
    device = os.environ.get("DLM_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DLM_DEVICE=cuda but CUDA is unavailable")
    threads = int(os.environ.get("DLM_THREADS", "0"))
    if threads > 0:  # Fargate reports the host's cores; use the vCPUs this task actually has
        torch.set_num_threads(threads)
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
    app.state.model = model.eval()
    app.state.tokenizer = tokenizer
    app.state.release = release
    app.state.device = device
    app.state.generation_lock = threading.Lock()
    yield


app = FastAPI(title="TinyStories DLM inference", lifespan=lifespan)


@app.get("/healthz")
def health(request: Request):
    state = request.app.state
    return {"status": "ready", "model_version": state.release["model_version"],
            "code_ref": state.release["code_ref"], "schedule": state.release["schedule"],
            "device": state.device, "context_length": state.model.config.seq_len}


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
