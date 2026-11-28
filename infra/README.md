# GPU server on AWS: tear-up and tear-down

One command creates, and one removes, an EC2 GPU server (**g4dn.xlarge, NVIDIA T4**) that serves the pre-trained DLM
from a Docker container (FastAPI + this project's own PyTorch sampler). Default region: **Osaka `ap-northeast-3`**.

```
Laptop -> SSM port forward (no inbound ports) -> EC2 127.0.0.1:8000 -> Docker (--gpus all) -> FastAPI -> dlm.sample on the T4
S3 versioned model folder -> downloaded and SHA-256 verified by the container at every start
git commit -> `git archive` -> linux/amd64 image -> ECR -> pulled by the instance
```

## Why not vLLM

vLLM and SGLang serve diffusion models with **block-causal** attention (finished blocks go into a KV cache). This model
was trained fully bidirectional and `dlm/sample.py` re-runs the whole window at every step, so those engines would
compute attention differently from training. vLLM has no entry for a LLaDA-1.0-style model. The best fit is the
project's own sampler behind FastAPI; speedups (CUDA graphs, batching, threshold commit) are a later, profiled step.
Details and sources: the plan this folder came from (`~/.claude/plans/you-will-plan-a-encapsulated-honey.md`).

## Before the first run (you)

1. `brew install --cask session-manager-plugin` (the tunnel and `smoke` need it).
2. In the AWS console (Osaka): Service Quotas -> EC2 -> **Running On-Demand G and VT instances** -> at least **4**.
   The deployer key cannot read or change quotas. Without it the launch fails with `VcpuLimitExceeded`.
3. A trained checkpoint folder (`final/` or an exported one). Without one, test the stack with
   `python infra/make_test_checkpoint.py --out /tmp/ckpt` (random weights, meaningless text).

## Use

```bash
cd DLM
infra/gpu.sh check                                   # read-only: credentials, permissions, AMI, subnet, quota hint

export MODEL_VERSION=cosine-step25000-v1
export LOCAL_MODEL_DIR=/absolute/path/to/pretrain_cosine/final
infra/gpu.sh up                                      # asks once; creates everything and launches the instance
infra/gpu.sh smoke                                   # tunnel, waits for /healthz, one greedy /generate
infra/gpu.sh tunnel                                  # keep a tunnel open on http://127.0.0.1:18000
infra/gpu.sh stop                                    # GPU billing ends, disk is kept (start resumes)
infra/gpu.sh down                                    # removes everything except the S3 bucket
infra/gpu.sh down --delete-models                    # ... and a script-created bucket with every release
```

`up` is idempotent (rerun after a failure). To roll out a new model use a **new** `MODEL_VERSION`, run `down` for the
instance (or terminate it) and `up` again: releases in S3 are immutable and the instance bootstraps once at first boot.
Set `DLM_DTYPE=fp16` only after a parity check against fp32 (a T4 has no bf16).

## What `up` creates (tagged `Project=dlm-gpu`; default VPC only; no VPC, NAT or load balancer)

| Resource | Name |
| --- | --- |
| S3 bucket (private, encrypted, **not versioned**) | `dlm-gpu-models-<account>-<region>`, or your `MODEL_BUCKET` (never deleted) |
| IAM role + instance profile | `dlm-gpu-instance-role` (SSM core, ECR read-only, read `models/tinystories/*`), `dlm-gpu-instance-profile` |
| ECR repository | `dlm-gpu` |
| Security group | `dlm-gpu-sg`, **no inbound rules** |
| EC2 instance | `dlm-gpu`, `g4dn.xlarge`, Deep Learning Base GPU AMI (Ubuntu 24.04, from the AWS SSM parameter), 100 GiB gp3 with delete-on-termination, public IPv4 (outbound only), IMDSv2 required with hop limit 2 |

The subnet is chosen in an Availability Zone that offers the instance type (in Osaka `g4dn.xlarge` is in 3b and 3c, not
3a) and the script tries the next AZ on `InsufficientInstanceCapacity`.

## Tear-down order and guarantees

`down`: terminate the instance and wait -> delete leftover tagged volumes -> security group (retries up to 5 minutes for
the network interface to release) -> instance profile (role removed first) -> IAM role (policies detached/deleted first)
-> ECR repository with its images -> optionally the bucket (`--delete-models`, only if the script created it, after
purging every version and delete marker). Each step tolerates "not found", so `down` can be rerun after a partial
failure. It ends with a leftover check using direct `describe` calls (the tagging API is denied for this key).

## Cost

The instance bills per second while running (g4dn.xlarge: $0.526/h us-east-1 and $0.736/h Singapore were verified
against AWS price data on 2026-11-21; **Osaka was not verified**, the Pricing API is denied for this key). Stopped, only
the 100 GiB gp3 disk bills (about $8 to $10 a month, region dependent). A public IPv4 address bills $0.005/h while the
instance runs. ECR and S3 storage are cents. Use `stop` when idle and `down` when done.

## What was tested, and what was not

Tested locally on this Mac, nothing created on AWS:
- `bash -n`; credential parsing on fake files (right section, quotes stripped, bad files rejected); user-data rendering.
- The image built for `linux/amd64` (emulated, 6.3 GB unpacked) and run on **CPU** against a random-weight release:
  `/healthz`, greedy `/generate`, `422` on an empty continue prompt and on `max_new_tokens: 0`, and refusals with the
  right message for a tampered weights file, a wrong code revision, `DLM_DEVICE=cuda` without a GPU, and `DLM_DTYPE=fp16`
  on CPU. The container runs as a non-root user.
- The base image `pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime` ships system Python 3.12 (PEP 668), hence
  `PIP_BREAK_SYSTEM_PACKAGES=1` in the Dockerfile. Its torch is `2.15.0+cu132` and `torch.cuda.get_arch_list()` contains
  **`sm_75`**, so the T4 is covered by the build (actual GPU execution is still untested).

Read-only AWS calls with the deployer key (earlier): default VPC, `g4dn.xlarge` offerings per AZ, AMI resolution (Osaka
`...20261120`, root volume 75 GiB gp3).

**Not tested on AWS:** `up`, `down`, `stop`, `start`, `smoke`, `tunnel`, the instance bootstrap (Docker/NVIDIA runtime on
the AMI, ECR login, image pull), the S3 download with the instance role, running CUDA on a real T4 with that image (the arch list is right; the host driver
620 / CUDA 13.5 on the AMI should support CUDA 13.2 per AWS docs, unverified) and the SSM tunnel (`ssm:StartSession` may be denied for the deployer key). A first real `up` then `down` is the test.

## API

| Endpoint | What it is |
| --- | --- |
| `GET /healthz` | readiness, model version, schedule, device |
| `POST /generate` | the original blocking endpoint (unchanged; `gpu.sh smoke` uses it) |
| `POST /v1/chat/completions` | OpenAI-compatible, modelled on Inception's Mercury API; blocking, streamed, or "diffusing" |

`/v1/chat/completions` request fields: `messages` (the **last** one must be `user`; its text is the opening words of the Story, empty
means a Story from scratch; earlier messages are ignored), `max_completion_tokens` or `max_tokens` (default 128, at most 480),
`temperature` (0.8), `seed` (0), `stream`, `diffusing`. Other OpenAI fields (`model`, `top_p`, `stop`, `user`, `stream_options`, ...) are
accepted and ignored; `n` above 1 is refused. Extensions: `chunk` (`block` default | `prefix`), `mode`, `steps`, `block_length`.
The reply's `model` is the deployed model version. The text does **not** echo the opening words (`prompt + content` equals `/generate`'s `text`).

```bash
# blocking
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Once upon a time"}],"max_completion_tokens":128}'

# streaming: Server-Sent Events; `delta.content` is new text to APPEND; ends with `data: [DONE]`
curl -sN localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":""}],"stream":true}'

# diffusing: every chunk is the WHOLE text so far (masked tokens drawn as a block): OVERWRITE the display on each chunk
curl -sN localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":""}],"stream":true,"diffusing":true}'
```

How the stream behaves:
- **Appendable (default):** a chunk arrives when a 32-token decoding block is finished (`chunk: "block"`), or as soon as the finished
  text at the front grows (`chunk: "prefix"`). Committed tokens never change, so nothing is ever retracted, and a half UTF-8 character
  is held back until the next chunk. Chunks come in bursts, like Mercury's.
- **Diffusing:** one chunk per denoising step; the last chunk is exactly the final text.
- The last chunk has `finish_reason` (`stop` = the Story ended, `length` = the token limit came first), `usage`, and a non-standard
  `dlm` object (`schedule`, `seed`, `latency_seconds`, `forward_calls`, `warnings`).
- Errors before the first byte are normal HTTP errors (`422` bad request, `503` busy with `Retry-After`). An error after streaming
  started is sent as a final `data: {"error": ...}` event followed by `[DONE]`.
- If the client disconnects, generation stops and the lock is released (tested before the first byte and mid-stream).
- Same seed, same text: the streamed text equals `/generate`'s for the same settings (tested against `dlm.sample`).
- All of it lives in `serve.py` because the image build copies only `serve.py` and `entrypoint.py` from this folder. An image for a
  code revision that is already in ECR is not rebuilt with a new `serve.py`: run `FORCE_BUILD=1 infra/gpu.sh up`.

Tests (real uvicorn server in a thread, random-weight tiny model, scripted stub models; no GPU, no AWS): `.venv/bin/python infra/test_serve.py`.
Not tested: a real client library (the OpenAI SDK), a real GPU, the real checkpoint, or a proxy/ingress that buffers SSE.

## Known limits

- One generation at a time (a second concurrent request gets `503`); no HTTPS, no login (private, tunnel-only access).
- Streaming sends text as it becomes final, not word by word, and the base model does not follow instructions: the last user message
  is the *opening words* of the Story (see below). A streamed request holds the one generation lock until it ends or the client leaves.
- No trained checkpoint exists yet; text quality and real latency are unmeasured. Latency work (profile, threshold
  commit, CUDA graphs, batching) comes after the baseline is running.
- `deploy.md` describes the same design by hand; this folder supersedes its provisioning steps.
