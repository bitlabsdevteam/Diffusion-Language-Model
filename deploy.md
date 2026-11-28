# Deploy the DLM on AWS: S3 + EC2 inference

Step-by-step guide for serving a trained TinyStories DLM with **Amazon S3, one GPU EC2 instance, Docker, and FastAPI**.

**Status:** deployment instructions prepared on 2026-11-22. AWS provisioning, GPU/container validation, and testing with the real checkpoint will happen when AWS access and the checkpoint are available. The Python examples use the existing `dlm.sample.load_checkpoint()` and `generate_text()` interfaces.

**Local verification:** checked all 24 shell blocks for syntax, parsed the IAM JSON and Docker command, and executed the artifact preparation/checksum and FastAPI examples against synthetic tiny CPU checkpoints in both `model.pt` and `model.safetensors` layouts. Verified generation parity with the existing sampler, request validation, busy responses, and rejection of mismatched schedules, checksums, and code revisions. These checks do not measure the real model's quality or EC2 performance.

“AWS Cloud Storage” is assumed to mean **Amazon S3**. Commands marked **Laptop** run on your Mac; commands marked **EC2** run on the Ubuntu instance. Run each numbered step in order. Replace values in angle brackets before executing commands.

## 1. Deployment approach

The first deployment provides a personal inference endpoint through an AWS Systems Manager (SSM) tunnel:

```text
Laptop / local application
  -> SSM port-forwarding tunnel
  -> EC2 localhost:8000
  -> FastAPI container
  -> existing PyTorch diffusion sampler
  -> model weights on the GPU

S3 versioned model folder -> EC2 EBS disk -> model loaded once at startup
GitHub code revision     -> Docker image -> inference application
```

- **S3** stores the weights, tokenizer, configuration, and release metadata. It does not execute the model. EC2 downloads the artifact before starting inference; individual requests do not fetch it from S3.
- **EC2** provides the GPU and runs the API. **EBS** is the attached disk holding Docker images and the downloaded model.
- The API initially supports **story generation and continuation**, matching pretraining. Instruction following is a later SFT phase.
- Use the existing PyTorch implementation for this deployment. vLLM v0.34.0 has model-specific diffusion support, but this project's custom `DiffusionTransformerLM` needs an additional integration before it can use vLLM.
- The service runs one model process and one generation at a time. Concurrent generation requests receive `503` and can retry; this version does not implement dynamic batching. Token streaming was added after this guide was written: `infra/serve.py` now also serves `POST /v1/chat/completions` (see `infra/README.md`), so the inline `serve.py` below is out of date.

## 2. Values to choose when provisioning

| Value | Starting choice / information needed |
| --- | --- |
| AWS account and access | Configure a local AWS CLI profile when provisioning begins |
| Region | Region of the existing S3 bucket, with GPU instance availability |
| S3 bucket | Existing bucket name, or a new bucket in the same region as EC2 |
| Model prefix | `models/tinystories/<MODEL_VERSION>/` |
| Model version | Unique release name, e.g. `cosine-step25000-v1`; label an intermediate checkpoint with its actual step |
| Checkpoint source | Local copy of the trained run's `final/`, `best/`, or exported checkpoint folder |
| Noise schedule | The selected run's `linear` or `cosine` schedule |
| Code revision | Full Git commit SHA compatible with the checkpoint and pushed to GitHub |
| Instance | Start with `g6.xlarge` (one NVIDIA L4 GPU), subject to regional capacity and quota |
| AMI | Current AWS Deep Learning **NVIDIA GPU** AMI, **Ubuntu, x86_64**, with a driver compatible with the container below |
| Disk | Start with 100 GiB gp3 EBS, or the AMI's larger required minimum |
| Access | SSM shell and port forwarding; the initial endpoint is accessible from the laptop through the tunnel |

The GPU choice is a starting point, not a measured performance requirement. Record the selected instance's hourly price, disk price, AMI ID, and regional G-instance vCPU quota during provisioning. A CPU deployment can be evaluated later against the measured latency target.

**Laptop — from the local `DLM/` repository root:**

```bash
export AWS_PROFILE="dlm-deploy"
export AWS_REGION="<your-aws-region>"
export MODEL_BUCKET="<your-s3-bucket>"
export MODEL_VERSION="cosine-step25000-v1"
export DLM_SCHEDULE="cosine"
export LOCAL_MODEL_DIR="/absolute/path/to/pretrain_cosine/final"
export DLM_CODE_REF="$(git rev-parse HEAD)"
export MODEL_URI="s3://${MODEL_BUCKET}/models/tinystories/${MODEL_VERSION}"
```

Keep these values for later steps. Environment variables are local to a terminal: a new terminal or an SSM session needs its own values. `DLM_CODE_REF` should identify the code that was checked with the selected weights, not an arbitrary newer revision.

**Schedule matters:** the newer Colab notebook defaults to cosine, while the older `pretrain_base` run used linear. Check `run_info.json` under `recipe.schedule`, or the training configuration for a `best/` checkpoint. The CLI sampler defaults to linear and does not automatically read the training schedule.

## 3. Configure AWS access

**When provisioning starts, on the laptop:**

1. Install AWS CLI v2 and the AWS Session Manager plugin using the official links in References.
2. Configure the `dlm-deploy` profile using the account's sign-in method. For IAM Identity Center / SSO:

   ```bash
   aws configure sso --profile dlm-deploy
   aws sso login --profile dlm-deploy
   ```

   If access is supplied through a different credential mechanism, configure that named profile instead. The later commands use the same profile. Keep credential values in the local AWS credential mechanism, outside this document and the repository.

3. Confirm the account and identity:

   ```bash
   aws --version
   session-manager-plugin --version
   aws sts get-caller-identity --profile "$AWS_PROFILE" --region "$AWS_REGION"
   ```

4. The provisioning identity needs permissions to manage the chosen EC2 instance, security group, IAM instance role/profile, and S3 prefix. It also needs `iam:PassRole` for the instance role, S3 upload/list access, and permission to start/terminate SSM sessions, including sessions using `AWS-StartPortForwardingSession`. Existing-account policies and boundaries will determine the exact policy.

The **EC2 instance uses its own IAM role** to download the model. Laptop credentials are not copied into the Docker image or onto the instance.

## 4. Prepare the model artifact

**Laptop:** bring the trained checkpoint from its current location, such as Colab Drive, to the folder in `LOCAL_MODEL_DIR`. A runnable training checkpoint has this layout:

```text
final/
  config.json
  model.pt
  tokenizer/
    tokenizer.json
    tokenizer_config.json
    tokenizer_manifest.json
  run_info.json                 # when available
  export.json                   # when exported from an unfinished run
```

The loader also supports a Hub-style folder with `model.safetensors`, `config.json`, and `tokenizer.json` at its top level. Use the **tokenizer trained with these weights**. If both weight formats are present, this loader selects `model.pt`.

If the only artifact is an unfinished run's **`resume.pt`**, export it first using the existing project environment:

```bash
.venv/bin/python run.py export \
  --resume "/absolute/path/to/run/resume.pt" \
  --out "/absolute/path/to/run/export_for_ec2" \
  --tokenizer-dir "/absolute/path/to/matching/tokenizer"
export LOCAL_MODEL_DIR="/absolute/path/to/run/export_for_ec2"
```

An export makes the checkpoint loadable; it does not finish training. Choose a model version name that reflects the exported training step.

Run one baseline sample before uploading:

```bash
.venv/bin/python run.py generate \
  --ckpt "$LOCAL_MODEL_DIR" --device cpu \
  --mode continue --prompt "Once upon a time, a little fox" \
  --max-new-tokens 64 --steps 64 --block-length 32 \
  --temperature 0 --seed 0 --schedule "$DLM_SCHEDULE"
```

Save the output for comparison with EC2. A working endpoint does not establish model quality; use the selected run's evaluation results and sample text as well. CPU and GPU floating-point differences can change generated text, so cross-device equality is not the acceptance criterion.

Create a separate upload folder with the model files and a checksum manifest. The following block validates the loader, tokenizer vocabulary/special IDs, and any recorded training schedule before copying the artifact.

```bash
export ARTIFACT_DIR="$(mktemp -d)"
.venv/bin/python - <<'PY'
import hashlib
import json
import os
import shutil
from pathlib import Path

from dlm.sample import load_checkpoint

source = Path(os.environ["LOCAL_MODEL_DIR"]).resolve()
target = Path(os.environ["ARTIFACT_DIR"])
schedule = os.environ["DLM_SCHEDULE"]
assert schedule in {"linear", "cosine"}
model, tok = load_checkpoint(source, "cpu")
assert tok.get_vocab_size() == model.config.vocab_size, "Tokenizer vocabulary mismatch"
assert tok.token_to_id("[MASK]") == model.config.mask_id, "MASK ID mismatch"
assert tok.token_to_id("[PAD]") == model.config.pad_id, "PAD ID mismatch"
info = source / "run_info.json"
if info.is_file():
    recorded = json.loads(info.read_text()).get("recipe", {}).get("schedule")
    assert recorded is None or recorded == schedule, "Schedule differs from the training run"

names = ["config.json", "model.pt" if (source / "model.pt").is_file() else "model.safetensors",
         "run_info.json", "export.json", "generation_config.json"]
tokenizer_source = source / "tokenizer" if (source / "tokenizer" / "tokenizer.json").is_file() else source
(target / "tokenizer").mkdir()
for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer_manifest.json"):
    if (tokenizer_source / name).is_file():
        shutil.copy2(tokenizer_source / name, target / "tokenizer" / name)
for name in names:
    if (source / name).is_file():
        shutil.copy2(source / name, target / name)

hashes = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(target.rglob("*")) if p.is_file()}
release = {"model_version": os.environ["MODEL_VERSION"], "code_ref": os.environ["DLM_CODE_REF"],
           "schedule": schedule, "sha256": hashes}
(target / "deployment.json").write_text(json.dumps(release, indent=2) + "\n")
print(f"Validated artifact: {target}")
PY
```

**Expected:** the baseline produces text and the preparation block prints `Validated artifact`. This folder contains inference artifacts, not the dataset or optimizer state.

## 5. Upload the artifact to S3

**AWS console:**

1. Open **S3**, select the existing bucket, and note its region. If creating a bucket, choose a general-purpose bucket in the EC2 region.
2. Keep **Block all public access** enabled; the instance accesses S3 through IAM.
3. For a new bucket, use the default SSE-S3 encryption and enable bucket versioning. If the existing bucket uses a customer-managed KMS key, include that key in the IAM setup in Step 6.
4. Use a **new model-version prefix for every release**. Do not replace the files under a release already being served.

**Laptop:** upload the artifact, with `deployment.json` last as the release-complete marker:

```bash
aws s3 sync "$ARTIFACT_DIR/" "$MODEL_URI/" \
  --exclude "deployment.json" --region "$AWS_REGION" --profile "$AWS_PROFILE"
aws s3 cp "$ARTIFACT_DIR/deployment.json" "$MODEL_URI/deployment.json" \
  --region "$AWS_REGION" --profile "$AWS_PROFILE"
aws s3 ls "$MODEL_URI/" --recursive --summarize \
  --region "$AWS_REGION" --profile "$AWS_PROFILE"
```

Run the second command only after the sync succeeds. Confirm the listing contains the weights, configuration, tokenizer, and `deployment.json`.

If the model files are already in S3, download that checkpoint to `LOCAL_MODEL_DIR`, then perform Step 4 and publish a new versioned artifact prefix. This gives the deployment a known file layout, code revision, and checksum manifest.

## 6. Create the EC2 instance role

**AWS console → IAM → Roles → Create role:**

1. Trusted entity: **AWS service**. Use case: **EC2**.
2. Attach the AWS-managed **`AmazonSSMManagedInstanceCore`** policy.
3. Name the role **`dlm-ec2-inference-role`**.
4. Add an inline policy named **`ReadDlmModelArtifacts`**. Replace `<your-s3-bucket>` in this JSON with the bucket name:

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": "s3:ListBucket",
         "Resource": "arn:aws:s3:::<your-s3-bucket>",
         "Condition": {
           "StringLike": {
             "s3:prefix": ["models/tinystories", "models/tinystories/*"]
           }
         }
       },
       {
         "Effect": "Allow",
         "Action": "s3:GetObject",
         "Resource": "arn:aws:s3:::<your-s3-bucket>/models/tinystories/*"
       }
     ]
   }
   ```

5. If the artifact is encrypted with a **customer-managed KMS key**, grant this role `kms:Decrypt` on that key and allow the role in the key policy. For a cross-account bucket, its bucket policy must also permit the role.

Creating the EC2 role through the IAM console creates the associated instance profile. Select that profile in the next step.

## 7. Launch and connect to the GPU instance

**AWS console → EC2 → Launch instance:**

1. Name: **`dlm-inference`**; region: the selected region.
2. Select a current, AWS-published **Deep Learning NVIDIA GPU AMI for Ubuntu, x86_64**. Record the AMI ID and driver version. The example container uses PyTorch 2.15.0 / CUDA 13.2; choose an AMI whose driver supports it. The container GPU probe in Step 10 is the compatibility check.
3. Instance type: **`g6.xlarge`**, using **On-Demand** for the first deployment.
4. Network: an existing/default VPC with a public subnet, an Internet Gateway route, and auto-assigned public IPv4. This provides outbound access to SSM, S3, GitHub, and container/package registries. An existing private-subnet setup can be used if it already provides the required outbound connectivity.
5. Security group: create **`dlm-inference-sg`**, with **no inbound rules** and outbound connectivity enabled. SSM opens the management connection from the instance; the API will bind to host loopback.
6. Key pair: proceed without a key pair for this SSM-based setup.
7. Storage: **100 GiB gp3 EBS**, or the AMI's larger required minimum.
8. Advanced details → IAM instance profile: **`dlm-ec2-inference-role`**.
9. Ensure the chosen Ubuntu image has SSM Agent running. AWS Ubuntu images commonly provide it as a snap; if using that image family, this user-data script installs it when absent and starts it:

   ```bash
   #!/bin/bash
   set -e
   snap list amazon-ssm-agent >/dev/null 2>&1 || snap install amazon-ssm-agent --classic
   snap start amazon-ssm-agent
   ```

10. Launch, wait for status checks and SSM registration, and record the **instance ID**.
11. Select the instance → **Connect → Session Manager → Connect**.

**EC2 — in the SSM shell:** switch to the Ubuntu account so all later paths use the same home directory:

```bash
sudo -iu ubuntu
nvidia-smi
```

**Expected:** an NVIDIA L4 is listed and the driver is loaded. The AWS account may need a G-instance quota increase before the launch succeeds.

## 8. Download the model and application code

**EC2 — as `ubuntu`:**

```bash
export AWS_REGION="<your-aws-region>"
export MODEL_BUCKET="<your-s3-bucket>"
export MODEL_VERSION="cosine-step25000-v1"
export DEPLOY_ROOT="$HOME/dlm-deploy"
export MODEL_DIR="$DEPLOY_ROOT/models/$MODEL_VERSION"
export MODEL_URI="s3://${MODEL_BUCKET}/models/tinystories/${MODEL_VERSION}"
unset AWS_PROFILE
mkdir -p "$MODEL_DIR"
cd "$DEPLOY_ROOT"

aws --version
aws sts get-caller-identity --region "$AWS_REGION"
aws s3 sync "$MODEL_URI/" "$MODEL_DIR/" --region "$AWS_REGION"
```

Use AWS CLI v2 on the host; install it from the official Linux instructions if the selected AMI does not include it. The identity should be the EC2 role. No `aws configure` step is needed on EC2.

Verify the downloaded files before loading the model:

```bash
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["MODEL_DIR"])
release = json.loads((root / "deployment.json").read_text())
assert release["model_version"] == os.environ["MODEL_VERSION"]
assert release["schedule"] in {"linear", "cosine"}
for name, expected in release["sha256"].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert actual == expected, f"Checksum mismatch: {name}"
print("Artifact checksums verified:", release["model_version"], release["schedule"], release["code_ref"])
PY
```

Clone the code and check out the recorded revision:

```bash
export DLM_CODE_REF="$(python3 -c 'import json,os; print(json.load(open(os.environ["MODEL_DIR"] + "/deployment.json"))["code_ref"])')"
git clone https://github.com/bitlabsdevteam/Diffusion-Language-Model.git "$DEPLOY_ROOT/app"
git -C "$DEPLOY_ROOT/app" checkout --detach "$DLM_CODE_REF"
```

The repository's root contains `dlm/` and `run.py`; there is no additional `DLM/` directory inside this clone. For later deployments, reuse the clone and `git fetch origin` before checking out a new recorded revision.

## 9. Create the inference service

**EC2:** create the following three files in **`$DEPLOY_ROOT`**, alongside `app/` and `models/`. These are complete templates to create during provisioning; they are not existing application files in the repository.

### 9.1 `serve.py`

This wrapper loads the checkpoint once, preserves the existing sampler, exposes readiness, and returns `503` while another generation is running. The schedule is read from `deployment.json`.

```python
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
    device = os.environ.get("DLM_DEVICE", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the host driver and Docker GPU runtime")
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
```

`text` includes the supplied opening words in continuation mode. Token limits are also constrained by the model's context window, including the fixed training header. `steps: null` preserves the sampler's one-token-per-step default; a smaller explicit value can improve latency but must be checked for quality. The sampler distributes steps across decoding blocks, so `forward_calls` is the actual work performed.

### 9.2 `Dockerfile`

The image tag below exists and matches the local project's PyTorch major/minor version at the time this guide was written. GPU operation and the real checkpoint still need to be tested on the selected EC2 AMI.

```dockerfile
FROM pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 DLM_MODEL_DIR=/model DLM_DEVICE=cuda
ARG CODE_REF
ENV DLM_CODE_REF=${CODE_REF}
RUN python -m pip install --no-cache-dir \
    "fastapi>=0.115,<1" "uvicorn>=0.30,<1" \
    "tokenizers==0.23.3" "safetensors>=0.5,<1" "numpy>=1.26,<3"
COPY app/dlm /app/dlm
COPY app/run.py /app/run.py
COPY serve.py /app/serve.py
RUN python -m pip freeze > /app/runtime-packages.txt
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"
CMD ["python", "-m", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

Only inference dependencies are installed. Training dependencies such as `datasets`, `transformers`, and `pyarrow` are not needed by the checkpoint loader and sampler. The resolved package versions are recorded inside the built image; a later rebuild can resolve different versions within these ranges, so retain the built image and its ID for rollback.

### 9.3 `.dockerignore`

```text
models/
app/.git/
app/.venv/
app/data/
app/checkpoints/
**/__pycache__/
**/*.pyc
```

The model is mounted separately at runtime, so replacing weights does not require baking them into an image.

## 10. Build and start the container

**EC2 — from `$DEPLOY_ROOT`:** verify Docker and the NVIDIA Container Toolkit are available on the chosen DLAMI:

```bash
sudo docker --version
nvidia-ctk --version
sudo systemctl enable --now docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

If either tool is missing, install Docker Engine using its Ubuntu instructions and NVIDIA Container Toolkit using NVIDIA's Ubuntu instructions in References, then repeat this block. The GPU driver itself comes from the selected DLAMI.

Run a CUDA operation **inside the intended base container**, not just `nvidia-smi` on the host:

```bash
sudo docker run --rm --gpus all \
  pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime \
  python -c 'import torch; assert torch.cuda.is_available(); x=torch.ones(8, device="cuda"); print(torch.__version__, torch.cuda.get_device_name(0), x.sum().item())'
```

**Expected:** PyTorch version, GPU name, and `8.0`. Resolve any driver/runtime incompatibility before proceeding.

Build the application, then test the real checkpoint through the CLI in that image:

```bash
cd "$DEPLOY_ROOT"
export IMAGE_TAG="dlm-inference:${MODEL_VERSION}"
export DLM_SCHEDULE="$(python3 -c 'import json,os; print(json.load(open(os.environ["MODEL_DIR"] + "/deployment.json"))["schedule"])')"
sudo docker build --build-arg CODE_REF="$DLM_CODE_REF" -t "$IMAGE_TAG" .

sudo docker run --rm --gpus all \
  --mount "type=bind,src=$MODEL_DIR,dst=/model,readonly" \
  "$IMAGE_TAG" python run.py generate \
  --ckpt /model --device cuda \
  --mode continue --prompt "Once upon a time, a little fox" \
  --max-new-tokens 64 --steps 64 --block-length 32 \
  --temperature 0 --seed 0 --schedule "$DLM_SCHEDULE"
```

If generation succeeds, start the API:

```bash
sudo docker run -d --name dlm-api \
  --restart unless-stopped --gpus all \
  --publish 127.0.0.1:8000:8000 \
  --mount "type=bind,src=$MODEL_DIR,dst=/model,readonly" \
  --log-opt max-size=10m --log-opt max-file=3 \
  "$IMAGE_TAG"

sudo docker logs --tail 100 dlm-api
curl --fail-with-body http://127.0.0.1:8000/healthz
```

Wait for Uvicorn's startup-complete log before the health request. The response should report the expected model version, schedule, code revision, and `"device":"cuda"`. Keep **one Uvicorn worker**: more workers would each load their own model and have separate concurrency locks.

Record the deployed image and package versions:

```bash
sudo docker inspect dlm-api --format '{{.Image}}'
sudo docker exec dlm-api python -m pip freeze
```

The restart policy restarts an exited container and restores it after Docker/instance restarts. Docker's `unhealthy` status alone does **not** automatically restart a stuck container; inspect the logs and restart it explicitly if necessary.

## 11. Test the endpoint from your laptop

**Laptop — terminal 1:** restore the profile and region variables from Step 2, set the instance ID, and open a tunnel:

```bash
export INSTANCE_ID="<your-ec2-instance-id>"
aws ssm start-session \
  --target "$INSTANCE_ID" \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8000"],"localPortNumber":["8000"]}' \
  --region "$AWS_REGION" --profile "$AWS_PROFILE"
```

Keep this terminal running. The endpoint is now available at `http://127.0.0.1:8000` **on your laptop**.

**Laptop — terminal 2:**

```bash
curl --fail-with-body http://127.0.0.1:8000/healthz

curl --fail-with-body --max-time 300 http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"mode":"continue","prompt":"Once upon a time, a little fox","max_new_tokens":64,"steps":64,"temperature":0,"seed":0}'

curl --fail-with-body --max-time 300 http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"mode":"story","prompt":"","max_new_tokens":128,"temperature":0.8,"seed":42}'
```

**Expected:** `200` responses containing text, the release name, schedule, measured latency, and model-forward count. `truncated: true` means an end token was not produced in the requested generation window. Any sampler collapse warnings are returned in `warnings`.

Validate these behaviours before using the endpoint:

1. Readiness reports the exact artifact and code revision selected.
2. CLI and API calls with the same parameters on the same GPU produce matching results for the greedy smoke test.
3. Empty `continue` prompts and requests with `max_new_tokens: 0` return `422`.
4. A request arriving while a generation is running receives `503` with `Retry-After: 1`.
5. After a container restart, the same release loads again.
6. Record warm-request latency for 64 and 128 tokens, then compare `steps: null` against a smaller step count for both speed and sample quality.

This completes the **SSM-accessible personal endpoint**. The URL works while the tunnel is open. If the intended client needs an always-reachable HTTPS endpoint, choose its domain and network access during provisioning, then add an HTTPS ingress and application authentication rather than relying on a laptop tunnel.

## 12. Operate, update, and stop the deployment

### Inspect and restart

**EC2:**

```bash
sudo docker ps
sudo docker logs --tail 100 dlm-api
nvidia-smi
df -h
sudo docker restart dlm-api
```

Check `/healthz` after a restart. The runtime does not need the training dataset, and it can continue serving from EBS without downloading weights again for every request.

### Update a model or code release

1. Repeat Steps 2, 4, and 5 on the laptop using a **new** model version and the intended code revision.
2. On EC2, download it to a **new** `$DEPLOY_ROOT/models/<new-version>/` directory and verify checksums using Step 8. Keep the old model directory.
3. Fetch and check out the new recorded code revision. Rebuild/tag an image for the new release if needed; retain the old image.
4. Run the new image's CLI smoke test against the new model using Step 10.
5. After the smoke test succeeds, replace the running container. This initial single-instance process has a short interruption:

   ```bash
   sudo docker stop dlm-api
   sudo docker rename dlm-api dlm-api-previous
   ```

6. Repeat the API `docker run` command in Step 10 with `IMAGE_TAG` and `MODEL_DIR` pointing to the new release, then repeat Step 11's health and generation checks. If `dlm-api-previous` already exists from an earlier update, give that older container a distinct archival name before Step 5.
7. If the new release fails, restore the previous container, whose original image, environment, and model mount are retained:

   ```bash
   sudo docker stop dlm-api
   sudo docker rm dlm-api
   sudo docker rename dlm-api-previous dlm-api
   sudo docker start dlm-api
   ```

   If creation of the new container failed entirely, skip its stop/remove commands. Check `/healthz` after rollback.

### Stop compute when it is not needed

**Laptop:**

```bash
aws ec2 stop-instances --instance-ids "$INSTANCE_ID" \
  --region "$AWS_REGION" --profile "$AWS_PROFILE"
```

Stopping the instance ends serving and GPU compute billing; EBS, S3, and any other retained billable resources continue to incur their storage/resource charges. Stopping only the Docker container does not stop EC2 billing.

To resume:

```bash
aws ec2 start-instances --instance-ids "$INSTANCE_ID" \
  --region "$AWS_REGION" --profile "$AWS_PROFILE"
```

Wait for EC2 and SSM readiness, reopen the tunnel, and check `/healthz`. The `unless-stopped` container restarts with Docker unless it was explicitly stopped. For permanent retirement, terminate the instance and review its EBS delete-on-termination setting; retain the S3 releases needed for future redeployment.

## 13. Troubleshooting

| Symptom | Check / action |
| --- | --- |
| GPU instance cannot launch | Regional G-instance vCPU quota, selected AMI architecture, and availability-zone capacity |
| Instance missing from SSM | Instance profile, SSM Agent status, region, and outbound HTTPS connectivity; Ubuntu snap service is typically `snap.amazon-ssm-agent.amazon-ssm-agent.service` |
| Laptop SSM port forwarding fails | Local Session Manager plugin, profile login, permission for `AWS-StartPortForwardingSession`, instance ID, and whether local port 8000 is already occupied |
| S3 `AccessDenied` | EC2 role's bucket/prefix policy, existing bucket policy, and KMS key policy if applicable |
| Missing `deployment.json` | Upload has not completed or the wrong release prefix was selected |
| Checksum mismatch | Re-download into a fresh model-version directory; check whether the release was overwritten |
| `CUDA is unavailable` / container GPU probe fails | Host driver, container CUDA compatibility, NVIDIA Container Toolkit configuration, and `--gpus all` |
| Checkpoint does not load | `MODEL_DIR` must contain the weights and `config.json` directly; export `resume.pt` first; verify the recorded code revision |
| Tokenizer mismatch | Use the tokenizer that trained these weights, not a separately retrained tokenizer |
| Wrong schedule or unexpected output | `deployment.json` schedule versus the run's `recipe.schedule`; generation does not infer it automatically |
| Out of GPU memory | Confirm one worker, inspect other GPU processes, and reduce `max_new_tokens`; this service does not batch requests |
| HTTP `422` | Request JSON, prompt/mode combination, parameter bounds, or a prompt that exceeds the model's context window |
| HTTP `503` | Another generation is running; retry with backoff |
| Output is repetitive or incoherent | Inspect training/evaluation results and sampler warnings; endpoint health does not indicate that the model trained successfully |
| A stopped/restarted instance has no API | Check Docker startup, container status, the retained EBS model path, and whether the container was manually stopped |

## References

- [AWS CLI v2 installation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- [AWS CLI SSO configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html)
- [Install the Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)
- [Session Manager sessions and port forwarding](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-sessions-start.html)
- [AWS Deep Learning AMIs](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)
- [Amazon EC2 G6 instances](https://aws.amazon.com/ec2/instance-types/g6/)
- [IAM roles for EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html)
- [AWS CLI `s3 sync`](https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html)
- [Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Container Toolkit installation and Docker configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [PyTorch Docker images](https://hub.docker.com/r/pytorch/pytorch/tags)
- [vLLM v0.34.0 supported models](https://github.com/vllm-project/vllm/blob/v0.34.0/docs/models/supported_models.md)
