# AWS Fargate deployment: tear-up and tear-down

Scripts that create and remove the whole inference service on **AWS Fargate** with one command each.
This replaces the EC2/GPU plan in `../deploy.md` for the CPU case (see "What changed" below).

```
Laptop -> (SSM tunnel | your-IP-only security group) -> Fargate task :8000 -> FastAPI -> sampler (CPU)
S3 versioned model folder -> downloaded and checksum-verified by the task at start -> memory
git commit -> `git archive` -> Docker image (ARM64) -> ECR -> task
```

## What changed from `deploy.md`

| | `deploy.md` | here |
| --- | --- | --- |
| Compute | EC2 `g6.xlarge`, NVIDIA L4 | **Fargate, CPU only** (to my knowledge Fargate has no GPU and ECS GPU tasks need the EC2 launch type; not re-checked against current AWS docs) |
| Image | `pytorch/pytorch:...-cuda...` | `python:3.12-slim` + CPU torch 2.15.0, **ARM64** (built natively on Apple Silicon) |
| Model | downloaded once to EBS by hand | downloaded from S3 by the task at every start, SHA-256 verified |
| Access | SSM shell + port forward | `ACCESS_MODE=ssm` (needs `session-manager-plugin`) or `ip` (security group open to your /32 only) |
| Code | `git clone` on the instance | `git archive` of the release's recorded commit on your laptop |

A 36M-parameter model fits comfortably on CPU. Whether the latency is acceptable is **not measured yet**.

## Files

- `dlm.sh`: `check`, `up`, `status`, `smoke`, `tunnel`, `allow-ip`, `down`, `model`.
- `Dockerfile`, `Dockerfile.dockerignore` (allow-list), `serve.py`, `entrypoint.py`: the container.
- `prepare_artifact.py`: validates a checkpoint and writes the versioned release folder with `deployment.json`.
- `make_test_checkpoint.py`: tiny random-weight checkpoint for testing the plumbing without a trained model.

## Use

```bash
cd DLM
export AWS_REGION=<region with a default VPC>      # us-east-1 has none in this account, see below
aws/dlm.sh check                                    # read-only: credentials, permissions, tools

export MODEL_VERSION=cosine-step25000-v1
export LOCAL_MODEL_DIR=/absolute/path/to/pretrain_cosine/final
aws/dlm.sh up                                       # asks once, then creates everything and starts the service
aws/dlm.sh status                                   # shows the endpoint
aws/dlm.sh smoke                                    # /healthz and a greedy /generate
aws/dlm.sh down                                     # asks once, removes everything (keeps the S3 bucket)
aws/dlm.sh down --delete-models                     # ... and deletes the script-created bucket and every release
```

`up` is idempotent: run it again after a failure, or with a new `MODEL_VERSION` to roll out a new release (the model
bucket is not touched by a service update). `up --no-service` creates only the infrastructure and the image.
Without `LOCAL_MODEL_DIR` and an existing release it stops before starting the service.

## What `up` creates (all tagged `Project=dlm-inference`)

| Resource | Name |
| --- | --- |
| S3 bucket (private, encrypted, **not versioned**) | `dlm-models-<account>-<region>`, or your `MODEL_BUCKET` (never deleted) |
| IAM roles | `dlm-ecs-execution-role` (ECR pull, logs), `dlm-ecs-task-role` (read `models/tinystories/*`, SSM channels) |
| CloudWatch log group | `/ecs/dlm-inference` (14 days) |
| ECR repository | `dlm-inference` |
| ECS cluster, task definition family, service | `dlm-inference`, `dlm-inference`, `dlm-api` (1 task) |
| Security group (default VPC) | `dlm-inference-sg` |

It uses the account's **default VPC**; it creates no VPC, NAT gateway or load balancer. Tasks get a public IP so they
can reach ECR, S3 and SSM; with no inbound rule that address accepts nothing.

## Tear-down order and guarantees

`down` removes, in this order: service (scaled to 0, then deleted, waits until inactive) -> stray tasks -> every
revision of the task-definition family -> cluster -> ECR repository with its images -> log group -> security group
(retries up to 10 minutes: Fargate network interfaces release slowly) -> the two IAM roles (policies detached/deleted
first). Every step tolerates "not found", so `down` works after a half-finished `up` and can be repeated. It ends with
a leftover check that names anything that still exists.

- The **S3 bucket is kept** unless you pass `--delete-models`, because it holds your trained releases. Even then it is
  deleted only if the script created it (name and `Project` tag), and every object version is purged first.
  A bucket you supplied through `MODEL_BUCKET` is never deleted.
- The ECS service-linked role (`AWSServiceRoleForECS`) is account-wide and shared; it is not touched.

## Access modes

- **`ip`** (used automatically when `session-manager-plugin` is missing): the security group allows tcp/8000 from
  this machine's current public IP /32 and nothing else. The traffic is plain HTTP and the API has no login, so treat
  it as a personal test endpoint. The task's public IP changes at every restart (`status` prints it); if your own IP
  changes, run `aws/dlm.sh allow-ip`.
- **`ssm`**: no inbound rule; `aws/dlm.sh tunnel` forwards `localhost:18000` through Session Manager. Install the
  plugin first: `brew install --cask session-manager-plugin`.

## Credentials

The scripts use `AWS_PROFILE` / `AWS_ACCESS_KEY_ID` if set, otherwise read **only** the `## aws_deployer_key` section
of `AWS_credential.md` into environment variables of the running script. Nothing is printed, written, passed as a
command-line argument, or baked into the image (the build context is a `git archive` of tracked files and the
Dockerfile ignore file is an allow-list). The credential file is git-ignored.

## Cost

Fargate bills per second while the service runs, whether or not anyone calls it: 2 vCPU / 4 GiB is the default
(`TASK_CPU=2048 TASK_MEMORY=4096`; smaller valid sizes cost less but are untested for this model). Check the Fargate ARM
price for your region before leaving it on. Also billed: ECR storage, CloudWatch logs, S3 bytes. Run `down` when not
using it.

## What was tested, and what was not

Tested on this Mac, with no AWS resources created:
- `bash -n`, credential parsing (right section, quotes stripped, wrong or missing keys rejected) on fake files.
- The image (ARM64, CPU torch `2.15.0+cpu`) built from a `git archive` context and run against a tiny random-weight
  release: `/healthz`, greedy and story `/generate`, `422` on bad requests, a tampered weights file refused, a wrong
  code revision refused.
- `entrypoint.py`'s S3 download and verification logic against a fake S3 client (11 checks): path traversal, empty
  prefix, bad URI, wrong version, missing `deployment.json`, checksum mismatch.
- Read-only AWS calls with the real credentials (`check`, `status`): identity, permissions to list IAM/ECS/ECR/EC2/S3/logs.

**Not tested:** any create or delete call against AWS (`up`, `down`, `model`, `smoke`, `tunnel`, `allow-ip`), the task
starting on Fargate, the S3 download with the task role, and the SSM tunnel. The first real `up` then `down` will
show whether the deployer key has every write permission it needs (for example `iam:CreateRole`, `ecs:CreateService`,
and creating the ECS service-linked role if the account has never used ECS) and whether teardown leaves nothing behind.

## Known limits

- One task, one generation at a time (a second concurrent request gets `503`); no autoscaling, no HTTPS, no login.
- Replacing a release stops the old task before the new one starts (`minimumHealthyPercent=0`): a short outage.
- The tagging-API leftover hint needs `tag:GetResources`; the deployer key currently lacks it, so the leftover check
  verifies each named resource directly instead.
- `us-east-1` has no default VPC in this account; use a region that does (checked: `ap-southeast-1`, `ap-southeast-2`,
  `us-west-2`, `eu-west-1`).
