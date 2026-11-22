#!/bin/bash
# EC2 user-data, rendered by infra/gpu.sh (placeholders @@NAME@@). Runs once, at first boot, as root.
# Everything is logged to /var/log/dlm-bootstrap.log. Contains no secrets: only names and URIs.
set -uo pipefail
exec > >(tee -a /var/log/dlm-bootstrap.log) 2>&1
echo "bootstrap start $(date -u +%FT%TZ)"
REGION="@@REGION@@"
REGISTRY="@@REGISTRY@@"
IMAGE="@@IMAGE@@"
MODEL_URI="@@MODEL_URI@@"
MODEL_VERSION="@@MODEL_VERSION@@"
DTYPE="@@DTYPE@@"

# SSM agent: normally running on the DLAMI; Ubuntu ships it as a snap.
if snap list amazon-ssm-agent >/dev/null 2>&1; then snap start amazon-ssm-agent || true; fi

# Docker and the GPU runtime come with the Deep Learning Base AMI; verify instead of assuming.
for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 5; done
docker info >/dev/null 2>&1 || { echo "FATAL: docker is not available"; exit 1; }
command -v nvidia-ctk >/dev/null || { echo "FATAL: nvidia-container-toolkit missing; install it (see deploy.md step 10)"; exit 1; }
nvidia-smi || { echo "FATAL: nvidia-smi failed (driver not loaded)"; exit 1; }
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
for i in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 2; done

if ! command -v aws >/dev/null; then snap install aws-cli --classic; fi
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY" || { echo "FATAL: ECR login failed"; exit 1; }
docker pull "$IMAGE" || { echo "FATAL: image pull failed"; exit 1; }

docker rm -f dlm-api >/dev/null 2>&1 || true
docker run -d --name dlm-api --restart unless-stopped --gpus all \
  --publish 127.0.0.1:8000:8000 \
  -e DLM_MODEL_URI="$MODEL_URI" -e MODEL_VERSION="$MODEL_VERSION" -e DLM_DTYPE="$DTYPE" -e AWS_REGION="$REGION" \
  --log-opt max-size=10m --log-opt max-file=3 \
  "$IMAGE" || { echo "FATAL: docker run failed"; exit 1; }
echo "bootstrap done $(date -u +%FT%TZ): container dlm-api started (model download and load continue inside it)"
