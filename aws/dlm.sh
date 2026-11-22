#!/usr/bin/env bash
# Tear-up / tear-down of the TinyStories DLM inference service on AWS Fargate (CPU, ARM64 by default).
#
#   aws/dlm.sh check                 read-only: credentials, permissions, default VPC, local tools
#   aws/dlm.sh up [--no-service] [--yes]
#   aws/dlm.sh status
#   aws/dlm.sh smoke                 /healthz + /generate against the running service
#   aws/dlm.sh tunnel                ACCESS_MODE=ssm only: port-forward the API to localhost:$LOCAL_PORT
#   aws/dlm.sh allow-ip              ACCESS_MODE=ip only: re-point the security group at this machine's current IP
#   aws/dlm.sh down [--delete-models] [--yes]
#
# Settings (environment): AWS_REGION (us-east-1), MODEL_VERSION, LOCAL_MODEL_DIR, DLM_SCHEDULE, MODEL_BUCKET,
# ACCESS_MODE (auto|ssm|ip), TASK_CPU (2048), TASK_MEMORY (4096), CPU_ARCH (ARM64|X86_64), LOCAL_PORT (18000).
# Credentials: AWS_PROFILE / AWS_ACCESS_KEY_ID if already set, else the `## aws_deployer_key` section of
# AWS_credential.md (git-ignored). Values are never printed, never passed on a command line, never written anywhere.
# Bash 3.2 compatible (macOS). No `set -x`: it would print credentials.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

# ---------------------------------------------------------------- settings and fixed names
PROJECT_TAG="${PROJECT_TAG:-dlm-inference}"
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
export AWS_REGION AWS_DEFAULT_REGION="$AWS_REGION" AWS_PAGER=""
CREDENTIAL_FILE="${CREDENTIAL_FILE:-$REPO/AWS_credential.md}"
CREDENTIAL_SECTION="${CREDENTIAL_SECTION:-aws_deployer_key}"
CPU_ARCH="${CPU_ARCH:-ARM64}"
TASK_CPU="${TASK_CPU:-2048}"
TASK_MEMORY="${TASK_MEMORY:-4096}"
PORT=8000
LOCAL_PORT="${LOCAL_PORT:-18000}"
ACCESS_MODE="${ACCESS_MODE:-auto}"
MODEL_VERSION="${MODEL_VERSION:-}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-}"
DLM_SCHEDULE="${DLM_SCHEDULE:-}"
MODEL_BUCKET="${MODEL_BUCKET:-}"

CLUSTER="dlm-inference"
SERVICE="dlm-api"
FAMILY="dlm-inference"
CONTAINER="dlm-api"
REPO_NAME="dlm-inference"
LOG_GROUP="/ecs/dlm-inference"
SG_NAME="dlm-inference-sg"
EXEC_ROLE="dlm-ecs-execution-role"
TASK_ROLE="dlm-ecs-task-role"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$REPO/.venv/bin/python" ]; then PYTHON_BIN="$REPO/.venv/bin/python"; else PYTHON_BIN="python3"; fi
fi

TMP_DIRS=()
TUNNEL_PID=""
cleanup() {
  [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null || true
  local d
  for d in ${TMP_DIRS[@]+"${TMP_DIRS[@]}"}; do rm -rf "$d"; done
}
trap cleanup EXIT

log()  { printf '==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
MKTMP=""
mktmp() { MKTMP="$(mktemp -d)"; TMP_DIRS+=("$MKTMP"); }  # sets $MKTMP (a subshell would lose the cleanup list)

# ---------------------------------------------------------------- credentials and identity
load_credentials() {
  if [ -n "${AWS_ACCESS_KEY_ID:-}" ] || [ -n "${AWS_PROFILE:-}" ]; then return 0; fi
  [ -r "$CREDENTIAL_FILE" ] || die "No AWS credentials: set AWS_PROFILE, or provide $CREDENTIAL_FILE with a '## $CREDENTIAL_SECTION' section"
  local line tag val key="" secret=""
  while IFS= read -r line; do
    tag="${line%% *}"; val="${line#* }"
    case "$tag" in K) key="$val";; S) secret="$val";; esac
  done < <(awk -v sec="$CREDENTIAL_SECTION" '
      /^##/ { in_sec = (index($0, sec) > 0); next }
      in_sec {
        line = $0; sub(/\r$/, "", line)
        if (match(line, /^[ \t]*[Aa]ccess_[Ss]ecret_[Kk]ey[ \t]*[=:]/)) { tag = "S" }
        else if (match(line, /^[ \t]*[Aa]ccess_[Kk]ey[ \t]*[=:]/))       { tag = "K" }
        else next
        v = substr(line, RLENGTH + 1); gsub(/["\047]/, "", v); gsub(/^[ \t]+|[ \t]+$/, "", v)
        if (v != "") print tag " " v
      }' "$CREDENTIAL_FILE")
  [ -n "$key" ] && [ -n "$secret" ] || die "Section '$CREDENTIAL_SECTION' of $CREDENTIAL_FILE has no access_key / access_secret_key"
  export AWS_ACCESS_KEY_ID="$key" AWS_SECRET_ACCESS_KEY="$secret"
}

ACCOUNT_ID=""
init_aws() {
  command -v aws >/dev/null || die "aws CLI not found"
  command -v jq  >/dev/null || die "jq not found"
  load_credentials
  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)" \
    || die "AWS rejected the credentials (aws sts get-caller-identity failed)"
  REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
  IMAGE_REPO="${REGISTRY}/${REPO_NAME}"
  if [ -n "$MODEL_BUCKET" ]; then BUCKET="$MODEL_BUCKET"; BUCKET_GIVEN=1
  else BUCKET="dlm-models-${ACCOUNT_ID}-${AWS_REGION}"; BUCKET_GIVEN=0; fi
  MODEL_URI="s3://${BUCKET}/models/tinystories/${MODEL_VERSION}"
  case "$CPU_ARCH" in
    ARM64)  DOCKER_PLATFORM="linux/arm64";;
    X86_64) DOCKER_PLATFORM="linux/amd64";;
    *) die "CPU_ARCH must be ARM64 or X86_64";;
  esac
}

resolve_access_mode() {
  case "$ACCESS_MODE" in
    ssm|ip) ;;
    auto) if command -v session-manager-plugin >/dev/null; then ACCESS_MODE=ssm; else ACCESS_MODE=ip; fi;;
    *) die "ACCESS_MODE must be auto, ssm or ip";;
  esac
}

# Prints "None" for nothing; normalise to empty.
none_to_empty() { case "$1" in None|null) printf '';; *) printf '%s' "$1";; esac; }

# ---------------------------------------------------------------- network (default VPC only)
VPC_ID=""; SUBNETS=""
find_network() {
  local raw
  raw="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)" \
    || die "ec2:DescribeVpcs failed (permissions?)"
  VPC_ID="$(none_to_empty "$raw")"
  [ -n "$VPC_ID" ] || die "No default VPC in $AWS_REGION. Set AWS_REGION to a region that has one, or create one (aws ec2 create-default-vpc)."
  SUBNETS="$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" Name=default-for-az,Values=true \
      --query 'Subnets[].SubnetId' --output text | tr '\t' ',')"
  [ -n "$SUBNETS" ] || die "The default VPC has no default subnets"
}

find_sg() {
  none_to_empty "$(aws ec2 describe-security-groups \
      --filters "Name=group-name,Values=$SG_NAME" "Name=tag:Project,Values=$PROJECT_TAG" \
      --query 'SecurityGroups[0].GroupId' --output text)"
}

my_ip() {
  local ip
  ip="$(curl -fsS --max-time 10 https://checkip.amazonaws.com | tr -d '[:space:]')" || die "Could not determine this machine's public IP"
  printf '%s' "$ip" | grep -Eq '^[0-9]{1,3}(\.[0-9]{1,3}){3}$' || die "Unexpected IP answer: $ip"
  printf '%s' "$ip"
}

ensure_sg() {
  local sg
  sg="$(find_sg)"
  if [ -z "$sg" ]; then
    log "security group $SG_NAME: creating (no inbound rules)"
    sg="$(aws ec2 create-security-group --group-name "$SG_NAME" --vpc-id "$VPC_ID" \
        --description "DLM inference on Fargate: no inbound unless ACCESS_MODE=ip" \
        --tag-specifications "ResourceType=security-group,Tags=[{Key=Project,Value=$PROJECT_TAG}]" \
        --query GroupId --output text)"
  else
    log "security group $SG_NAME: exists ($sg)"
  fi
  SG_ID="$sg"
  sync_ingress
}

# ssm mode: no inbound rule at all. ip mode: tcp/$PORT from this machine's /32 only (never 0.0.0.0/0).
sync_ingress() {
  local want="" c current
  if [ "$ACCESS_MODE" = "ip" ]; then want="$(my_ip)/32"; fi
  current="$(none_to_empty "$(aws ec2 describe-security-groups --group-ids "$SG_ID" \
      --query 'SecurityGroups[0].IpPermissions[].IpRanges[].CidrIp' --output text)")"
  for c in $current; do
    if [ "$c" != "$want" ]; then
      note "revoking inbound $c"
      aws ec2 revoke-security-group-ingress --group-id "$SG_ID" --protocol tcp --port "$PORT" --cidr "$c" >/dev/null
    fi
  done
  if [ -n "$want" ] && ! printf '%s\n' $current | grep -qx "$want"; then
    note "allowing inbound tcp/$PORT from $want"
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port "$PORT" --cidr "$want" >/dev/null
  fi
}

# ---------------------------------------------------------------- S3
bucket_exists() { aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; }
bucket_is_ours() {
  [ "$(aws s3api get-bucket-tagging --bucket "$BUCKET" --query "TagSet[?Key=='Project'].Value | [0]" --output text 2>/dev/null)" = "$PROJECT_TAG" ]
}

ensure_bucket() {
  if bucket_exists; then
    if [ "$BUCKET_GIVEN" = 1 ]; then log "S3 bucket $BUCKET: using your existing bucket (never deleted by this script)"
    elif bucket_is_ours; then log "S3 bucket $BUCKET: exists"
    else die "Bucket $BUCKET exists but is not tagged Project=$PROJECT_TAG; refusing to use it. Set MODEL_BUCKET to use it deliberately."; fi
    return 0
  fi
  [ "$BUCKET_GIVEN" = 0 ] || die "MODEL_BUCKET=$BUCKET does not exist or is not accessible"
  log "S3 bucket $BUCKET: creating (private, encrypted, not versioned)"
  if [ "$AWS_REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration "LocationConstraint=$AWS_REGION" >/dev/null
  fi
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
      BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
      '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  aws s3api put-bucket-tagging --bucket "$BUCKET" --tagging "TagSet=[{Key=Project,Value=$PROJECT_TAG}]"
}

release_present() { aws s3api head-object --bucket "$BUCKET" --key "models/tinystories/${MODEL_VERSION}/deployment.json" >/dev/null 2>&1; }
release_code_ref() { aws s3 cp "$MODEL_URI/deployment.json" - 2>/dev/null | jq -r '.code_ref' 2>/dev/null || true; }

purge_bucket() {
  log "S3 bucket $BUCKET: deleting every object, version and delete marker"
  aws s3 rm "s3://$BUCKET" --recursive >/dev/null 2>&1 || true
  local doc n
  mktmp; doc="$MKTMP/del.json"
  while :; do
    aws s3api list-object-versions --bucket "$BUCKET" --max-items 500 --output json \
      | jq '{Objects: ([(.Versions // [])[], (.DeleteMarkers // [])[]] | map({Key, VersionId})), Quiet: true}' > "$doc"
    n="$(jq '.Objects | length' "$doc")"
    [ "$n" -gt 0 ] || break
    aws s3api delete-objects --bucket "$BUCKET" --delete "file://$doc" >/dev/null
  done
  aws s3api delete-bucket --bucket "$BUCKET"
}

# ---------------------------------------------------------------- IAM
trust_policy() {
  cat <<'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
}

role_exists() { aws iam get-role --role-name "$1" >/dev/null 2>&1; }

ensure_role() {  # name
  if role_exists "$1"; then log "IAM role $1: exists"; return 0; fi
  log "IAM role $1: creating"
  aws iam create-role --role-name "$1" --assume-role-policy-document "$(trust_policy)" \
      --tags "Key=Project,Value=$PROJECT_TAG" >/dev/null
  aws iam wait role-exists --role-name "$1"
  ROLES_CREATED=1
}

ensure_roles() {
  ROLES_CREATED=0
  ensure_role "$EXEC_ROLE"
  aws iam attach-role-policy --role-name "$EXEC_ROLE" \
      --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
  ensure_role "$TASK_ROLE"
  # Read-only on the model prefix, plus the SSM channels ECS Exec / port forwarding need.
  aws iam put-role-policy --role-name "$TASK_ROLE" --policy-name ReadDlmModelArtifacts --policy-document "$(cat <<JSON
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":"s3:ListBucket","Resource":"arn:aws:s3:::${BUCKET}",
  "Condition":{"StringLike":{"s3:prefix":["models/tinystories","models/tinystories/*"]}}},
 {"Effect":"Allow","Action":"s3:GetObject","Resource":"arn:aws:s3:::${BUCKET}/models/tinystories/*"}]}
JSON
)"
  aws iam put-role-policy --role-name "$TASK_ROLE" --policy-name EcsExecSsmChannels --policy-document \
      '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["ssmmessages:CreateControlChannel","ssmmessages:CreateDataChannel","ssmmessages:OpenControlChannel","ssmmessages:OpenDataChannel"],"Resource":"*"}]}'
  if [ "$ROLES_CREATED" = 1 ]; then note "waiting for IAM to propagate"; sleep 15; fi
}

delete_role() {  # name
  if ! role_exists "$1"; then log "IAM role $1: not found"; return 0; fi
  log "IAM role $1: deleting"
  local arn name
  for arn in $(aws iam list-attached-role-policies --role-name "$1" --query 'AttachedPolicies[].PolicyArn' --output text); do
    aws iam detach-role-policy --role-name "$1" --policy-arn "$arn"
  done
  for name in $(aws iam list-role-policies --role-name "$1" --query 'PolicyNames' --output text); do
    aws iam delete-role-policy --role-name "$1" --policy-name "$name"
  done
  aws iam delete-role --role-name "$1"
}

# ---------------------------------------------------------------- logs, registry, cluster
ensure_log_group() {
  if [ "$(aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" --query "length(logGroups[?logGroupName=='$LOG_GROUP'])" --output text)" = "0" ]; then
    log "log group $LOG_GROUP: creating (14-day retention)"
    aws logs create-log-group --log-group-name "$LOG_GROUP" --tags "Project=$PROJECT_TAG"
  else
    log "log group $LOG_GROUP: exists"
  fi
  aws logs put-retention-policy --log-group-name "$LOG_GROUP" --retention-in-days 14
}

ensure_ecr() {
  if aws ecr describe-repositories --repository-names "$REPO_NAME" >/dev/null 2>&1; then
    log "ECR repository $REPO_NAME: exists"
  else
    log "ECR repository $REPO_NAME: creating"
    aws ecr create-repository --repository-name "$REPO_NAME" --tags "Key=Project,Value=$PROJECT_TAG" >/dev/null
  fi
}

cluster_active() { [ "$(aws ecs describe-clusters --clusters "$CLUSTER" --query "clusters[?status=='ACTIVE'] | length(@)" --output text)" != "0" ]; }

ensure_cluster() {
  if cluster_active; then log "ECS cluster $CLUSTER: exists"; return 0; fi
  log "ECS cluster $CLUSTER: creating"
  aws ecs create-cluster --cluster-name "$CLUSTER" --tags "key=Project,value=$PROJECT_TAG" >/dev/null
}

# ---------------------------------------------------------------- model release
cmd_model() {
  [ -n "$MODEL_VERSION" ]   || die "Set MODEL_VERSION (a unique release name, e.g. cosine-step25000-v1)"
  [ -n "$LOCAL_MODEL_DIR" ] || die "Set LOCAL_MODEL_DIR to the trained run's final/ (or exported) folder"
  [ -d "$LOCAL_MODEL_DIR" ] || die "$LOCAL_MODEL_DIR is not a directory"
  if release_present; then
    log "model release $MODEL_VERSION: already in S3 (releases are immutable; use a new MODEL_VERSION to publish another)"
    return 0
  fi
  if [ -n "$(git -C "$REPO" status --porcelain -- dlm run.py)" ] && [ "${ALLOW_DIRTY:-0}" != 1 ]; then
    die "dlm/ or run.py has uncommitted changes; the release records the git commit. Commit them or set ALLOW_DIRTY=1."
  fi
  local code_ref art
  code_ref="$(git -C "$REPO" rev-parse HEAD)"
  mktmp; art="$MKTMP"
  log "model release $MODEL_VERSION: validating $LOCAL_MODEL_DIR (code $code_ref)"
  local sched=()
  [ -z "$DLM_SCHEDULE" ] || sched=(--schedule "$DLM_SCHEDULE")
  "$PYTHON_BIN" "$HERE/prepare_artifact.py" --source "$LOCAL_MODEL_DIR" --target "$art" \
      --model-version "$MODEL_VERSION" --code-ref "$code_ref" ${sched[@]+"${sched[@]}"}
  log "model release $MODEL_VERSION: uploading to $MODEL_URI (deployment.json last)"
  aws s3 sync "$art/" "$MODEL_URI/" --exclude deployment.json --only-show-errors
  aws s3 cp "$art/deployment.json" "$MODEL_URI/deployment.json" --only-show-errors
}

# ---------------------------------------------------------------- image
IMAGE_URI=""
build_and_push() {
  command -v docker >/dev/null || die "docker not found"
  docker info >/dev/null 2>&1 || die "The Docker daemon is not running"
  local ref short ctx
  ref="$(release_code_ref)"
  [ -n "$ref" ] || ref="$(git -C "$REPO" rev-parse HEAD)"
  git -C "$REPO" cat-file -e "${ref}^{commit}" 2>/dev/null || die "Code revision $ref is not in this clone; fetch it first"
  short="$(printf '%s' "$ref" | cut -c1-12)"
  IMAGE_URI="${IMAGE_REPO}:code-${short}-${CPU_ARCH}"
  if [ "${FORCE_BUILD:-0}" != 1 ] && aws ecr describe-images --repository-name "$REPO_NAME" --image-ids "imageTag=code-${short}-${CPU_ARCH}" >/dev/null 2>&1; then
    log "image $IMAGE_URI: already in ECR"
    CODE_REF="$ref"; return 0
  fi
  mktmp; ctx="$MKTMP"
  # Only tracked files of that commit: AWS_credential.md, data/, .venv/ and checkpoints/ cannot get in.
  git -C "$REPO" archive "$ref" dlm run.py | tar -x -C "$ctx"
  cp "$HERE/Dockerfile" "$HERE/Dockerfile.dockerignore" "$HERE/serve.py" "$HERE/entrypoint.py" "$ctx/"
  [ ! -e "$ctx/AWS_credential.md" ] || die "credential file found in the build context"
  log "image $IMAGE_URI: building for $DOCKER_PLATFORM from commit $short"
  docker build --platform "$DOCKER_PLATFORM" --provenance=false --build-arg "CODE_REF=$ref" -t "$IMAGE_URI" "$ctx"
  log "image $IMAGE_URI: pushing"
  aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
  docker push "$IMAGE_URI" >/dev/null
  CODE_REF="$ref"
}

# ---------------------------------------------------------------- task definition and service
TASKDEF_ARN=""
register_taskdef() {
  local exec_arn task_arn doc
  exec_arn="arn:aws:iam::${ACCOUNT_ID}:role/${EXEC_ROLE}"
  task_arn="arn:aws:iam::${ACCOUNT_ID}:role/${TASK_ROLE}"
  mktmp; doc="$MKTMP/taskdef.json"
  jq -n --arg family "$FAMILY" --arg cpu "$TASK_CPU" --arg mem "$TASK_MEMORY" --arg arch "$CPU_ARCH" \
        --arg exec "$exec_arn" --arg task "$task_arn" --arg name "$CONTAINER" --arg image "$IMAGE_URI" \
        --arg uri "$MODEL_URI" --arg ver "$MODEL_VERSION" --arg region "$AWS_REGION" --arg lg "$LOG_GROUP" \
        --arg threads "$((TASK_CPU / 1024 > 0 ? TASK_CPU / 1024 : 1))" --arg tag "$PROJECT_TAG" --argjson port "$PORT" '
    {family: $family, networkMode: "awsvpc", requiresCompatibilities: ["FARGATE"], cpu: $cpu, memory: $mem,
     runtimePlatform: {cpuArchitecture: $arch, operatingSystemFamily: "LINUX"},
     executionRoleArn: $exec, taskRoleArn: $task,
     containerDefinitions: [{name: $name, image: $image, essential: true,
       portMappings: [{containerPort: $port, protocol: "tcp"}],
       environment: [{name: "DLM_MODEL_URI", value: $uri}, {name: "MODEL_VERSION", value: $ver},
                     {name: "DLM_DEVICE", value: "cpu"}, {name: "DLM_THREADS", value: $threads},
                     {name: "AWS_REGION", value: $region}],
       logConfiguration: {logDriver: "awslogs",
         options: {"awslogs-group": $lg, "awslogs-region": $region, "awslogs-stream-prefix": "dlm"}},
       healthCheck: {command: ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)\""],
                     interval: 30, timeout: 5, retries: 3, startPeriod: 120},
       linuxParameters: {initProcessEnabled: true}}],
     tags: [{key: "Project", value: $tag}]}' > "$doc"
  TASKDEF_ARN="$(aws ecs register-task-definition --cli-input-json "file://$doc" --query 'taskDefinition.taskDefinitionArn' --output text)"
  log "task definition: registered ${TASKDEF_ARN##*/}"
}

service_status() { none_to_empty "$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --query 'services[0].status' --output text 2>/dev/null)"; }

ensure_service() {
  local netcfg="awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}"
  if [ "$(service_status)" = "ACTIVE" ]; then
    log "ECS service $SERVICE: updating to the new task definition"
    aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" --task-definition "$TASKDEF_ARN" \
        --desired-count 1 --network-configuration "$netcfg" --enable-execute-command --force-new-deployment >/dev/null
  else
    log "ECS service $SERVICE: creating (1 task, Fargate $CPU_ARCH, ${TASK_CPU} CPU units / ${TASK_MEMORY} MiB)"
    aws ecs create-service --cluster "$CLUSTER" --service-name "$SERVICE" --task-definition "$TASKDEF_ARN" \
        --desired-count 1 --launch-type FARGATE --network-configuration "$netcfg" --enable-execute-command \
        --deployment-configuration "minimumHealthyPercent=0,maximumPercent=100,deploymentCircuitBreaker={enable=true,rollback=false}" \
        --enable-ecs-managed-tags --propagate-tags SERVICE --tags "key=Project,value=$PROJECT_TAG" >/dev/null
  fi
  log "waiting for the service to become stable (model download + start, up to ~10 minutes)"
  if ! aws ecs wait services-stable --cluster "$CLUSTER" --services "$SERVICE"; then
    warn "service did not become stable. Recent events and task logs:"
    aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" --query 'services[0].events[:5].message' --output text >&2 || true
    warn "logs: aws logs tail $LOG_GROUP --since 30m"
    return 1
  fi
}

TASK_ARN=""; TASK_ID=""; RUNTIME_ID=""; TASK_STATUS=""; PUBLIC_IP=""
load_task() {
  TASK_ARN="$(none_to_empty "$(aws ecs list-tasks --cluster "$CLUSTER" --service-name "$SERVICE" --desired-status RUNNING --query 'taskArns[0]' --output text 2>/dev/null)")"
  TASK_ID=""; RUNTIME_ID=""; TASK_STATUS=""; PUBLIC_IP=""
  [ -n "$TASK_ARN" ] || return 0
  local info eni
  info="$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$TASK_ARN" --output json)"
  TASK_ID="${TASK_ARN##*/}"
  TASK_STATUS="$(printf '%s' "$info" | jq -r '.tasks[0].lastStatus')"
  RUNTIME_ID="$(printf '%s' "$info" | jq -r '.tasks[0].containers[0].runtimeId // empty')"
  eni="$(printf '%s' "$info" | jq -r '[.tasks[0].attachments[].details[] | select(.name=="networkInterfaceId") | .value][0] // empty')"
  if [ -n "$eni" ]; then
    PUBLIC_IP="$(none_to_empty "$(aws ec2 describe-network-interfaces --network-interface-ids "$eni" --query 'NetworkInterfaces[0].Association.PublicIp' --output text 2>/dev/null)")"
  fi
}

# ---------------------------------------------------------------- commands
confirm() {  # message
  [ "${ASSUME_YES:-0}" = 1 ] && return 0
  printf '%s [y/N] ' "$1"; local a; read -r a || a=""
  case "$a" in y|Y|yes|YES) return 0;; *) die "Cancelled";; esac
}

cmd_check() {
  init_aws; resolve_access_mode
  log "account $ACCOUNT_ID, region $AWS_REGION, access mode: $ACCESS_MODE"
  local ok=1 t
  for t in docker jq git curl; do command -v "$t" >/dev/null && note "tool $t: ok" || { note "tool $t: MISSING"; ok=0; }; done
  docker info >/dev/null 2>&1 && note "docker daemon: running" || { note "docker daemon: NOT running"; ok=0; }
  if [ "$ACCESS_MODE" = ssm ]; then note "session-manager-plugin: ok"
  else note "session-manager-plugin: not installed (ip mode will be used; install it for the zero-inbound ssm mode)"; fi
  find_network; note "default VPC $VPC_ID, subnets $SUBNETS"
  # Read-only probes of each service the scripts will create things in.
  probe() { if "$@" >/dev/null 2>&1; then note "permission $PROBE_NAME: ok"; else note "permission $PROBE_NAME: DENIED or failing"; ok=0; fi; }
  PROBE_NAME="iam"   probe aws iam list-roles --max-items 1
  PROBE_NAME="ecs"   probe aws ecs list-clusters
  PROBE_NAME="ecr"   probe aws ecr describe-repositories --max-items 1
  PROBE_NAME="ec2"   probe aws ec2 describe-security-groups --max-items 1
  PROBE_NAME="s3"    probe aws s3api list-buckets
  PROBE_NAME="logs"  probe aws logs describe-log-groups --limit 1
  # Optional: only the extra "tagged resource still listed" hint at the end of 'down' uses it.
  aws resourcegroupstaggingapi get-resources --resources-per-page 1 >/dev/null 2>&1 \
    && note "permission tagging: ok" || note "permission tagging: denied (optional; 'down' still verifies each resource directly)"
  [ "$ok" = 1 ] && log "check passed (read-only probes; write permissions are only proven by 'up')" || die "check found problems (see above)"
}

cmd_up() {
  local no_service=0
  while [ $# -gt 0 ]; do case "$1" in --no-service) no_service=1;; --yes) ASSUME_YES=1;; *) die "unknown option $1";; esac; shift; done
  init_aws; resolve_access_mode; find_network
  [ "$no_service" = 1 ] || [ -n "$MODEL_VERSION" ] || die "Set MODEL_VERSION, or use --no-service to create only the infrastructure"
  log "Tear-up plan: account $ACCOUNT_ID, region $AWS_REGION"
  note "creates (all tagged Project=$PROJECT_TAG): S3 bucket (if none given), 2 IAM roles, log group, ECR repository,"
  note "ECS cluster, security group, task definition and a Fargate service (${TASK_CPU} CPU units / ${TASK_MEMORY} MiB, $CPU_ARCH, always on)."
  note "access mode: $ACCESS_MODE   model release: ${MODEL_VERSION:-none yet}"
  note "Fargate bills per second while the service runs; 'dlm.sh down' removes everything."
  confirm "Create these resources?"
  ensure_bucket; ensure_roles; ensure_log_group; ensure_ecr; ensure_cluster; ensure_sg
  [ -z "$LOCAL_MODEL_DIR" ] || cmd_model
  build_and_push
  if [ "$no_service" = 1 ]; then log "infrastructure ready; service not started (--no-service)"; return 0; fi
  if ! release_present; then
    warn "No model release $MODEL_VERSION in s3://$BUCKET. Set LOCAL_MODEL_DIR and run 'up' again; the service was not started."
    return 0
  fi
  register_taskdef
  ensure_service
  cmd_status
}

cmd_status() {
  init_aws; resolve_access_mode
  log "account $ACCOUNT_ID, region $AWS_REGION, access mode $ACCESS_MODE"
  local sg status
  note "ECS cluster:  $(cluster_active && echo present || echo absent)"
  status="$(service_status)"; note "ECS service:  ${status:-absent}"
  sg="$(find_sg)"; note "security grp:  ${sg:-absent}"
  note "ECR repo:     $(aws ecr describe-repositories --repository-names "$REPO_NAME" >/dev/null 2>&1 && echo present || echo absent)"
  note "S3 bucket:    $BUCKET ($(bucket_exists && echo present || echo absent))"
  note "IAM roles:    $(role_exists "$EXEC_ROLE" && echo "$EXEC_ROLE ") $(role_exists "$TASK_ROLE" && echo "$TASK_ROLE")"
  if [ "$status" = "ACTIVE" ]; then
    load_task
    if [ -z "$TASK_ARN" ]; then note "task:         none running"
    else
      note "task:         $TASK_ID ($TASK_STATUS)"
      if [ "$ACCESS_MODE" = ip ]; then note "endpoint:     http://${PUBLIC_IP:-<pending>}:$PORT   (allowed source: this machine's IP only; HTTP, not HTTPS)"
      else note "endpoint:     run 'aws/dlm.sh tunnel', then http://127.0.0.1:$LOCAL_PORT"; fi
    fi
  fi
}

cmd_tunnel() {
  init_aws; resolve_access_mode
  [ "$ACCESS_MODE" = ssm ] || die "tunnel needs ACCESS_MODE=ssm and the session-manager-plugin (brew install --cask session-manager-plugin)"
  load_task; [ -n "$RUNTIME_ID" ] || die "No running task with a runtime ID yet; check 'status'"
  log "forwarding localhost:$LOCAL_PORT -> task $TASK_ID port $PORT (Ctrl-C to close)"
  aws ssm start-session --target "ecs:${CLUSTER}_${TASK_ID}_${RUNTIME_ID}" \
      --document-name AWS-StartPortForwardingSession \
      --parameters "{\"portNumber\":[\"$PORT\"],\"localPortNumber\":[\"$LOCAL_PORT\"]}"
}

cmd_allow_ip() {
  init_aws; resolve_access_mode
  [ "$ACCESS_MODE" = ip ] || die "allow-ip only applies to ACCESS_MODE=ip"
  SG_ID="$(find_sg)"; [ -n "$SG_ID" ] || die "No security group; run 'up' first"
  sync_ingress
}

cmd_smoke() {
  init_aws; resolve_access_mode
  load_task; [ -n "$TASK_ARN" ] || die "No running task"
  local base
  if [ "$ACCESS_MODE" = ip ]; then
    [ -n "$PUBLIC_IP" ] || die "The task has no public IP yet"
    base="http://${PUBLIC_IP}:$PORT"
  else
    aws ssm start-session --target "ecs:${CLUSTER}_${TASK_ID}_${RUNTIME_ID}" --document-name AWS-StartPortForwardingSession \
        --parameters "{\"portNumber\":[\"$PORT\"],\"localPortNumber\":[\"$LOCAL_PORT\"]}" >/dev/null &
    TUNNEL_PID=$!
    sleep 8
    base="http://127.0.0.1:$LOCAL_PORT"
  fi
  log "GET $base/healthz"
  curl -fsS --max-time 30 "$base/healthz"; echo
  log "POST $base/generate"
  curl -fsS --max-time 300 "$base/generate" -H 'Content-Type: application/json' \
      -d '{"mode":"continue","prompt":"Once upon a time, a little fox","max_new_tokens":64,"steps":64,"temperature":0,"seed":0}'; echo
}

cmd_down() {
  local delete_models=0
  while [ $# -gt 0 ]; do case "$1" in --delete-models) delete_models=1;; --yes) ASSUME_YES=1;; *) die "unknown option $1";; esac; shift; done
  init_aws
  log "Tear-down plan: account $ACCOUNT_ID, region $AWS_REGION"
  note "deletes: ECS service, tasks, task definitions, cluster, ECR repository (images), log group, security group, 2 IAM roles."
  if [ "$delete_models" = 1 ]; then
    if [ "$BUCKET_GIVEN" = 1 ]; then note "S3: $BUCKET is YOUR bucket; it and its model releases are left alone."
    else note "S3: ALSO deletes bucket $BUCKET with every model release in it (irreversible)."; fi
  else
    note "S3: bucket $BUCKET and the model releases in it are kept (use --delete-models to remove a script-created bucket)."
  fi
  confirm "Delete these resources?"

  # 1. service and tasks
  case "$(service_status)" in
    ACTIVE|DRAINING)
      log "ECS service $SERVICE: scaling to 0 and deleting"
      aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" --desired-count 0 >/dev/null 2>&1 || true
      aws ecs delete-service --cluster "$CLUSTER" --service "$SERVICE" --force >/dev/null
      aws ecs wait services-inactive --cluster "$CLUSTER" --services "$SERVICE" || warn "service still draining; continuing";;
    *) log "ECS service $SERVICE: not found";;
  esac
  if cluster_active; then
    local t
    for t in $(aws ecs list-tasks --cluster "$CLUSTER" --query 'taskArns' --output text); do
      log "stopping stray task ${t##*/}"; aws ecs stop-task --cluster "$CLUSTER" --task "$t" >/dev/null || true
    done
  fi
  # 2. task definitions (every revision of our family)
  local td found=0
  for td in $(aws ecs list-task-definitions --family-prefix "$FAMILY" --status ACTIVE --query 'taskDefinitionArns' --output text); do
    case "$td" in *"task-definition/${FAMILY}:"*)
      found=1; log "task definition ${td##*/}: deregistering"
      aws ecs deregister-task-definition --task-definition "$td" >/dev/null
      aws ecs delete-task-definitions --task-definitions "$td" >/dev/null 2>&1 || true;;
    esac
  done
  [ "$found" = 1 ] || log "task definitions: none active"
  # 3. cluster
  if cluster_active; then
    log "ECS cluster $CLUSTER: deleting"
    local i
    for i in 1 2 3 4 5 6; do
      aws ecs delete-cluster --cluster "$CLUSTER" >/dev/null 2>&1 && break
      note "cluster busy, retrying in 10s"; sleep 10
    done
  else log "ECS cluster $CLUSTER: not found"; fi
  # 4. registry and logs
  if aws ecr describe-repositories --repository-names "$REPO_NAME" >/dev/null 2>&1; then
    log "ECR repository $REPO_NAME: deleting with its images"
    aws ecr delete-repository --repository-name "$REPO_NAME" --force >/dev/null
  else log "ECR repository $REPO_NAME: not found"; fi
  command -v docker >/dev/null && docker logout "$REGISTRY" >/dev/null 2>&1 || true
  if [ "$(aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" --query "length(logGroups[?logGroupName=='$LOG_GROUP'])" --output text)" != "0" ]; then
    log "log group $LOG_GROUP: deleting"; aws logs delete-log-group --log-group-name "$LOG_GROUP"
  else log "log group $LOG_GROUP: not found"; fi
  # 5. security group (Fargate network interfaces release slowly: DependencyViolation until then)
  local sg out n
  sg="$(find_sg)"
  if [ -z "$sg" ]; then log "security group $SG_NAME: not found"
  else
    log "security group $SG_NAME ($sg): deleting (waits for the task's network interface to release)"
    for n in $(seq 1 60); do
      if out="$(aws ec2 delete-security-group --group-id "$sg" 2>&1)"; then break; fi
      case "$out" in
        *DependencyViolation*) note "still in use, retrying in 10s ($n/60)"; sleep 10;;
        *InvalidGroup.NotFound*) break;;
        *) die "could not delete the security group: $out";;
      esac
      [ "$n" -lt 60 ] || die "security group $sg is still in use after 10 minutes; rerun 'down'"
    done
  fi
  # 6. IAM
  delete_role "$TASK_ROLE"; delete_role "$EXEC_ROLE"
  # 7. S3, only on request
  if [ "$delete_models" = 1 ] && [ "$BUCKET_GIVEN" = 0 ]; then
    if ! bucket_exists; then log "S3 bucket $BUCKET: not found"
    elif bucket_is_ours; then purge_bucket
    else warn "bucket $BUCKET is not tagged Project=$PROJECT_TAG; leaving it alone"; fi
  fi
  leftover_check
}

leftover_check() {
  log "leftover check"
  local left=0 r arns
  for r in "$EXEC_ROLE" "$TASK_ROLE"; do role_exists "$r" && { note "STILL EXISTS: IAM role $r"; left=1; }; done
  cluster_active && { note "STILL EXISTS: ECS cluster $CLUSTER"; left=1; }
  [ -z "$(find_sg)" ] || { note "STILL EXISTS: security group $SG_NAME"; left=1; }
  aws ecr describe-repositories --repository-names "$REPO_NAME" >/dev/null 2>&1 && { note "STILL EXISTS: ECR repository $REPO_NAME"; left=1; }
  if bucket_exists; then
    if [ "$BUCKET_GIVEN" = 1 ]; then note "kept (yours): S3 bucket $BUCKET"
    else note "kept on purpose: S3 bucket $BUCKET (rerun with --delete-models to remove it; it costs only its stored bytes)"; fi
  fi
  # The tagging API lags behind deletions by a while, so this is a hint, not a verdict.
  arns="$(aws resourcegroupstaggingapi get-resources --tag-filters "Key=Project,Values=$PROJECT_TAG" \
      --query 'ResourceTagMappingList[].ResourceARN' --output text 2>/dev/null || true)"
  for r in $arns; do
    case "$r" in "arn:aws:s3:::${BUCKET}") ;; *) note "tagged resource still listed: $r (the tagging index can lag)";; esac
  done
  [ "$left" = 0 ] && log "nothing left behind that this script created" || die "some resources remain (see above); rerun 'down'"
}

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

main() {
  local cmd="${1:-}"; [ $# -eq 0 ] || shift
  case "$cmd" in
    check) cmd_check "$@";;
    up) cmd_up "$@";;
    status) cmd_status "$@";;
    smoke) cmd_smoke "$@";;
    tunnel) cmd_tunnel "$@";;
    allow-ip) cmd_allow_ip "$@";;
    down) cmd_down "$@";;
    model) init_aws; ensure_bucket; cmd_model;;
    ""|-h|--help|help) usage;;
    *) usage; die "unknown command: $cmd";;
  esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
