#!/usr/bin/env bash
# Tear-up / tear-down of a GPU server (EC2 g4dn, NVIDIA T4) that serves the TinyStories DLM from a Docker container.
#
#   infra/gpu.sh check                 read-only: credentials, permissions, AMI, subnet, quota reminder, local tools
#   infra/gpu.sh up [--yes]            create everything, build/push the image, launch the instance
#   infra/gpu.sh status
#   infra/gpu.sh tunnel                SSM port forward: localhost:$LOCAL_PORT -> the instance's API (Ctrl-C to close)
#   infra/gpu.sh smoke                 tunnel + wait for /healthz + one /generate, then close the tunnel
#   infra/gpu.sh stop | start          stop GPU billing (keeps the disk) / resume
#   infra/gpu.sh model                 validate LOCAL_MODEL_DIR and upload it as release MODEL_VERSION
#   infra/gpu.sh down [--delete-models] [--yes]
#
# Settings (environment): AWS_REGION (ap-northeast-3), INSTANCE_TYPE (g4dn.xlarge), VOLUME_GB (100), MODEL_VERSION,
# LOCAL_MODEL_DIR, DLM_SCHEDULE, DLM_DTYPE (fp32|fp16), MODEL_BUCKET, LOCAL_PORT (18000), SUBNET_ID.
# Credentials: AWS_PROFILE / AWS_ACCESS_KEY_ID if already set, else the `## aws_deployer_key` section of
# AWS_credential.md (git-ignored). Values are never printed, never passed on a command line, never written anywhere.
# Bash 3.2 compatible (macOS). No `set -x`: it would print credentials.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

# ---------------------------------------------------------------- settings and fixed names
PROJECT_TAG="${PROJECT_TAG:-dlm-gpu}"
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-ap-northeast-3}}"
export AWS_REGION AWS_DEFAULT_REGION="$AWS_REGION" AWS_PAGER=""
CREDENTIAL_FILE="${CREDENTIAL_FILE:-$REPO/AWS_credential.md}"
CREDENTIAL_SECTION="${CREDENTIAL_SECTION:-aws_deployer_key}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
VOLUME_GB="${VOLUME_GB:-100}"
AMI_PARAM="${AMI_PARAM:-/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id}"
PORT=8000
LOCAL_PORT="${LOCAL_PORT:-18000}"
MODEL_VERSION="${MODEL_VERSION:-}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-}"
DLM_SCHEDULE="${DLM_SCHEDULE:-}"
DLM_DTYPE="${DLM_DTYPE:-fp32}"
MODEL_BUCKET="${MODEL_BUCKET:-}"
SUBNET_ID="${SUBNET_ID:-}"

NAME_TAG="dlm-gpu"
REPO_NAME="dlm-gpu"
SG_NAME="dlm-gpu-sg"
ROLE_NAME="dlm-gpu-instance-role"
PROFILE_NAME="dlm-gpu-instance-profile"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$REPO/.venv/bin/python" ]; then PYTHON_BIN="$REPO/.venv/bin/python"; else PYTHON_BIN="python3"; fi
fi

TMP_DIRS=()
TUNNEL_PID=""
cleanup() {
  if [ -n "$TUNNEL_PID" ]; then pkill -P "$TUNNEL_PID" 2>/dev/null || true; kill "$TUNNEL_PID" 2>/dev/null || true; fi
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
none_to_empty() { case "$1" in None|null) printf '';; *) printf '%s' "$1";; esac; }

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

ACCOUNT_ID=""; REGISTRY=""; IMAGE_REPO=""; BUCKET=""; BUCKET_GIVEN=0; MODEL_URI=""
init_aws() {
  command -v aws >/dev/null || die "aws CLI not found"
  command -v jq  >/dev/null || die "jq not found"
  case "$DLM_DTYPE" in fp32|fp16) ;; *) die "DLM_DTYPE must be fp32 or fp16";; esac
  load_credentials
  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)" \
    || die "AWS rejected the credentials (aws sts get-caller-identity failed)"
  REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
  IMAGE_REPO="${REGISTRY}/${REPO_NAME}"
  if [ -n "$MODEL_BUCKET" ]; then BUCKET="$MODEL_BUCKET"; BUCKET_GIVEN=1
  else BUCKET="dlm-gpu-models-${ACCOUNT_ID}-${AWS_REGION}"; BUCKET_GIVEN=0; fi
  MODEL_URI="s3://${BUCKET}/models/tinystories/${MODEL_VERSION}"
}

# ---------------------------------------------------------------- network (default VPC only)
VPC_ID=""; SUBNET_CANDIDATES=""; SG_ID=""
find_network() {
  local raw azs az sn
  raw="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)" \
    || die "ec2:DescribeVpcs failed (permissions?)"
  VPC_ID="$(none_to_empty "$raw")"
  [ -n "$VPC_ID" ] || die "No default VPC in $AWS_REGION. Set AWS_REGION to a region that has one."
  if [ -n "$SUBNET_ID" ]; then SUBNET_CANDIDATES="$SUBNET_ID"; return 0; fi
  # Only subnets in an Availability Zone that offers the instance type (g4dn is not offered in every AZ).
  azs="$(aws ec2 describe-instance-type-offerings --location-type availability-zone \
      --filters "Name=instance-type,Values=$INSTANCE_TYPE" --query 'InstanceTypeOfferings[].Location' --output text)" \
    || die "ec2:DescribeInstanceTypeOfferings failed"
  [ -n "$(none_to_empty "$azs")" ] || die "$INSTANCE_TYPE is not offered in $AWS_REGION"
  for az in $azs; do
    sn="$(none_to_empty "$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" Name=default-for-az,Values=true \
        "Name=availability-zone,Values=$az" --query 'Subnets[0].SubnetId' --output text)")"
    [ -z "$sn" ] || SUBNET_CANDIDATES="$SUBNET_CANDIDATES $sn"
  done
  SUBNET_CANDIDATES="${SUBNET_CANDIDATES# }"
  [ -n "$SUBNET_CANDIDATES" ] || die "The default VPC has no default subnet in an AZ that offers $INSTANCE_TYPE"
}

find_sg() {
  none_to_empty "$(aws ec2 describe-security-groups \
      --filters "Name=group-name,Values=$SG_NAME" "Name=tag:Project,Values=$PROJECT_TAG" \
      --query 'SecurityGroups[0].GroupId' --output text)"
}

ensure_sg() {
  SG_ID="$(find_sg)"
  if [ -z "$SG_ID" ]; then
    log "security group $SG_NAME: creating (no inbound rules; access is through SSM only)"
    SG_ID="$(aws ec2 create-security-group --group-name "$SG_NAME" --vpc-id "$VPC_ID" \
        --description "DLM GPU inference: no inbound, SSM only" \
        --tag-specifications "ResourceType=security-group,Tags=[{Key=Project,Value=$PROJECT_TAG}]" \
        --query GroupId --output text)"
  else
    log "security group $SG_NAME: exists ($SG_ID)"
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
  aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration "LocationConstraint=$AWS_REGION" >/dev/null
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

# ---------------------------------------------------------------- IAM (instance role + profile)
role_exists() { aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; }
profile_exists() { aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1; }

ensure_iam() {
  local created=0
  if role_exists; then log "IAM role $ROLE_NAME: exists"
  else
    log "IAM role $ROLE_NAME: creating"
    aws iam create-role --role-name "$ROLE_NAME" --tags "Key=Project,Value=$PROJECT_TAG" --assume-role-policy-document \
        '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
    aws iam wait role-exists --role-name "$ROLE_NAME"; created=1
  fi
  aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
  aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
  aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name ReadDlmModelArtifacts --policy-document "$(cat <<JSON
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":"s3:ListBucket","Resource":"arn:aws:s3:::${BUCKET}",
  "Condition":{"StringLike":{"s3:prefix":["models/tinystories","models/tinystories/*"]}}},
 {"Effect":"Allow","Action":"s3:GetObject","Resource":"arn:aws:s3:::${BUCKET}/models/tinystories/*"}]}
JSON
)"
  if profile_exists; then log "instance profile $PROFILE_NAME: exists"
  else
    log "instance profile $PROFILE_NAME: creating"
    aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" --tags "Key=Project,Value=$PROJECT_TAG" >/dev/null
    aws iam wait instance-profile-exists --instance-profile-name "$PROFILE_NAME"; created=1
  fi
  if [ -z "$(none_to_empty "$(aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" --query 'InstanceProfile.Roles[0].RoleName' --output text)")" ]; then
    aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME"
    created=1
  fi
  if [ "$created" = 1 ]; then note "waiting for IAM to propagate"; sleep 15; fi
}

delete_iam() {
  local arn name
  if profile_exists; then
    log "instance profile $PROFILE_NAME: deleting"
    for name in $(none_to_empty "$(aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" --query 'InstanceProfile.Roles[].RoleName' --output text)"); do
      aws iam remove-role-from-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$name"
    done
    aws iam delete-instance-profile --instance-profile-name "$PROFILE_NAME"
  else log "instance profile $PROFILE_NAME: not found"; fi
  if role_exists; then
    log "IAM role $ROLE_NAME: deleting"
    for arn in $(aws iam list-attached-role-policies --role-name "$ROLE_NAME" --query 'AttachedPolicies[].PolicyArn' --output text); do
      aws iam detach-role-policy --role-name "$ROLE_NAME" --policy-arn "$arn"
    done
    for name in $(aws iam list-role-policies --role-name "$ROLE_NAME" --query 'PolicyNames' --output text); do
      aws iam delete-role-policy --role-name "$ROLE_NAME" --policy-name "$name"
    done
    aws iam delete-role --role-name "$ROLE_NAME"
  else log "IAM role $ROLE_NAME: not found"; fi
}

# ---------------------------------------------------------------- registry
ensure_ecr() {
  if aws ecr describe-repositories --repository-names "$REPO_NAME" >/dev/null 2>&1; then
    log "ECR repository $REPO_NAME: exists"
  else
    log "ECR repository $REPO_NAME: creating"
    aws ecr create-repository --repository-name "$REPO_NAME" --tags "Key=Project,Value=$PROJECT_TAG" >/dev/null
  fi
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

# ---------------------------------------------------------------- image (linux/amd64: the Mac is arm64, the T4 host is x86_64)
IMAGE_URI=""
build_and_push() {
  command -v docker >/dev/null || die "docker not found"
  docker info >/dev/null 2>&1 || die "The Docker daemon is not running"
  local ref short ctx
  ref="$(release_code_ref)"
  [ -n "$ref" ] || ref="$(git -C "$REPO" rev-parse HEAD)"
  git -C "$REPO" cat-file -e "${ref}^{commit}" 2>/dev/null || die "Code revision $ref is not in this clone; fetch it first"
  short="$(printf '%s' "$ref" | cut -c1-12)"
  IMAGE_URI="${IMAGE_REPO}:code-${short}-amd64"
  if [ "${FORCE_BUILD:-0}" != 1 ] && aws ecr describe-images --repository-name "$REPO_NAME" --image-ids "imageTag=code-${short}-amd64" >/dev/null 2>&1; then
    log "image $IMAGE_URI: already in ECR"; return 0
  fi
  mktmp; ctx="$MKTMP"
  # Only tracked files of that commit: AWS_credential.md, data/, .venv/ and checkpoints/ cannot get in.
  git -C "$REPO" archive "$ref" dlm run.py | tar -x -C "$ctx"
  cp "$HERE/Dockerfile" "$HERE/Dockerfile.dockerignore" "$HERE/serve.py" "$HERE/entrypoint.py" "$ctx/"
  [ ! -e "$ctx/AWS_credential.md" ] || die "credential file found in the build context"
  log "image $IMAGE_URI: building for linux/amd64 from commit $short (emulated on Apple Silicon; the first build is slow)"
  docker build --platform linux/amd64 --provenance=false --build-arg "CODE_REF=$ref" -t "$IMAGE_URI" "$ctx"
  log "image $IMAGE_URI: pushing"
  aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
  docker push "$IMAGE_URI" >/dev/null
}

# ---------------------------------------------------------------- instance
INSTANCE_ID=""; INSTANCE_STATE=""
find_instance() {  # newest non-terminated instance with our tag
  local row
  row="$(aws ec2 describe-instances --filters "Name=tag:Project,Values=$PROJECT_TAG" \
      Name=instance-state-name,Values=pending,running,stopping,stopped \
      --query 'sort_by(Reservations[].Instances[], &LaunchTime)[-1].[InstanceId,State.Name]' --output text 2>/dev/null || true)"
  INSTANCE_ID=""; INSTANCE_STATE=""
  case "$row" in ""|None*) return 0;; esac
  INSTANCE_ID="$(printf '%s' "$row" | awk '{print $1}')"; INSTANCE_STATE="$(printf '%s' "$row" | awk '{print $2}')"
}

render_userdata() {  # prints the path of the rendered file
  local tpl out
  tpl="$(cat "$HERE/bootstrap.sh.tpl")"
  tpl="${tpl//@@REGION@@/$AWS_REGION}"; tpl="${tpl//@@REGISTRY@@/$REGISTRY}"; tpl="${tpl//@@IMAGE@@/$IMAGE_URI}"
  tpl="${tpl//@@MODEL_URI@@/$MODEL_URI}"; tpl="${tpl//@@MODEL_VERSION@@/$MODEL_VERSION}"; tpl="${tpl//@@DTYPE@@/$DLM_DTYPE}"
  mktmp; out="$MKTMP/user-data.sh"; printf '%s\n' "$tpl" > "$out"
  USERDATA_FILE="$out"
}

launch_instance() {
  local ami sn tags out
  ami="$(aws ssm get-parameter --name "$AMI_PARAM" --query Parameter.Value --output text)" || die "Could not resolve the AMI from $AMI_PARAM"
  log "AMI $ami (Deep Learning Base GPU, Ubuntu 24.04), instance type $INSTANCE_TYPE, ${VOLUME_GB} GiB gp3"
  tags="{Key=Name,Value=$NAME_TAG},{Key=Project,Value=$PROJECT_TAG}"
  render_userdata
  for sn in $SUBNET_CANDIDATES; do
    note "launching in subnet $sn"
    if out="$(aws ec2 run-instances --image-id "$ami" --instance-type "$INSTANCE_TYPE" --count 1 \
        --subnet-id "$sn" --security-group-ids "$SG_ID" --associate-public-ip-address \
        --iam-instance-profile "Name=$PROFILE_NAME" \
        --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$VOLUME_GB,VolumeType=gp3,DeleteOnTermination=true}" \
        --metadata-options "HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled" \
        --instance-initiated-shutdown-behavior stop --user-data "file://$USERDATA_FILE" \
        --tag-specifications "ResourceType=instance,Tags=[$tags]" "ResourceType=volume,Tags=[$tags]" "ResourceType=network-interface,Tags=[$tags]" \
        --query 'Instances[0].InstanceId' --output text 2>&1)"; then
      INSTANCE_ID="$out"; return 0
    fi
    case "$out" in
      *InsufficientInstanceCapacity*) warn "no capacity in that AZ; trying the next one";;
      *VcpuLimitExceeded*) die "GPU vCPU quota is 0 or too low in $AWS_REGION. Request 'Running On-Demand G and VT instances' >= 4 in the Service Quotas console, then rerun.";;
      *) die "run-instances failed: $out";;
    esac
  done
  die "No Availability Zone had capacity for $INSTANCE_TYPE right now; try again later"
}

# ---------------------------------------------------------------- commands
confirm() {  # message
  [ "${ASSUME_YES:-0}" = 1 ] && return 0
  printf '%s [y/N] ' "$1"; local a; read -r a || a=""
  case "$a" in y|Y|yes|YES) return 0;; *) die "Cancelled";; esac
}

need_plugin() {
  command -v session-manager-plugin >/dev/null || die "session-manager-plugin is not installed. Run: brew install --cask session-manager-plugin"
}

cmd_check() {
  init_aws
  log "account $ACCOUNT_ID, region $AWS_REGION, instance type $INSTANCE_TYPE"
  local ok=1 t ami
  for t in docker jq git curl; do command -v "$t" >/dev/null && note "tool $t: ok" || { note "tool $t: MISSING"; ok=0; }; done
  docker info >/dev/null 2>&1 && note "docker daemon: running" || { note "docker daemon: NOT running"; ok=0; }
  command -v session-manager-plugin >/dev/null && note "session-manager-plugin: ok" \
    || { note "session-manager-plugin: MISSING (brew install --cask session-manager-plugin)"; ok=0; }
  find_network; note "default VPC $VPC_ID; subnets in AZs offering $INSTANCE_TYPE: $SUBNET_CANDIDATES"
  ami="$(aws ssm get-parameter --name "$AMI_PARAM" --query Parameter.Value --output text 2>/dev/null || true)"
  if [ -n "$ami" ]; then note "AMI resolves: $ami ($(aws ec2 describe-images --image-ids "$ami" --query 'Images[0].Name' --output text 2>/dev/null))"
  else note "AMI parameter did not resolve"; ok=0; fi
  probe() { if "$@" >/dev/null 2>&1; then note "permission $PROBE_NAME: ok"; else note "permission $PROBE_NAME: DENIED or failing"; ok=0; fi; }
  PROBE_NAME="iam"  probe aws iam list-roles --max-items 1
  PROBE_NAME="ec2"  probe aws ec2 describe-instances --max-results 5
  PROBE_NAME="ecr"  probe aws ecr describe-repositories --max-items 1
  PROBE_NAME="s3"   probe aws s3api list-buckets
  # EC2 dry run: no instance is created. DryRunOperation = allowed; it also reveals a missing quota.
  if [ -n "$ami" ]; then
    local sn out; sn="${SUBNET_CANDIDATES%% *}"
    out="$(aws ec2 run-instances --dry-run --image-id "$ami" --instance-type "$INSTANCE_TYPE" --subnet-id "$sn" 2>&1 || true)"
    case "$out" in
      *DryRunOperation*) note "ec2:RunInstances dry run: permitted";;
      *VcpuLimitExceeded*) note "ec2:RunInstances dry run: GPU vCPU quota too low in $AWS_REGION (request >= 4)"; ok=0;;
      *UnauthorizedOperation*|*AccessDenied*) note "ec2:RunInstances dry run: DENIED"; ok=0;;
      *) note "ec2:RunInstances dry run: unclear: $(printf '%s' "$out" | head -c 200)";;
    esac
  fi
  local q
  if q="$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --query 'Quota.Value' --output text 2>/dev/null)"; then
    note "GPU vCPU quota (Running On-Demand G and VT): $q (needs >= 4)"
    case "$q" in 0|0.0) ok=0;; esac
  else
    note "GPU vCPU quota: cannot be read with this key; check 'Running On-Demand G and VT instances' (>= 4) in the console"
  fi
  note "not checkable here: ssm:StartSession (the tunnel) and iam:PassRole; the first live 'up' / 'tunnel' proves them"
  [ "$ok" = 1 ] && log "check passed (read-only; write permissions are only proven by 'up')" || die "check found problems (see above)"
}

cmd_up() {
  while [ $# -gt 0 ]; do case "$1" in --yes) ASSUME_YES=1;; *) die "unknown option $1";; esac; shift; done
  init_aws
  [ -n "$MODEL_VERSION" ] || die "Set MODEL_VERSION (and LOCAL_MODEL_DIR the first time) so the server has a release to load"
  need_plugin; find_network
  log "Tear-up plan: account $ACCOUNT_ID, region $AWS_REGION"
  note "creates (tagged Project=$PROJECT_TAG): S3 bucket (if none given), IAM role + instance profile, ECR repository,"
  note "security group (no inbound), and one $INSTANCE_TYPE instance with ${VOLUME_GB} GiB gp3 in the default VPC."
  note "model release: $MODEL_VERSION   dtype: $DLM_DTYPE"
  note "The instance bills per second while it runs (g4dn.xlarge is about \$0.5-0.8/hour depending on region; check your region's price)."
  note "'infra/gpu.sh stop' ends GPU billing; 'infra/gpu.sh down' removes everything."
  confirm "Create these resources?"
  ensure_bucket; ensure_iam; ensure_ecr; ensure_sg
  [ -z "$LOCAL_MODEL_DIR" ] || cmd_model
  release_present || die "No model release $MODEL_VERSION in s3://$BUCKET. Set LOCAL_MODEL_DIR and rerun."
  build_and_push
  find_instance
  if [ -n "$INSTANCE_ID" ]; then
    log "instance $INSTANCE_ID already exists ($INSTANCE_STATE); not launching another"
    [ "$INSTANCE_STATE" != stopped ] || cmd_start_inner
  else
    launch_instance
    log "instance $INSTANCE_ID: waiting for running state"
    aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
    log "instance $INSTANCE_ID: waiting for status checks"
    aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID"
  fi
  log "instance is up. The container needs a few more minutes (image pull, model download). Next: infra/gpu.sh smoke"
}

cmd_status() {
  init_aws
  log "account $ACCOUNT_ID, region $AWS_REGION"
  local sg
  find_instance
  note "instance:     ${INSTANCE_ID:-absent}${INSTANCE_STATE:+ ($INSTANCE_STATE)}"
  sg="$(find_sg)"; note "security grp: ${sg:-absent}"
  note "ECR repo:     $(aws ecr describe-repositories --repository-names "$REPO_NAME" >/dev/null 2>&1 && echo present || echo absent)"
  note "S3 bucket:    $BUCKET ($(bucket_exists && echo present || echo absent))"
  note "IAM:          role $(role_exists && echo present || echo absent), profile $(profile_exists && echo present || echo absent)"
  [ -z "$INSTANCE_ID" ] || note "next:         infra/gpu.sh smoke   (needs session-manager-plugin)"
}

cmd_start_inner() {
  log "instance $INSTANCE_ID: starting"
  aws ec2 start-instances --instance-ids "$INSTANCE_ID" >/dev/null
  aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
  aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID"
}
cmd_start() { init_aws; find_instance; [ -n "$INSTANCE_ID" ] || die "No instance"; [ "$INSTANCE_STATE" = stopped ] || die "Instance is $INSTANCE_STATE"; cmd_start_inner
  log "started. The container restarts by itself (--restart unless-stopped); give it a minute, then: infra/gpu.sh smoke"; }
cmd_stop() {
  init_aws; find_instance; [ -n "$INSTANCE_ID" ] || die "No instance"
  log "instance $INSTANCE_ID: stopping (GPU billing ends; the ${VOLUME_GB} GiB disk keeps billing)"
  aws ec2 stop-instances --instance-ids "$INSTANCE_ID" >/dev/null
  aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID"
}

ssm_forward() {  # runs in the foreground
  aws ssm start-session --target "$INSTANCE_ID" --document-name AWS-StartPortForwardingSession \
      --parameters "{\"portNumber\":[\"$PORT\"],\"localPortNumber\":[\"$LOCAL_PORT\"]}"
}

cmd_tunnel() {
  init_aws; need_plugin; find_instance
  [ "$INSTANCE_STATE" = running ] || die "No running instance (state: ${INSTANCE_STATE:-none})"
  log "forwarding localhost:$LOCAL_PORT -> $INSTANCE_ID port $PORT (Ctrl-C to close)"
  ssm_forward
}

cmd_smoke() {
  init_aws; need_plugin; find_instance
  [ "$INSTANCE_STATE" = running ] || die "No running instance (state: ${INSTANCE_STATE:-none})"
  ssm_forward >/dev/null &
  TUNNEL_PID=$!
  local base="http://127.0.0.1:$LOCAL_PORT" i
  log "waiting for $base/healthz through the SSM tunnel (the first start can take ~10 minutes)"
  for i in $(seq 1 60); do
    kill -0 "$TUNNEL_PID" 2>/dev/null || die "The SSM tunnel exited (permissions for ssm:StartSession? plugin? agent not registered yet?)"
    curl -fsS --max-time 5 "$base/healthz" 2>/dev/null && { echo; break; }
    [ "$i" -lt 60 ] || die "The API did not become healthy; inspect /var/log/dlm-bootstrap.log and 'docker logs dlm-api' on the instance (aws ssm start-session --target $INSTANCE_ID)"
    sleep 10
  done
  log "POST $base/generate"
  curl -fsS --max-time 300 "$base/generate" -H 'Content-Type: application/json' \
      -d '{"mode":"continue","prompt":"Once upon a time, a little fox","max_new_tokens":64,"steps":64,"temperature":0,"seed":0}'; echo
}

cmd_down() {
  local delete_models=0
  while [ $# -gt 0 ]; do case "$1" in --delete-models) delete_models=1;; --yes) ASSUME_YES=1;; *) die "unknown option $1";; esac; shift; done
  init_aws
  log "Tear-down plan: account $ACCOUNT_ID, region $AWS_REGION"
  note "deletes: the instance (and its disk), security group, instance profile + IAM role, ECR repository (images)."
  if [ "$delete_models" = 1 ]; then
    if [ "$BUCKET_GIVEN" = 1 ]; then note "S3: $BUCKET is YOUR bucket; it and its model releases are left alone."
    else note "S3: ALSO deletes bucket $BUCKET with every model release in it (irreversible)."; fi
  else
    note "S3: bucket $BUCKET and the model releases in it are kept (use --delete-models to remove a script-created bucket)."
  fi
  confirm "Delete these resources?"

  # 1. instances (every non-terminated instance carrying our tag)
  local id ids
  ids="$(none_to_empty "$(aws ec2 describe-instances --filters "Name=tag:Project,Values=$PROJECT_TAG" \
      Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text)")"
  if [ -z "$ids" ]; then log "instance: not found"
  else
    for id in $ids; do log "instance $id: terminating"; aws ec2 terminate-instances --instance-ids "$id" >/dev/null; done
    # shellcheck disable=SC2086
    aws ec2 wait instance-terminated --instance-ids $ids
  fi
  # 2. tagged volumes that outlived their instance (DeleteOnTermination should prevent this)
  local vol
  for vol in $(none_to_empty "$(aws ec2 describe-volumes --filters "Name=tag:Project,Values=$PROJECT_TAG" Name=status,Values=available --query 'Volumes[].VolumeId' --output text)"); do
    log "volume $vol: deleting (left over, unattached)"; aws ec2 delete-volume --volume-id "$vol" >/dev/null
  done
  # 3. security group (the instance's network interface releases a little after termination)
  local sg out n
  sg="$(find_sg)"
  if [ -z "$sg" ]; then log "security group $SG_NAME: not found"
  else
    log "security group $SG_NAME ($sg): deleting"
    for n in $(seq 1 30); do
      if out="$(aws ec2 delete-security-group --group-id "$sg" 2>&1)"; then break; fi
      case "$out" in
        *DependencyViolation*) note "still in use, retrying in 10s ($n/30)"; sleep 10;;
        *InvalidGroup.NotFound*) break;;
        *) die "could not delete the security group: $out";;
      esac
      [ "$n" -lt 30 ] || die "security group $sg is still in use after 5 minutes; rerun 'down'"
    done
  fi
  # 4. IAM, 5. registry
  delete_iam
  if aws ecr describe-repositories --repository-names "$REPO_NAME" >/dev/null 2>&1; then
    log "ECR repository $REPO_NAME: deleting with its images"
    aws ecr delete-repository --repository-name "$REPO_NAME" --force >/dev/null
  else log "ECR repository $REPO_NAME: not found"; fi
  command -v docker >/dev/null && docker logout "$REGISTRY" >/dev/null 2>&1 || true
  # 6. S3, only on request
  if [ "$delete_models" = 1 ] && [ "$BUCKET_GIVEN" = 0 ]; then
    if ! bucket_exists; then log "S3 bucket $BUCKET: not found"
    elif bucket_is_ours; then purge_bucket
    else warn "bucket $BUCKET is not tagged Project=$PROJECT_TAG; leaving it alone"; fi
  fi
  leftover_check
}

leftover_check() {
  log "leftover check (direct describe calls)"
  local left=0 out
  find_instance; [ -z "$INSTANCE_ID" ] || { note "STILL EXISTS: instance $INSTANCE_ID ($INSTANCE_STATE)"; left=1; }
  out="$(none_to_empty "$(aws ec2 describe-volumes --filters "Name=tag:Project,Values=$PROJECT_TAG" --query 'Volumes[].VolumeId' --output text)")"
  [ -z "$out" ] || { note "STILL EXISTS: volumes $out"; left=1; }
  out="$(none_to_empty "$(aws ec2 describe-addresses --filters "Name=tag:Project,Values=$PROJECT_TAG" --query 'Addresses[].AllocationId' --output text)")"
  [ -z "$out" ] || { note "STILL EXISTS: Elastic IPs $out"; left=1; }
  [ -z "$(find_sg)" ] || { note "STILL EXISTS: security group $SG_NAME"; left=1; }
  profile_exists && { note "STILL EXISTS: instance profile $PROFILE_NAME"; left=1; }
  role_exists && { note "STILL EXISTS: IAM role $ROLE_NAME"; left=1; }
  aws ecr describe-repositories --repository-names "$REPO_NAME" >/dev/null 2>&1 && { note "STILL EXISTS: ECR repository $REPO_NAME"; left=1; }
  if bucket_exists; then
    if [ "$BUCKET_GIVEN" = 1 ]; then note "kept (yours): S3 bucket $BUCKET"
    else note "kept on purpose: S3 bucket $BUCKET (rerun with --delete-models to remove it; it costs only its stored bytes)"; fi
  fi
  [ "$left" = 0 ] && log "nothing left behind that this script created" || die "some resources remain (see above); rerun 'down'"
}

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }

main() {
  local cmd="${1:-}"; [ $# -eq 0 ] || shift
  case "$cmd" in
    check) cmd_check "$@";;
    up) cmd_up "$@";;
    status) cmd_status "$@";;
    tunnel) cmd_tunnel "$@";;
    smoke) cmd_smoke "$@";;
    stop) cmd_stop "$@";;
    start) cmd_start "$@";;
    model) init_aws; ensure_bucket; cmd_model;;
    down) cmd_down "$@";;
    ""|-h|--help|help) usage;;
    *) usage; die "unknown command: $cmd";;
  esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
