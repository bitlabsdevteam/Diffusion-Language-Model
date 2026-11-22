"""Container entrypoint: fetch the model release from S3 (when DLM_MODEL_URI is set), verify it, start the API.

The task role provides the S3 credentials; nothing is stored in the image. `deployment.json` is the release marker
and carries a SHA-256 per file, so a partial or tampered download stops the container instead of serving.
"""
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath


def split_uri(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"DLM_MODEL_URI must start with s3://, got {uri!r}")
    bucket, _, prefix = uri[5:].partition("/")
    if not bucket or not prefix.strip("/"):
        raise ValueError("DLM_MODEL_URI must be s3://<bucket>/<prefix>")
    return bucket, prefix.strip("/") + "/"


def safe_target(root: Path, relative: str) -> Path:
    """Destination of an S3 key below `root`; refuses absolute paths and '..'."""
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or ".." in parts:
        raise ValueError(f"Unsafe object key: {relative!r}")
    return root.joinpath(*parts)


def download(client, uri: str, root: Path) -> int:
    bucket, prefix = split_uri(uri)
    count = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            target = safe_target(root, key[len(prefix):])
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
            count += 1
    if not count:
        raise FileNotFoundError(f"No objects under {uri}")
    return count


def verify(root: Path, expected_version: str | None) -> dict:
    marker = root / "deployment.json"
    if not marker.is_file():
        raise FileNotFoundError("deployment.json is missing: the release upload is incomplete")
    release = json.loads(marker.read_text())
    if expected_version and release["model_version"] != expected_version:
        raise ValueError(f"Release is {release['model_version']!r}, expected {expected_version!r}")
    if release["schedule"] not in {"linear", "cosine"}:
        raise ValueError("deployment.json must specify linear or cosine")
    for name, digest in release["sha256"].items():
        path = safe_target(root, name)
        if not path.is_file():
            raise FileNotFoundError(f"Release file missing: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    return release


def main() -> None:
    root = Path(os.environ.get("DLM_MODEL_DIR", "/model"))
    uri = os.environ.get("DLM_MODEL_URI", "")
    if uri:
        import boto3

        root.mkdir(parents=True, exist_ok=True)
        n = download(boto3.client("s3"), uri, root)
        print(f"downloaded {n} files from {uri}", flush=True)
    release = verify(root, os.environ.get("MODEL_VERSION") or None)
    print(f"verified release {release['model_version']} (schedule {release['schedule']}, "
          f"code {release['code_ref']})", flush=True)
    os.execvp("python", ["python", "-m", "uvicorn", "serve:app", "--host", "0.0.0.0",
                         "--port", os.environ.get("PORT", "8000"), "--workers", "1"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - one clear line in the task logs, non-zero exit
        print(f"entrypoint failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
