# GPU inference image for an AWS EC2 g4dn (NVIDIA T4) instance. Built by infra/gpu.sh for linux/amd64 from a clean
# `git archive` of the release's code revision: the build context holds only dlm/, run.py, serve.py, entrypoint.py.
FROM pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime
# The base image uses the system Python (PEP 668 'externally managed'); a container may install into it.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 \
    DLM_MODEL_DIR=/model DLM_DEVICE=cuda DLM_DTYPE=fp32
RUN python -m pip install \
    "fastapi>=0.115,<1" "uvicorn>=0.30,<1" "tokenizers==0.23.3" \
    "safetensors>=0.5,<1" "numpy>=1.26,<3" "boto3>=1.34,<2"
RUN useradd --create-home app && mkdir /model && chown app /model
WORKDIR /app
ARG CODE_REF
ENV DLM_CODE_REF=${CODE_REF}
COPY dlm /app/dlm
COPY run.py serve.py entrypoint.py /app/
RUN python -m pip freeze > /app/runtime-packages.txt
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"
ENTRYPOINT ["python", "entrypoint.py"]
