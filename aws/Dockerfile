# CPU inference image for AWS Fargate (Fargate has no GPU). Built by aws/dlm.sh from a clean `git archive`
# of the release's code revision: the build context holds only dlm/, run.py, serve.py and entrypoint.py.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    DLM_MODEL_DIR=/model DLM_DEVICE=cpu
# CPU wheels (the PyPI x86_64 wheel is the large CUDA build); fall back to PyPI, where aarch64 wheels are CPU.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0" \
 || python -m pip install "torch==2.15.0"
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
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"
ENTRYPOINT ["python", "entrypoint.py"]
