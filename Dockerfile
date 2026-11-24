# syntax=docker/dockerfile:1
# Second Chair (D27) — the app image for compose.yaml (CPU, default) and compose.gpu.yaml (GPU).
#
#   docker compose build                                         # target cpu  → second-chair:cpu
#   docker compose -f compose.yaml -f compose.gpu.yaml build     # target gpu  → second-chair:gpu
#
# D14 (amended 11-23): inside the image `serve` binds the container wildcard 0.0.0.0, allowed ONLY
# because SECOND_CHAIR_IN_CONTAINER=1 is set here; the loopback boundary is the HOST edge —
# compose publishes 127.0.0.1:<port> only (pinned by tests/test_compose.py). A bare `serve`
# outside this image never has the flag and still refuses any non-loopback host.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# ── base: the app + the [local] extra (faster-whisper on CPU, Ollama chat over HTTP) ──────────
FROM ${PYTHON_IMAGE} AS base
ARG APP_UID=1000
ARG APP_GID=1000
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# Non-root. UID/GID default 1000 so files written on the host bind mount belong to the usual
# first desktop user; override with --build-arg APP_UID=$(id -u) APP_GID=$(id -g).
RUN groupadd -g "${APP_GID}" app \
 && useradd -m -u "${APP_UID}" -g app -s /usr/sbin/nologin app
WORKDIR /opt/second-chair
COPY pyproject.toml README.md ./
COPY src ./src
# Pins are owned by pyproject.toml (project.md Policy Overrides); nothing else is installed.
RUN pip install ".[local]" && rm -rf /opt/second-chair/build
ENV SECOND_CHAIR_IN_CONTAINER=1 \
    RECRUITER_COPILOT_HOST=0.0.0.0 \
    PORT=8765 \
    SECOND_CHAIR_WORKSPACE=/workspace \
    SESSIONS_DIR=/workspace/sessions \
    HF_HOME=/home/app/.cache/huggingface
# /workspace is a host bind mount (D18 purge works on the host folder); the HF cache is a named
# volume — both pre-created here so a fresh named volume inherits the app user's ownership.
RUN mkdir -p /workspace "${HF_HOME}" \
 && chown -R app:app /workspace /home/app/.cache
USER app
WORKDIR /workspace
EXPOSE 8765
# The Host header is a loopback name, which the workspace app requires (D14).
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8765') + '/', timeout=4)"]
ENTRYPOINT ["second-chair"]
CMD ["serve"]

# ── gpu: + cuBLAS 12 and cuDNN 9, which faster-whisper's ctranslate2 needs on CUDA ────────────
# Pip wheels, not an nvidia/cuda base: the GPU image stays the same slim Debian + Python as the
# CPU one (one Dockerfile, one tested code path), and these exact versions are the ones the lab
# host already runs on its Blackwell (sm_120) card with torch cu128. The driver itself (libcuda)
# is injected at run time by the NVIDIA container toolkit (compose.gpu.yaml device reservation).
FROM base AS gpu
USER root
RUN pip install "nvidia-cublas-cu12==12.8.4.1" "nvidia-cudnn-cu12==9.19.0.56"
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER app

# ── cpu (last = the default target) ─────────────────────────────────────────────────────────
FROM base AS cpu
