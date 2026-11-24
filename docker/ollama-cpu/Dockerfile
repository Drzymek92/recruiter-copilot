# syntax=docker/dockerfile:1@sha256:572cebbc2b15e6f84c077742952674889cb63cd5e9b80abdffcbe587b6f19183
# Second Chair — the bundled Ollama for the CPU compose file (#2045). GPU: compose.gpu.yaml uses
# the official ollama/ollama image instead (it carries the CUDA runtimes).
#
# The official ollama/ollama:0.40.2 image is 9.4 GB, almost all of it CUDA/Vulkan runtimes a CPU
# seat never loads. This image installs the SAME release from its official GitHub archive and
# drops those runtimes: 274 MB, same `ollama` binary, same llama-server, same CPU backends
# (measured #2045: identical tokens and report on the fixture, cold start 0.3 s).
#
# Supply chain: BuildKit's `ADD --checksum` refuses to build on a sha256 mismatch, and
# `sha256sum -c` checks it again. The sha256 is the one in the release's sha256sum.txt (equal to
# GitHub's asset digest). Bump: README "Pinned images" — change OLLAMA_VERSION + OLLAMA_SHA256
# together, and the ollama/ollama tag + digest in compose.gpu.yaml to the same version
# (tests/test_compose.py keeps them in step).

ARG BASE=python:3.12-slim-bookworm@sha256:44cf1793efa7c7a476bb0f2cff81443f4a072c90c1cb127e2924933daa3e8c64

FROM ${BASE} AS fetch
ARG OLLAMA_VERSION=0.40.2
ARG OLLAMA_SHA256=726bee78706c281b0eeef00746efe51a044d71c592c3f0b195820707f31fdf04
RUN apt-get update \
 && apt-get install -y --no-install-recommends zstd=1.5.4+dfsg2-5 \
 && rm -rf /var/lib/apt/lists/*
ADD --checksum=sha256:${OLLAMA_SHA256} \
    https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst \
    /tmp/ollama.tar.zst
# Keep bin/ollama + the top-level lib/ollama files (llama-server, ggml CPU backends, libgomp);
# drop every backend subdirectory (cuda_v12, cuda_v13, vulkan) and the links into them.
RUN echo "${OLLAMA_SHA256}  /tmp/ollama.tar.zst" | sha256sum -c - \
 && mkdir -p /opt/ollama \
 && tar --zstd -xf /tmp/ollama.tar.zst -C /opt/ollama \
 && rm /tmp/ollama.tar.zst \
 && find /opt/ollama/lib/ollama -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + \
 && find /opt/ollama/lib/ollama -xtype l -delete \
 && /opt/ollama/bin/ollama --version

# The same pinned base as the app image (shared layers on disk); it also brings ca-certificates,
# which `ollama pull` needs for the model registry. Runs as root like the official image, so the
# ollama-models volume (/root/.ollama) works with either image.
FROM ${BASE}
COPY --from=fetch /opt/ollama/bin/ollama /usr/bin/ollama
COPY --from=fetch /opt/ollama/lib/ollama /usr/lib/ollama
ENV OLLAMA_HOST=0.0.0.0:11434
EXPOSE 11434
ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]
