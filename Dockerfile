# ── VibeBlade — OpenAI-compatible speculative decoding proxy ────────────────
# CPU-only default. Build with --build-arg CUDA=1 for the CUDA build stage.

FROM python:3.11-slim AS base

WORKDIR /app

# Build tools for the optional C++ backend (kept, but cache-friendly layer)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY vibeblade ./vibeblade
COPY cpp ./cpp

RUN pip install --no-cache-dir . "fastapi>=0.110" "uvicorn>=0.29"

# Non-root runtime user
RUN useradd --create-home --shell /usr/sbin/nologin vibeblade
USER vibeblade

EXPOSE 8080
ENV VIBEBLADE_HOST=0.0.0.0 VIBEBLADE_PORT=8080

# Caller auth: pass --require-api-key or VIBEBLADE_API_KEY at runtime.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3).status==200 else 1)"

ENTRYPOINT ["vibeblade"]
CMD ["serve", "--backend", "openai", "--backend-url", "http://host.docker.internal:8000", "--model", "default"]

FROM base AS cuda
# CUDA stage: for NVIDIA hosts. The apt nvidia-cuda-toolkit package is too old
# on most distros for CUDA 13 - build this stage from a CUDA base image instead:
#   docker build --target cuda --build-arg BASE=nvidia/cuda:12.4.1-devel-ubuntu22.04 .
ARG BASE=nvidia/cuda:12.4.1-devel-ubuntu22.04
RUN echo "CUDA stage placeholder: rebuild cpp backend with -DENABLE_CUDA=ON on a CUDA base image (${BASE})"
FROM cuda AS final-cuda
USER vibeblade
