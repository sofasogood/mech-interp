##############################
# Stage 0 – runtime image
##############################
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# ── System deps ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
        python3-pip git curl ca-certificates \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ─────────────────────────────────────────────
# Torch 2.4.0+cu121 works with CUDA 12.4 runtime libs.
RUN pip install --no-cache-dir \
        torch==2.4.0+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
        "vllm>=0.4.3" \
        accelerate==0.28.0 \
        sentencepiece==0.2.0

# ── Expose & default cmd ───────────────────────────────────
EXPOSE 8000
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--help"]     # replaced at run-time by your shell script
