################  Base layer  ################
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
   # ← your requested tag

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

################  Python libs  ##############
# PyTorch wheel compiled for CUDA-12.1+ runs fine on 12.0 libs.
# If you hit symbol errors, drop back to torch==2.2.2+cu121
RUN pip install --no-cache-dir \
        torch==2.4.0+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
        transformers==4.40.0 \
	    typeguard==4.2.1 \
        transformer-lens==1.10.0 \
        sentencepiece==0.2.0 && \
        pre-commit && \
    pip install --no-cache-dir \
        datasets \
        transformers

################  App code  #################
COPY serve_hello.py .
CMD ["python", "serve_hello.py"]
