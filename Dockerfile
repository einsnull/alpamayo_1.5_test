# Alpamayo 1.5 Docker - Using official PyTorch base image

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/workspace/alpamayo:${PYTHONPATH}"

# Install only what's missing from base image
RUN apt-get update && apt-get install -y \
    git curl wget vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/alpamayo

# Copy project files
COPY pyproject.toml uv.lock src/ ./

# Install Python dependencies (skip torch, torchvision - already in base)
RUN pip install --upgrade pip && \
    pip install \
        transformers==4.57.1 \
        accelerate>=1.12.0 \
        bitsandbytes peft \
        einops hydra-core hydra-colorlog \
        pandas pillow matplotlib seaborn \
        av huggingface_hub datasets \
        scipy scikit-learn tensorboard tqdm

# Install physical-ai-av
RUN pip install physical-ai-av==0.2.0 || pip install physical-ai-av || true

# Create directories and copy scripts
RUN mkdir -p /workspace/alpamayo/{data,outputs,scripts,configs} \
    /root/.cache/huggingface

COPY scripts/ /workspace/alpamayo/scripts/
COPY configs/ /workspace/alpamayo/configs/

CMD ["/bin/bash"]
