FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    runpod \
    boto3 \
    pyloudnorm \
    scipy \
    soundfile \
    transformers \
    accelerate \
    huggingface_hub \
    requests

# Clone repository directly and install in editable mode
RUN git clone https://github.com/QwenLM/Qwen3-TTS.git /app/Qwen3-TTS && \
    cd /app/Qwen3-TTS && \
    pip3 install --no-cache-dir -e .

ENV PYTHONPATH="/app/Qwen3-TTS:${PYTHONPATH}"

# Download model weights to HF cache
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-Base'); snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
