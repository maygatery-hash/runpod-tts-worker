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
    accelerate

RUN pip3 install --no-cache-dir git+https://github.com/QwenLM/Qwen3-TTS.git

RUN python3 -c "from qwen_tts import Qwen3TTSModel; Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
