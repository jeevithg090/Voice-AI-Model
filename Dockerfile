FROM python:3.10-slim

ARG ROCM_VERSION=7.1
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/rocm${ROCM_VERSION}
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
  && python -m pip install -r /app/requirements.txt

COPY . /app

EXPOSE 8080
CMD ["python", "src/realtime/signaling_server.py"]
