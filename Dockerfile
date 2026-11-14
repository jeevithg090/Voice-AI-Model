FROM python:3.10-slim

ARG ROCM_VERSION=7.1
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/rocm${ROCM_VERSION} \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

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
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -fsS http://localhost:8080/healthz || exit 1

# Single gevent worker: the app keeps per-process model + asyncio state.
CMD ["gunicorn", "-k", "gevent", "-w", "1", "--worker-connections", "1000", "--timeout", "0", \
     "-b", "0.0.0.0:8080", "src.realtime.signaling_server:app"]
