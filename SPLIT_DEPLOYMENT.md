# Split Architecture Deployment Guide

## Overview

This guide explains how to deploy the Voice AI system in a **split architecture** where:
- **Backend (GPU)**: Runs on DigitalOcean MI300X droplet
- **Frontend**: Runs on your local PC browser

## Prerequisites

- DigitalOcean MI300X GPU droplet (or any AMD ROCm-compatible GPU server)
- Docker and Docker Compose installed on the droplet
- Local PC with modern browser (Chrome/Edge recommended)
- Network connectivity between your PC and the droplet

---

## Part 1: Backend Deployment (DigitalOcean GPU)

### 1.1 Create and Configure Droplet

1. Create a **GPU Droplet** on DigitalOcean
   - Select **MI300X** plan
   - Choose **AI/ML-ready AMD** image (ROCm preinstalled)

2. SSH into the droplet:
   ```bash
   ssh root@<DROPLET_IP>
   ```

### 1.2 Bootstrap the System

Run the bootstrap script to mount scratch disk and install Docker:
```bash
git clone <your-repo-url>
cd solvathon_layer1-main\ 3
sudo bash deploy/do_gpu_bootstrap.sh
```

This script:
- Mounts the scratch disk at `/mnt/scratch`
- Creates cache directories
- Installs Docker + Compose plugin

### 1.3 Configure Environment

```bash
cp .env.example .env
nano .env
```

Key settings:
```bash
# Ollama (internal Docker network)
OLLAMA_BASE_URL=http://ollama:11434

# Redis (internal Docker network)
REDIS_HOST=redis
REDIS_PORT=6379

# Cache directories (use scratch disk)
HF_CACHE_DIR=/mnt/scratch/hf
TORCH_CACHE_DIR=/mnt/scratch/torch

# TTS backend
TTS_BACKEND=auto
```

### 1.4 Start Services

```bash
docker compose up --build -d
```

This starts:
- `app` — Flask server with WebRTC/WebSocket (port 8080)
- `ollama` — LLM server (internal, port 11434)
- `redis` — Session storage (internal, port 6379)
- `nginx` — Reverse proxy with CORS (ports 80, 443)

### 1.5 Pull LLM Models

```bash
# Access the Ollama container
docker exec -it solvathon_ollama bash

# Pull required models
ollama pull llama3.2:3b
ollama pull llama3.2:1b
ollama pull nomic-embed-text

# Exit
exit
```

### 1.6 Verify Backend

Wait ~60 seconds for models to load, then test:

```bash
curl http://localhost:8080/ready
```

Expected response:
```json
{
  "ready": true,
  "whisper_ready": true,
  "lid_ready": true,
  "device": "cuda"
}
```

---

## Part 2: Frontend Setup (Local PC)

### 2.1 Access the Call Interface

Open your browser and navigate to:
```
http://<DROPLET_IP>/call
```

Or if using the nginx proxy:
```
http://<DROPLET_IP>:8080/call
```

### 2.2 Configure Backend URL

In the call interface:
1. Enter the backend URL: `http://<DROPLET_IP>:8080`
2. Select language mode (Auto-detect recommended)
3. Select connection mode:
   - **WebRTC** — Low latency, requires open UDP ports
   - **WebSocket** — NAT-friendly fallback, slightly higher latency

### 2.3 Start a Call

1. Click **"Start Call"** button
2. Grant microphone permissions when prompted
3. Speak into your microphone
4. Wait ~2 seconds after speaking (silence detection triggers STT)
5. Hear AI response through your speakers

---

## Part 3: Testing & Verification

### 3.1 Run Pipeline Tests

From your local machine:

```bash
python test_pipeline.py --server http://<DROPLET_IP>:8080
```

All tests should pass:
- ✓ Server Health
- ✓ Model Readiness
- ✓ CORS Support
- ✓ ICE Configuration
- ✓ SDP Offer/Answer
- ✓ Metrics

### 3.2 Browser DevTools Debugging

Open Chrome DevTools (F12) → Console tab to see:
- WebRTC connection state changes
- ICE candidate gathering
- Audio track events
- WebSocket messages

Common issues:
- **"Mic: blocked"** — Grant microphone permissions
- **"Connection: failed"** — Check firewall, try WebSocket mode
- **"Server: loading..."** — Wait for models to load (~60s)

---

## Part 4: Firewall & Network Configuration

### 4.1 DigitalOcean Firewall Rules

Allow inbound traffic on:
- **80** — HTTP (nginx)
- **443** — HTTPS (nginx)
- **8080** — Flask app (if not using nginx proxy)
- **UDP 10000-20000** — WebRTC media (TURN fallback if blocked)

### 4.2 TURN Server (NAT Traversal)

The system uses free TURN servers (`openrelay.metered.ca`) for NAT traversal. If WebRTC fails:

1. Switch to **WebSocket mode** in the call interface
2. Or configure your own TURN server in `signaling_server.py`:
   ```python
   ICE_SERVERS = [
       RTCIceServer(urls="stun:stun.l.google.com:19302"),
       RTCIceServer(
           urls="turn:your-turn-server.com:3478",
           username="your-username",
           credential="your-password"
       )
   ]
   ```

### 4.3 SSH Tunnel (Alternative Access)

If direct access is blocked, create an SSH tunnel:

```bash
ssh -L 8080:localhost:8080 root@<DROPLET_IP>
```

Then access the interface at:
```
http://localhost:8080/call
```

---

## Part 5: Production Considerations

### 5.1 HTTPS/TLS Setup

For production, enable HTTPS in nginx:

1. Obtain SSL certificate (Let's Encrypt):
   ```bash
   docker run --rm -v /etc/letsencrypt:/etc/letsencrypt certbot/certbot \
     certonly --standalone -d yourdomain.com
   ```

2. Uncomment TLS server block in `deploy/nginx.conf`

3. Update certificate paths:
   ```nginx
   ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
   ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
   ```

4. Restart nginx:
   ```bash
   docker compose restart nginx
   ```

### 5.2 Monitoring

Monitor system health:
```bash
# View logs
docker compose logs -f app

# Check metrics
curl http://localhost:8080/metrics

# Resource usage
docker stats
```

### 5.3 Model Updates

To update LLM models:
```bash
docker exec -it solvathon_ollama ollama pull llama3.2:3b
docker compose restart app
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Models not loading | Wait 60s, check `HF_CACHE_DIR` and `TORCH_CACHE_DIR` permissions |
| WebRTC connection fails | Switch to WebSocket mode, check firewall UDP ports |
| No audio playback | Check browser autoplay policy, ensure user interaction before audio |
| High latency | Use GPU instance, check `/metrics` for bottlenecks |
| CORS errors | Verify nginx CORS headers, check browser console |
| Redis connection failed | Ensure Redis container is running: `docker ps` |
| Ollama not responding | Restart Ollama: `docker compose restart ollama` |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Local PC (Browser)                                          │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ call_interface.html                                  │  │
│  │  - Captures mic audio                                │  │
│  │  - Sends via WebRTC or WebSocket                     │  │
│  │  - Plays AI response audio                           │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                       │
└─────────────────────┼───────────────────────────────────────┘
                      │
                      │ Internet
                      │
┌─────────────────────┼───────────────────────────────────────┐
│ DigitalOcean GPU    ▼                                       │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ nginx (CORS + WebSocket proxy)                       │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                       │
│  ┌──────────────────▼───────────────────────────────────┐  │
│  │ Flask App (signaling_server.py)                      │  │
│  │  - WebRTC/WebSocket handler                          │  │
│  │  - Audio routing                                     │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                       │
│         ┌───────────┼────────────┬─────────────┐           │
│         │           │            │             │           │
│         ▼           ▼            ▼             ▼           │
│  ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ MMS-LID  │ │ Whisper │ │  Ollama  │ │ Piper/Edge   │  │
│  │ (Lang ID)│ │  (STT)  │ │  (LLM)   │ │    (TTS)     │  │
│  └──────────┘ └─────────┘ └────┬─────┘ └──────────────┘  │
│                                 │                          │
│                           ┌─────▼──────┐                   │
│                           │   Redis    │                   │
│                           │ (Context)  │                   │
│                           └────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start Summary

**On GPU Droplet:**
```bash
git clone <repo>
cd solvathon_layer1-main\ 3
sudo bash deploy/do_gpu_bootstrap.sh
cp .env.example .env
docker compose up -d
docker exec -it solvathon_ollama ollama pull llama3.2:3b
python test_pipeline.py --server http://localhost:8080
```

**On Local PC:**
```
1. Open: http://<DROPLET_IP>:8080/call
2. Enter backend URL: http://<DROPLET_IP>:8080
3. Click "Start Call"
4. Speak and enjoy!
```
