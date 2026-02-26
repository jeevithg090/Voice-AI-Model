# Solvathon Layer 1: Real-Time Voice AI

> **Low-latency, multilingual, empathetic voice intelligence for emergency response.**

This repository implements the **first layer** of a multi-tiered emergency response AI. It handles the immediate vocal interaction with a caller, determining their language, emotional state, and intent in real-time.

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **Ollama** running locally (`llama3.2:3b` model pulled).
- **Redis** running locally.
- **Outbound internet** for Edge TTS fallback (Kannada primary).

### 2. Installation
```bash
# Clone the repo
git clone <repo-url>
cd solvathon_layer1

# Install Dependencies
pip install -r requirements.txt
python3 -m pip install piper-tts

# Download TTS Models
chmod +x src/tts/setup_piper.sh
./src/tts/setup_piper.sh

# Optional: place custom Tamil Piper model in repo root before setup:
# ta_IN-iitm-female-s1-medium.onnx
# ta_IN-iitm-female-s1-medium.onnx.json

# Optional config
cp .env.example .env
```

### 3. Run
```bash
python src/realtime/signaling_server.py
```
Open [http://localhost:8080](http://localhost:8080) in your browser.
Admin panel is available at [http://localhost:8080/admin](http://localhost:8080/admin).
Press-to-talk upload UI is available at [http://localhost:8080/push-to-talk](http://localhost:8080/push-to-talk).

### 4. Admin Panel (New)
- Create and manage **voice agent templates**.
- Create multiple **agents** with per-agent config (tone, objective, generation params).
- Activate one agent as the production runtime profile.
- View conversation history with transcript and recording playback.
- API routes:
  - `GET/POST /admin/templates`
  - `GET/POST /admin/agents`
  - `POST /admin/agents/<id>/activate`
  - `GET /admin/conversations`
  - `GET /admin/conversations/<id>`

---

## 📚 Documentation
For detailed system architecture, configuration, and API reference, please see the **[Technical Manual](./TECHNICAL_MANUAL.md)**.
For GPU deployment on DigitalOcean, see **[DEPLOYMENT.md](./DEPLOYMENT.md)**.

## ✨ Key Features
- **🗣️ Multilingual**: Fluent in English, Hindi, Tamil, Telugu, Kannada.
- **⚡ Real-time**: <500ms latency using streamed LLM tokens.
- **🚑 Emergency Aware**: Detects screaming/distress and switches to "Empathy Mode".
- **🔊 TTS**: Piper (English/Hindi/Telugu/Tamil when model is present) + Edge TTS (Kannada primary fallback path).
- **🎯 Purpose Routing**: Built-in voice agent purposes for Hospital Kiosk, College Admission, and Laptop Customer Support (`agent_type`).
