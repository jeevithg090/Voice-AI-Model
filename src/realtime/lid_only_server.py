import asyncio
import threading
import time
import numpy as np
import librosa
import torch

from flask import Flask, request, jsonify
from flask_cors import CORS
from aiortc import RTCPeerConnection, RTCSessionDescription
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from src.config import settings

# -----------------------
# Setup Async Loop
# -----------------------
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def run_loop():
    loop.run_forever()

threading.Thread(target=run_loop, daemon=True).start()

# -----------------------
# Flask App
# -----------------------
app = Flask(__name__)
CORS(app)
pcs = set()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("🌍 Loading MMS Language ID model...")
processor = AutoFeatureExtractor.from_pretrained(settings.LID_MODEL)
model = Wav2Vec2ForSequenceClassification.from_pretrained(settings.LID_MODEL).to(device)
model.eval()
id2label = model.config.id2label

# Only allow Indian languages
INDIAN_LANGS = {'hin','tam','tel','kan','ben','guj','mal','mar','pan','ori','asm','urd'}

BUFFER_SECONDS = 10
TARGET_SR = 16000
audio_buffer = np.array([], dtype=np.float32)

# -----------------------
# Audio Processing
# -----------------------
async def process_audio_track(track):
    global audio_buffer

    print("🎙 Audio track started")

    while True:
        frame = await track.recv()
        pcm = frame.to_ndarray()

        if pcm.ndim > 1:
            pcm = pcm.mean(axis=0)

        pcm = pcm.astype(np.float32)  # DO NOT SCALE
        audio_buffer = np.concatenate([audio_buffer, pcm])

        if len(audio_buffer) >= 48000 * 20:  # 6 seconds
            segment = audio_buffer.copy()
            audio_buffer = np.array([], dtype=np.float32)

            await run_language_detection_live(segment)

async def run_language_detection_live(audio_chunk_48k):
    print("🔎 Running Language Detection...")

    try:
        audio_16k = librosa.resample(audio_chunk_48k, orig_sr=48000, target_sr=16000)

        waveform = torch.tensor(audio_16k).float()
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        predictions = sorted(
            [{'code': id2label[i], 'prob': float(p)} for i, p in enumerate(probs)],
            key=lambda x: x['prob'], reverse=True
        )

        best = predictions[0]

        print("\n" + "="*50)
        print(f"🌍 TOP LANGUAGE: {best['code']} ({best['prob']:.2f})")
        print("="*50)

    except Exception as e:
        print("❌ LID ERROR:", e)


async def run_language_detection(audio_chunk):
    print("🔎 Running Language Detection...")

    audio_16k = librosa.resample(audio_chunk, orig_sr=48000, target_sr=TARGET_SR)
    inputs = processor(audio_16k, sampling_rate=TARGET_SR, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    # Filter only Indian languages
    candidates = []
    for i, p in enumerate(probs):
        code = id2label[i]
        if code in INDIAN_LANGS:
            candidates.append((code, p.item()))

    if not candidates:
        print("🌍 No Indian language detected")
        return

    lang, conf = max(candidates, key=lambda x: x[1])
    print(f"🌍 Detected: {lang} | Confidence: {conf:.2f}")


# -----------------------
# WebRTC Offer Handling
# -----------------------
@app.route("/offer", methods=["POST"])
def offer():
    params = request.json
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            print("🎧 Receiving audio from browser")
            asyncio.run_coroutine_threadsafe(process_audio_track(track), loop)

    async def handle_offer():
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return pc.localDescription

    future = asyncio.run_coroutine_threadsafe(handle_offer(), loop)
    local_desc = future.result()

    return jsonify({"sdp": local_desc.sdp, "type": local_desc.type})


# -----------------------
# Start Server
# -----------------------
if __name__ == "__main__":
    print("🚀 LID ONLY Server Running at http://localhost:8080")
    app.run("0.0.0.0", 8080)
