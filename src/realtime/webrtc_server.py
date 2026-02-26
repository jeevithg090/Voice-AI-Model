import asyncio
import numpy as np
import torch
import librosa

from aiortc import RTCPeerConnection, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole

from transformers import MimiModel, AutoFeatureExtractor
from src.models.emergency_classifier import MimiEmergencyClassifier


# -----------------------
# Device
# -----------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# -----------------------
# Load Mimi
# -----------------------
print("Loading Mimi model...")
mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
TARGET_SR = fe.sampling_rate


# -----------------------
# Load Emergency Classifier
# -----------------------
print("Loading Emergency Classifier...")
clf = MimiEmergencyClassifier().to(device)
clf.load_state_dict(torch.load("models/emergency_classifier.pt", map_location=device))
clf.eval()


# -----------------------
# Audio Processing
# -----------------------
async def process_audio_track(track: MediaStreamTrack):
    print("🎙 Audio track started")

    audio_buffer = np.array([], dtype=np.float32)

    while True:
        frame = await track.recv()

        # Convert frame → numpy mono float32
        pcm = frame.to_ndarray()

        if pcm.ndim > 1:
            pcm = pcm.mean(axis=0)

        pcm = pcm.astype(np.float32) / 32768.0

        # WebRTC gives 48kHz → convert to Mimi SR (usually 16kHz)
        audio_buffer = np.concatenate([audio_buffer, pcm])

        if len(audio_buffer) >= TARGET_SR:  # process every 1 sec
            chunk = audio_buffer[:TARGET_SR]
            audio_buffer = audio_buffer[TARGET_SR:]

            await run_emergency_detection(chunk)


async def run_emergency_detection(audio_chunk: np.ndarray):
    # Resample
    audio_16k = librosa.resample(audio_chunk, orig_sr=48000, target_sr=TARGET_SR)

    inputs = fe(
        raw_audio=audio_16k,
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        codes = mimi.encode(inputs["input_values"]).audio_codes
        prob = clf(codes).item()

    print(f"🚑 Emergency probability: {prob:.3f}")

    if prob > 0.7:
        print("🚨 DISTRESS DETECTED 🚨")


# -----------------------
# WebRTC Server
# -----------------------
pcs = set()

async def run_server():
    print("✅ WebRTC Audio Server Running...")

    # Keep process alive
    while True:
        await asyncio.sleep(3600)


# -----------------------
# Entry
# -----------------------
if __name__ == "__main__":
    asyncio.run(run_server())
