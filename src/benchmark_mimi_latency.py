import time
import torch
import librosa
from transformers import MimiModel, AutoFeatureExtractor
import os

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# Load Mimi
mimi = MimiModel.from_pretrained("kyutai/mimi").to(device)
mimi.eval()
fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

# Pick a test file
test_dir = "data/processed/normal"
test_file = os.path.join(test_dir, next(f for f in os.listdir(test_dir) if f.endswith(".wav")))
print("Testing on:", test_file)

audio, sr = librosa.load(test_file, sr=fe.sampling_rate)

# Warmup runs (important for GPU)
for _ in range(3):
    inputs = fe(raw_audio=audio, sampling_rate=fe.sampling_rate, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = mimi.encode(inputs["input_values"])

# Measure latency
runs = 20
times = []

for _ in range(runs):
    start = time.time()

    inputs = fe(raw_audio=audio, sampling_rate=fe.sampling_rate, return_tensors="pt").to(device)
    with torch.no_grad():
        encoded = mimi.encode(inputs["input_values"])

    end = time.time()
    times.append((end - start) * 1000)  # convert to ms

avg = sum(times) / len(times)

print(f"\nAverage Mimi encode latency: {avg:.2f} ms")
print(f"Min: {min(times):.2f} ms | Max: {max(times):.2f} ms")

# Extra info
codes = encoded.audio_codes
print("Output token shape:", codes.shape)
