import time
import torch
import librosa
from transformers import MimiModel, AutoFeatureExtractor
from src.models.emergency_classifier import MimiEmergencyClassifier

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)

# Load models
mimi = MimiModel.from_pretrained("kyutai/mimi").to(device)
mimi.eval()
fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

clf = MimiEmergencyClassifier().to(device)
clf.load_state_dict(torch.load("models/emergency_classifier.pt", map_location=device))
clf.eval()

# Test file (change if you want)
test_file = "data/processed/emergency/" + next(f for f in __import__("os").listdir("data/processed/emergency") if f.endswith(".wav"))
print("Testing on:", test_file)

audio, sr = librosa.load(test_file, sr=fe.sampling_rate)

# Warmup (important for GPU timing)
for _ in range(3):
    inputs = fe(raw_audio=audio, sampling_rate=fe.sampling_rate, return_tensors="pt").to(device)
    with torch.no_grad():
        codes = mimi.encode(inputs["input_values"]).audio_codes
        _ = clf(codes)

# Measure
runs = 20
times = []

for _ in range(runs):
    start = time.time()

    inputs = fe(raw_audio=audio, sampling_rate=fe.sampling_rate, return_tensors="pt").to(device)
    with torch.no_grad():
        codes = mimi.encode(inputs["input_values"]).audio_codes
        prob = clf(codes)

    end = time.time()
    times.append((end - start) * 1000)  # ms

avg = sum(times) / len(times)
print(f"\nAverage emergency detection latency: {avg:.2f} ms")
print(f"Min: {min(times):.2f} ms | Max: {max(times):.2f} ms")
print(f"Predicted probability: {prob.item():.3f}")
