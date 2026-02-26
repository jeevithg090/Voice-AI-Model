import os
import random
import torch
import librosa
from transformers import MimiModel, AutoFeatureExtractor
from src.models.emergency_classifier import MimiEmergencyClassifier

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load models
mimi = MimiModel.from_pretrained("kyutai/mimi").to(device)
mimi.eval()
fe = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

clf = MimiEmergencyClassifier().to(device)
clf.load_state_dict(torch.load("models/emergency_classifier.pt", map_location=device))
clf.eval()

# Pick random file
normal_dir = "data/processed/normal"
emg_dir = "data/processed/emergency"

all_files = [(os.path.join(normal_dir, f), 0) for f in os.listdir(normal_dir) if f.endswith(".wav")]
all_files += [(os.path.join(emg_dir, f), 1) for f in os.listdir(emg_dir) if f.endswith(".wav")]

path, true_label = random.choice(all_files)

print(f"\nTesting file: {path}")
print("True label:", "EMERGENCY" if true_label == 1 else "NORMAL")

# Load audio
audio, sr = librosa.load(path, sr=fe.sampling_rate)

# Encode with Mimi
inputs = fe(raw_audio=audio, sampling_rate=fe.sampling_rate, return_tensors="pt").to(device)

with torch.no_grad():
    codes = mimi.encode(inputs["input_values"]).audio_codes
    prob = clf(codes).item()

print(f"Predicted emergency probability: {prob:.3f}")

if prob > 0.5:
    print("🚨 Model says: EMERGENCY")
else:
    print("✅ Model says: NORMAL")
