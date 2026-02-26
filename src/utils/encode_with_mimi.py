import os
import torch
import librosa
from tqdm import tqdm
from transformers import MimiModel, AutoFeatureExtractor

SRC_ROOT = "data/processed"
DST_ROOT = "data/processed_tokens"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

model = MimiModel.from_pretrained("kyutai/mimi").to(device)
model.eval()

feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
TARGET_SR = feature_extractor.sampling_rate

def encode_folder(label):
    src_dir = os.path.join(SRC_ROOT, label)
    dst_dir = os.path.join(DST_ROOT, label)
    os.makedirs(dst_dir, exist_ok=True)

    files = [f for f in os.listdir(src_dir) if f.endswith(".wav")]

    for fname in tqdm(files, desc=f"Encoding {label}"):
        path = os.path.join(src_dir, fname)

        audio, sr = librosa.load(path, sr=TARGET_SR)

        inputs = feature_extractor(
            raw_audio=audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            encoded = model.encode(inputs["input_values"])

        codes = encoded.audio_codes.squeeze(0).cpu()

        torch.save(
            {"acoustic_codes": codes},
            os.path.join(dst_dir, fname.replace(".wav", ".pt"))
        )

encode_folder("normal")
encode_folder("emergency")

print("All files encoded with Mimi!")
