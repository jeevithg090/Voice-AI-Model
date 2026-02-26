import os
import librosa
import soundfile as sf
from tqdm import tqdm

SRC_DIR = "data/ravdess_raw"
NORMAL_DIR = "data/processed/normal"
EMG_DIR = "data/processed/emergency"

TARGET_SR = 16000

for root, _, files in os.walk(SRC_DIR):
    for fname in tqdm(files):
        if not fname.endswith(".wav"):
            continue

        parts = fname.replace(".wav", "").split("-")
        if len(parts) != 7:
            continue

        modality, channel, emotion, intensity, statement, rep, actor = parts

        if modality != "03" or channel != "01":
            continue

        src_path = os.path.join(root, fname)
        y, sr = librosa.load(src_path, sr=TARGET_SR)

        # Keep full audio, only trim silence at ends
        y_trim, _ = librosa.effects.trim(y, top_db=25)

        if emotion in ["01", "02"]:
            out_dir = NORMAL_DIR
        elif emotion in ["04", "05", "06"]:
            out_dir = EMG_DIR
        else:
            continue

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"ravdess_{actor}_{fname}")

        sf.write(out_path, y_trim, TARGET_SR)

