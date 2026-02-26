import os
import librosa
import soundfile as sf
from tqdm import tqdm

SRC_DIR = "data/ravdess_raw"
DST_DIR = "data/processed/normal"

os.makedirs(DST_DIR, exist_ok=True)

TARGET_SR = 16000
CLIP_LEN = 2 * TARGET_SR

VALID_EMOTIONS = {"01", "02", "03"}  # neutral, calm

files = []

for root, _, filenames in os.walk(SRC_DIR):
    for fname in filenames:
        if fname.endswith(".wav"):
            parts = fname.replace(".wav", "").split("-")
            if len(parts) != 7:
                continue
            modality, channel, emotion = parts[0], parts[1], parts[2]
            if modality == "03" and channel == "01" and emotion in VALID_EMOTIONS:
                files.append(os.path.join(root, fname))

print(f"Adding {len(files)} neutral/calm files to NORMAL class")

for path in tqdm(files):
    fname = "neutral_" + os.path.basename(path)
    y, sr = librosa.load(path, sr=TARGET_SR)

    if len(y) > CLIP_LEN:
        y = y[:CLIP_LEN]
    else:
        y = librosa.util.fix_length(y, size=CLIP_LEN)

    sf.write(os.path.join(DST_DIR, fname), y, TARGET_SR)

