import os
import librosa
import soundfile as sf
from tqdm import tqdm

SRC_DIR = "data/ravdess_raw"   # <-- FIXED PATH
DST_DIR = "data/processed/emergency"

os.makedirs(DST_DIR, exist_ok=True)

TARGET_SR = 16000
CLIP_LEN = 2 * TARGET_SR  # 2 seconds

VALID_EMOTIONS = {"04", "05", "06"}  # sad, angry, fearful

files_to_process = []

for root, _, files in os.walk(SRC_DIR):
    for fname in files:
        if fname.endswith(".wav"):
            parts = fname.replace(".wav", "").split("-")

            if len(parts) != 7:
                continue

            modality = parts[0]   # should be 03 (audio-only)
            channel = parts[1]    # 01 = speech
            emotion = parts[2]    # 04,05,06 are distress-like

            if modality == "03" and channel == "01" and emotion in VALID_EMOTIONS:
                files_to_process.append(os.path.join(root, fname))

print(f"Found {len(files_to_process)} candidate emergency clips.")

for path in tqdm(files_to_process):
    fname = os.path.basename(path)
    y, sr = librosa.load(path, sr=TARGET_SR)

    if len(y) > CLIP_LEN:
        y = y[:CLIP_LEN]
    else:
        y = librosa.util.fix_length(y, size=CLIP_LEN)

    out_name = f"emg_{fname}"
    sf.write(os.path.join(DST_DIR, out_name), y, TARGET_SR)

print("Emergency dataset prepared.")

