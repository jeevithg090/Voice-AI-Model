import os
import librosa
import soundfile as sf
from tqdm import tqdm

SRC_DIR = "data/raw/normal"
DST_DIR = "data/processed/normal"
os.makedirs(DST_DIR, exist_ok=True)

TARGET_SR = 16000
CLIP_LEN = 2 * TARGET_SR  # 2 seconds

files = [f for f in os.listdir(SRC_DIR) if f.endswith(".wav")]

for fname in tqdm(files):
    path = os.path.join(SRC_DIR, fname)
    y, sr = librosa.load(path, sr=TARGET_SR)

    if len(y) > CLIP_LEN:
        y = y[:CLIP_LEN]
    else:
        y = librosa.util.fix_length(y, size=CLIP_LEN)

    sf.write(os.path.join(DST_DIR, fname), y, TARGET_SR)

