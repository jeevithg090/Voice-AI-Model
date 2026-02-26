import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

CV_ROOT = "data/commonvoice"
CSV_PATH = os.path.join(CV_ROOT, "cv-valid-train.csv")
AUDIO_DIR = os.path.join(CV_ROOT, "cv-valid-train")
OUT_DIR = "data/processed/normal"

TARGET_SR = 16000

df = pd.read_csv(CSV_PATH)
df = df[df["accent"] == "indian"]

print(f"Found {len(df)} Indian-accent clips")

os.makedirs(OUT_DIR, exist_ok=True)

for _, row in tqdm(df.iterrows(), total=len(df)):
    path = os.path.join(AUDIO_DIR, row["filename"])
    if not os.path.exists(path):
        continue

    y, sr = librosa.load(path, sr=TARGET_SR)
    y_trim, _ = librosa.effects.trim(y, top_db=25)

    if len(y_trim) < TARGET_SR * 2:
        continue

    fname = f"cv_{row['filename'].replace('/', '_').replace('.mp3','.wav')}"
    sf.write(os.path.join(OUT_DIR, fname), y_trim, TARGET_SR)
