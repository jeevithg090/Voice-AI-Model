import os
import subprocess

SRC_DIR = "data/commonvoice/cv-valid-train/cv-valid-train"
DST_DIR = "data/raw/normal"

os.makedirs(DST_DIR, exist_ok=True)

for fname in os.listdir(SRC_DIR):
    if fname.endswith(".mp3"):
        src_path = os.path.join(SRC_DIR, fname)
        dst_path = os.path.join(DST_DIR, fname.replace(".mp3", ".wav"))

        cmd = [
            "ffmpeg", "-loglevel", "quiet", "-y",
            "-i", src_path,
            "-ar", "16000",
            "-ac", "1",
            dst_path
        ]

        subprocess.run(cmd)

print("Conversion complete!")