import os
import csv

ROOT = "data/processed_tokens"
OUT_CSV = "data/metadata.csv"

rows = []

def get_speaker_id(fname):
    if fname.startswith("ravdess"):
        # ravdess_01_03-01-05-01-01-02-01.pt
        return fname.split("_")[1]  # actor id
    else:
        # Common Voice — use filename as pseudo speaker
        return fname.split("_")[-1].replace(".pt", "")

for label_name, label in [("normal", 0), ("emergency", 1)]:
    folder = os.path.join(ROOT, label_name)
    for fname in os.listdir(folder):
        if not fname.endswith(".pt"):
            continue

        speaker_id = get_speaker_id(fname)
        source = "ravdess" if fname.startswith("ravdess") else "commonvoice"

        rows.append([
            os.path.join(ROOT, label_name, fname),
            label,
            speaker_id,
            source
        ])

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "label", "speaker_id", "source"])
    writer.writerows(rows)

print(f"Metadata saved with {len(rows)} samples.")
