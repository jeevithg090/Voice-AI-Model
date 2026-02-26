import pandas as pd
import random
import json

random.seed(42)

df = pd.read_csv("data/metadata.csv")

splits = {"train": [], "val": [], "test": []}

for source in df["source"].unique():
    sub = df[df["source"] == source]
    speakers = list(sub["speaker_id"].unique())
    random.shuffle(speakers)

    n = len(speakers)
    train_spk = speakers[:int(0.7*n)]
    val_spk   = speakers[int(0.7*n):int(0.85*n)]
    test_spk  = speakers[int(0.85*n):]

    for _, row in sub.iterrows():
        if row["speaker_id"] in train_spk:
            splits["train"].append(row["filepath"])
        elif row["speaker_id"] in val_spk:
            splits["val"].append(row["filepath"])
        else:
            splits["test"].append(row["filepath"])

with open("data/splits.json", "w") as f:
    json.dump(splits, f)

print("Speaker-aware splits created.")
