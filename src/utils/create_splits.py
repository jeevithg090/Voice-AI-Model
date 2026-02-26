import os, random, json

random.seed(42)

root = "data/processed_tokens"
splits = {"train": [], "val": [], "test": []}

for label in ["normal", "emergency"]:
    files = [os.path.join(label, f) for f in os.listdir(os.path.join(root, label)) if f.endswith(".pt")]
    random.shuffle(files)

    n = len(files)
    splits["train"] += files[:int(0.7*n)]
    splits["val"]   += files[int(0.7*n):int(0.85*n)]
    splits["test"]  += files[int(0.85*n):]

with open("data/splits.json", "w") as f:
    json.dump(splits, f)

print("Splits created.")

