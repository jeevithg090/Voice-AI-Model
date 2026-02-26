import torch
from torch.utils.data import DataLoader
from src.data.mimi_dataset import MimiEmergencyDataset
from src.models.emergency_classifier import MimiEmergencyClassifier
from src.data.collate import mimi_collate   # ✅ ADD THIS

device = "mps" if torch.backends.mps.is_available() else "cpu"

test_ds = MimiEmergencyDataset("test")
loader = DataLoader(test_ds, batch_size=32, collate_fn=mimi_collate)  # ✅ USE COLLATE

model = MimiEmergencyClassifier().to(device)
model.load_state_dict(torch.load("models/emergency_classifier.pt", map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for codes, labels in loader:
        codes, labels = codes.to(device), labels.to(device)
        preds = model(codes)
        predicted = (preds > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct/total:.2%}")
