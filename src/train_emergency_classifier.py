import torch
from torch.utils.data import DataLoader
from src.data.mimi_dataset import MimiEmergencyDataset
from src.models.emergency_classifier import MimiEmergencyClassifier
from src.data.collate import mimi_collate


device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load split datasets
train_ds = MimiEmergencyDataset("train")
val_ds   = MimiEmergencyDataset("val")


train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=mimi_collate)
val_loader   = DataLoader(val_ds, batch_size=32, collate_fn=mimi_collate)

model = MimiEmergencyClassifier().to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    total_loss = 0

    for codes, labels in train_loader:
        codes, labels = codes.to(device), labels.to(device)

        preds = model(codes)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for codes, labels in val_loader:
            codes, labels = codes.to(device), labels.to(device)
            preds = model(codes)
            predicted = (preds > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2%}")

print("Training complete.")
torch.save(model.state_dict(), "models/emergency_classifier.pt")

