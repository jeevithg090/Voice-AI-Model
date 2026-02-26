import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.datasets.speech_dataset import SpeechDataset
from src.models.speech_emergency_model import SpeechEmergencyEmbeddingModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SpeechDataset("data/raw")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = SpeechEmergencyEmbeddingModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(10):
    model.train()
    total_loss = 0

    for waveforms, labels in tqdm(loader):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        embeddings, logits = model(waveforms)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "models/emergency_embedding_model.pt")

