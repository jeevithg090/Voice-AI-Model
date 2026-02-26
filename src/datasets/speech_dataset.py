import os
import torch
import torchaudio
from torch.utils.data import Dataset

class SpeechDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, clip_seconds=2):
        self.files = []
        self.labels = []
        self.sample_rate = sample_rate
        self.clip_len = sample_rate * clip_seconds

        for label, folder in enumerate(["normal", "emergency"]):
            folder_path = os.path.join(root_dir, folder)
            for fname in os.listdir(folder_path):
                if fname.endswith(".wav"):
                    self.files.append(os.path.join(folder_path, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        waveform = waveform.mean(dim=0)

        if waveform.shape[0] > self.clip_len:
            waveform = waveform[:self.clip_len]
        else:
            pad = self.clip_len - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform, torch.tensor(self.labels[idx], dtype=torch.float32)

