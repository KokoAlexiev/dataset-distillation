import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import psutil, time, os
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ----------------------------
# Transformer Model
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=28):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(1)

    def forward(self, x):
        return x + self.pe[:x.size(0)].to(x.device)

class SimpleTransformer(nn.Module):
    def __init__(self, num_classes=10, emb_size=64, num_heads=2):
        super().__init__()
        self.embedding = nn.Linear(28, emb_size)
        self.pos_enc = PositionalEncoding(emb_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.classifier(x)

# ----------------------------
# Load MNIST subset (1k)
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
subset_indices = torch.randperm(len(train_data))[:1000]
train_loader = DataLoader(Subset(train_data, subset_indices), batch_size=256, shuffle=True)

test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform),
                         batch_size=1024, shuffle=False)

# ----------------------------
# Train & Evaluate Transformer
# ----------------------------
model = SimpleTransformer()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
start_time = time.time()

for epoch in range(5):
    for x_batch, y_batch in train_loader:
        preds = model(x_batch)
        loss = F.cross_entropy(preds, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
epoch_time = (time.time() - start_time) / 5
memory_usage = psutil.Process().memory_info().rss / (1024**2)

model.eval()
correct = total = 0
with torch.no_grad():
    for x_test, y_test in test_loader:
        preds = model(x_test)
        correct += (preds.argmax(1) == y_test).sum().item()
        total += y_test.size(0)
acc = 100 * correct / total
print(f"✅ Transformer Real Subset Accuracy: {acc:.2f}%")

# ----------------------------
# Save to CSV
# ----------------------------
csv_path = "dd_vs_pdd_baseline_results.csv"
row = {
    "Method": "Real Subset",
    "Distilled By": "CNN",
    "Trained On": "Transformer",
    "Images_Per_Class": "100",
    "Total_Images": 1000,
    "Test_Accuracy_Mean": round(acc, 2),
    "Test_Accuracy_Std": 0.0,
    "Training_Time_Per_Epoch": round(epoch_time, 4),
    "Memory_MB": round(memory_usage, 2)
}

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
else:
    df = pd.DataFrame([row])
df.to_csv(csv_path, index=False)
print(f"✅ Logged Transformer real baseline to {csv_path}")