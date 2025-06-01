import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time, psutil, os
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pdd_path", type=str, default="pdd_distilled.pt")
args = parser.parse_args()


# ----------------------------
# 1. Transformer Definition
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
# 2. Load distilled set from CNN PDD
# ----------------------------
p#dd_path = "my project/baseline/pdd_distilled.pt"  # Path where you saved from pdd.py
checkpoint = torch.load(pdd_path)

distilled_imgs = checkpoint['images']  # Shape: [N, 1, 28, 28]
distilled_labels = checkpoint['labels']  # Shape: [N]

# Count images per class
labels_np = distilled_labels.cpu().numpy()
counts = np.bincount(labels_np, minlength=10)
unique_counts = np.unique(counts)

if len(unique_counts) == 1:
    images_per_class = int(unique_counts[0])
else:
    images_per_class = "uneven"

print(f"✅ Loaded {distilled_imgs.shape[0]} PDD images distilled from CNN")

# ----------------------------
# 3. Prepare test loader
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_loader = DataLoader(datasets.MNIST(root='./data', train=False, transform=transform),
                         batch_size=1024, shuffle=False)

# ----------------------------
# 4. Evaluate Transformer on PDD
# ----------------------------
num_evals = 5
base_seed = 42
lr = 0.1
steps = 3
accuracies = []
epoch_times = []
memory_usages = []

for i in range(num_evals):
    seed = base_seed + i
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SimpleTransformer()
    model.train()
    start_time = time.time()
    for _ in range(steps):
        loss = F.cross_entropy(model(distilled_imgs), distilled_labels)
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False)
        with torch.no_grad():
            for p, g in zip(model.parameters(), grads):
                p.sub_(lr * g)
    epoch_time = (time.time() - start_time) / steps
    memory = psutil.Process().memory_info().rss / (1024 ** 2)
    memory_usages.append(memory)

    # Eval
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            preds = model(x_test)
            correct += (preds.argmax(1) == y_test).sum().item()
            total += y_test.size(0)
    acc = 100 * correct / total
    accuracies.append(acc)
    print(f"[Eval {i+1}/{num_evals}] Accuracy: {acc:.2f}%")

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
print(f"\n📌 Transformer on CNN PDD dataset: {mean_acc:.2f}% ± {std_acc:.2f}%")

# ----------------------------
# 5. Save results to CSV
# ----------------------------
csv_path = "dd_vs_pdd_baseline_results.csv"
row = {
    "Method": "PDD",
    "Architecture": "Transformer",
    "Images_Per_Class": images_per_class,
    "Total_Images": distilled_imgs.shape[0],
    "Test_Accuracy_Mean": round(mean_acc, 2),
    "Test_Accuracy_Std": round(std_acc, 2),
    "Training_Time_Per_Epoch": round(epoch_time, 4),
    "Memory_MB": round(np.mean(memory_usages), 2)
}

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame([row])
    df = pd.concat([df, new_df], ignore_index=True)

    # Drop duplicates based on Method + Architecture + Total_Images
    df['ID'] = df['Method'].astype(str) + "_" + df.get('Architecture', "").astype(str) + "_" + df['Total_Images'].astype(str)
    df = df.drop_duplicates(subset='ID', keep='last').drop(columns='ID')
else:
    df = pd.DataFrame([row])
df.to_csv(csv_path, index=False)
print(f"✅ Logged cross-architecture PDD result to {csv_path}")
