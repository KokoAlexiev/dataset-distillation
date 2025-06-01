# single_stepDD_transformer.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time, psutil, random, os
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# ----------------------------
# 1. Simple Transformer for MNIST
# ----------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, num_classes=10, emb_size=64, num_heads=2):
        super().__init__()
        self.embedding = nn.Linear(28, emb_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1)                 # (B, 28, 28)
        x = self.embedding(x)            # (B, 28, emb)
        x = x.permute(1, 0, 2)           # (seq_len, B, emb)
        x = self.transformer(x)          # (seq_len, B, emb)
        x = x.mean(dim=0)                # (B, emb)
        return self.classifier(x)        # (B, num_classes)

# ----------------------------
# 2. Load Dataset
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
subset_indices = torch.randperm(len(mnist_train))[:1024]
real_loader = DataLoader(Subset(mnist_train, subset_indices), batch_size=1024, shuffle=True)
real_x, real_y = next(iter(real_loader))

# ----------------------------
# 3. Load distilled data from CNN (DD-CNN)
# ----------------------------
checkpoint = torch.load("../baseline/dd_distilled.pt")  # adjust if needed
distilled_imgs = checkpoint['images']
distilled_labels = checkpoint['labels']
print(f"✅ Loaded CNN-distilled dataset with {distilled_imgs.shape[0]} images")

# ----------------------------
# 5. Visualize distilled images
# ----------------------------
imgs = distilled_imgs.detach().cpu()
imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i, ax in enumerate(axes):
    ax.imshow(imgs[i][0], cmap='gray')
    ax.axis('off')
    ax.set_title(f"Label: {distilled_labels[i].item()}")
plt.tight_layout()
plt.show()
utils.save_image(imgs, '../distilled_images_transformer.png', nrow=10, normalize=True)
print("✅ Distilled images saved as 'distilled_images_transformer.png'")

# ----------------------------
# 6. Evaluate DD over 5 seeds
# ----------------------------
num_evals = 5
base_seed = 42
test_loader = DataLoader(datasets.MNIST(root='./data', train=False, transform=transform),
                         batch_size=1024, shuffle=False)

accuracies = []
for i in range(num_evals):
    seed = base_seed + i
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    eval_net = SimpleTransformer()
    eval_net.train()
    loss = F.cross_entropy(eval_net(distilled_imgs), distilled_labels)
    grads = torch.autograd.grad(loss, eval_net.parameters())
    with torch.no_grad():
        for p, g in zip(eval_net.parameters(), grads):
            p.sub_(0.1 * g)

    eval_net.eval()
    correct = total = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            preds = eval_net(x_test)
            correct += (preds.argmax(dim=1) == y_test).sum().item()
            total += y_test.size(0)

    acc = 100 * correct / total
    accuracies.append(acc)
    print(f"[Eval {i+1}/{num_evals}] Test Accuracy (Seed {seed}): {acc:.2f}%")

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
print(f"\n📌 Final Transformer DD Accuracy over {num_evals} runs: {mean_acc:.2f}% ± {std_acc:.2f}%")

# ----------------------------
# 7. Append to shared results CSV
# ----------------------------
csv_path = "../dd_vs_pdd_baseline_results.csv"
transformer_row = {
    "Method": "DD-CNN → Transformer",
    "Images_Per_Class": 1,
    "Total_Images": 10,
    "Test_Accuracy_Mean": round(mean_acc, 2),
    "Test_Accuracy_Std": round(std_acc, 2),
    "Training_Time_Per_Epoch": round(epoch_time, 4),
    "Memory_MB": round(memory_usage, 2)
}

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df = pd.concat([df, pd.DataFrame([transformer_row])], ignore_index=True)
else:
    df = pd.DataFrame([transformer_row])
df.to_csv(csv_path, index=False)
print(f"✅ Transformer DD results saved to {csv_path}")
