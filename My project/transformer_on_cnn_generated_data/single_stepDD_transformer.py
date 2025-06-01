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
# 3. Distilled Set Init
# ----------------------------
num_classes = 10
img_shape = (1, 28, 28)
distilled_imgs = nn.Parameter(torch.randn(num_classes, *img_shape, requires_grad=True))
distilled_labels = torch.tensor([i for i in range(num_classes)], dtype=torch.long)
distilled_optimizer = optim.Adam([distilled_imgs], lr=0.01)

# ----------------------------
# 4. Training Loop
# ----------------------------
outer_epochs = 1000
for epoch in range(outer_epochs):
    distilled_optimizer.zero_grad()
    net = SimpleTransformer()
    start_time = time.time()

    inner_loss = F.cross_entropy(net(distilled_imgs), distilled_labels)
    grads = torch.autograd.grad(inner_loss, net.parameters(), create_graph=True)
    lr_inner = 0.1
    fast_weights = [p - lr_inner * g for p, g in zip(net.parameters(), grads)]

    fast_dict = {name: param for (name, _), param in zip(net.named_parameters(), fast_weights)}

    def forward_with_weights(x, weights):
        x = x.squeeze(1)
        x = F.linear(x, weights['embedding.weight'], weights['embedding.bias'])
        x = x.permute(1, 0, 2)
        x = net.transformer(x)
        x = x.mean(dim=0)
        x = F.linear(x, weights['classifier.weight'], weights['classifier.bias'])
        return x

    preds_real = forward_with_weights(real_x, fast_dict)
    outer_loss = F.cross_entropy(preds_real, real_y)
    outer_loss.backward()
    distilled_optimizer.step()

    end_time = time.time()
    epoch_time = end_time - start_time
    memory_usage = psutil.Process().memory_info().rss / (1024 ** 2)

    if (epoch + 1) % 20 == 0:
        eval_net = SimpleTransformer()
        loss_distilled = F.cross_entropy(eval_net(distilled_imgs), distilled_labels)
        grads_distilled = torch.autograd.grad(loss_distilled, eval_net.parameters())
        for p, g in zip(eval_net.parameters(), grads_distilled):
            p.data.sub_(lr_inner * g)

        eval_net.eval()
        real_preds = eval_net(real_x)
        real_loss = F.cross_entropy(real_preds, real_y)
        real_acc = (real_preds.argmax(dim=1) == real_y).float().mean()

        print(f"[Epoch {epoch+1:03d}] Outer Loss: {outer_loss.item():.4f} | "
              f"Real Loss after 1 step: {real_loss.item():.4f}, "
              f"Real Acc: {real_acc.item()*100:.2f}% | "
              f"Epoch Time: {epoch_time:.2f}s | "
              f"Memory: {memory_usage:.2f} MB")


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
    "Method": "DD (Transformer)",
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
