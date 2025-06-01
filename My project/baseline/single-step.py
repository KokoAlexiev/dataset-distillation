import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import psutil
import random
import os
import pandas as pd
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--images_per_class", type=int, default=1)
args = parser.parse_args()

# Define global experiment settings
num_classes = 10
images_per_class = args.images_per_class
real_subset_size = num_classes * images_per_class

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -------------------------
# 1. Simple CNN for MNIST
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----------------------------
# 2. Load MNIST subset
# ----------------------------
torch.manual_seed(0)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
real_subset_size = num_classes * images_per_class
subset_indices = torch.randperm(len(mnist_train))[:real_subset_size]
mnist_subset = Subset(mnist_train, subset_indices)
real_loader = DataLoader(mnist_subset, batch_size=1024, shuffle=True)

real_x, real_y = [], []
class_counts = {i: 0 for i in range(10)}

# Iterate over the entire MNIST training set
for x, y in mnist_train:
    label = y
    if class_counts[label] < images_per_class:
        real_x.append(x.unsqueeze(0))
        real_y.append(torch.tensor([label]))
        class_counts[label] += 1
    if sum(class_counts.values()) == images_per_class * 10:
        break

real_x = torch.cat(real_x)
real_y = torch.cat(real_y)



# ----------------------------
# 3. Create distilled set
# ----------------------------
num_classes = 10
img_shape = (1, 28, 28)
images_per_class = args.images_per_class
distilled_imgs = nn.Parameter(torch.randn(num_classes * images_per_class, *img_shape, requires_grad=True))
distilled_labels = torch.tensor([i for i in range(num_classes) for _ in range(images_per_class)], dtype=torch.long)
distilled_optimizer = optim.Adam([distilled_imgs], lr=0.01)

# ----------------------------
# 4. Single-step DD Training
# ----------------------------
outer_epochs = 1000
for epoch in range(outer_epochs):
    distilled_optimizer.zero_grad()
    net = SimpleCNN()
    start_time = time.time()

    inner_loss = F.cross_entropy(net(distilled_imgs), distilled_labels)
    grads = torch.autograd.grad(inner_loss, net.parameters(), create_graph=True)
    lr_inner = 0.1
    fast_weights = [p - lr_inner * g for p, g in zip(net.parameters(), grads)]

    weight_keys = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc.weight', 'fc.bias']
    fast_dict = {k: v for k, v in zip(weight_keys, fast_weights)}

    def forward_with_params(x, weights):
        x = F.relu(F.conv2d(x, weights['conv1.weight'], weights['conv1.bias'], padding=1))
        x = F.max_pool2d(x, 2)
        x = F.relu(F.conv2d(x, weights['conv2.weight'], weights['conv2.bias'], padding=1))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.linear(x, weights['fc.weight'], weights['fc.bias'])
        return x

    preds_real = forward_with_params(real_x, fast_dict)
    outer_loss = F.cross_entropy(preds_real, real_y)
    outer_loss.backward()
    distilled_optimizer.step()

    end_time = time.time()
    epoch_time = end_time - start_time
    memory_usage = psutil.Process().memory_info().rss / (1024 ** 2)

    if (epoch + 1) % 20 == 0:
        net_after_1_step = SimpleCNN()
        loss_distilled = F.cross_entropy(net_after_1_step(distilled_imgs), distilled_labels)
        grads_distilled = torch.autograd.grad(loss_distilled, net_after_1_step.parameters())
        for p, g in zip(net_after_1_step.parameters(), grads_distilled):
            p.data.sub_(0.1 * g)

        net_after_1_step.eval()
        real_preds = net_after_1_step(real_x)
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
utils.save_image(imgs, 'distilled_images.png', nrow=10, normalize=True)
print("✅ Distilled images saved as 'distilled_images.png'")

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

    eval_net = SimpleCNN()
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
print(f"\n📌 Final Test Accuracy for DD over {num_evals} runs: {mean_acc:.2f}% ± {std_acc:.2f}%")

# ----------------------------
# 7. Real Subset Baseline CNN
# ----------------------------
print("\n🧪 Real Subset Baseline Evaluation")
baseline_net = SimpleCNN()
optimizer = optim.SGD(baseline_net.parameters(), lr=0.1, momentum=0.9)
start_time = time.time()
for epoch in range(5):
    for x_batch, y_batch in real_loader:
        preds = baseline_net(x_batch)
        loss = F.cross_entropy(preds, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
baseline_time = (time.time() - start_time) / 5

baseline_net.eval()
correct = total = 0
with torch.no_grad():
    for x_test, y_test in test_loader:
        preds = baseline_net(x_test)
        correct += (preds.argmax(1) == y_test).sum().item()
        total += y_test.size(0)

baseline_acc = 100 * correct / total
print(f"📌 Real Subset Baseline Accuracy: {baseline_acc:.2f}%")

# ----------------------------
# 8. Save both results to CSV
# ----------------------------
csv_path = "dd_vs_pdd_baseline_results.csv"
dd_row = {
    "Method": "DD (Single-Step)",
    "Architecture": "CNN",
    "Images_Per_Class": images_per_class,
    "Total_Images": num_classes * images_per_class,
    "Test_Accuracy_Mean": round(mean_acc, 2),
    "Test_Accuracy_Std": round(std_acc, 2),
    "Training_Time_Per_Epoch": round(epoch_time, 4),
    "Memory_MB": round(memory_usage, 2)
}
baseline_row = {
    "Method": "Real Subset",
    "Architecture": "CNN",
    "Images_Per_Class": images_per_class,
    "Total_Images": real_subset_size,
    "Test_Accuracy_Mean": round(baseline_acc, 2),
    "Test_Accuracy_Std": 0.00,
    "Training_Time_Per_Epoch": round(baseline_time, 4),
    "Memory_MB": round(memory_usage, 2)
}

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df = pd.concat([df, pd.DataFrame([dd_row, baseline_row])], ignore_index=True)
else:
    df = pd.DataFrame([dd_row, baseline_row])

df.to_csv(csv_path, index=False)
print(f"✅ All results saved to {csv_path}")