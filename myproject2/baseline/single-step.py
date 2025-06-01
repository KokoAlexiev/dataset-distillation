import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import psutil
import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, Subset


###############################################################################
# 0. Utilities
###############################################################################
def save_classwise_tensors(images, labels, output_dir="./results", prefix="distilled"):
    os.makedirs(output_dir, exist_ok=True)
    for c in range(10):
        class_imgs = images[labels == c]
        path = os.path.join(output_dir, f"{prefix}_class_{c}.pt")
        torch.save(class_imgs, path)
        print(f"✅ Saved {class_imgs.size(0)} images to {path}")

###################################################
# 1. Simple CNN for MNIST
###################################################
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

###################################################
# 2. Load a Subset of MNIST
###################################################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# For demonstration, pick a random subset of 1024 real images:
subset_indices = torch.randperm(len(mnist_train))[:1024]
mnist_subset = Subset(mnist_train, subset_indices)
real_loader = DataLoader(mnist_subset, batch_size=256, shuffle=True)

###################################################
# 3. Create Distilled Set (10 images/class)
###################################################
num_classes = 10
images_per_class = 10
total_distilled = num_classes * images_per_class  # 10 classes * 10 = 100

img_shape = (1, 28, 28)
distilled_imgs = nn.Parameter(
    torch.randn(total_distilled, *img_shape, requires_grad=True)
)

# Create repeated labels: e.g. [0,0,...(10 times), 1,1,...(10 times), ...]
distilled_labels = torch.tensor(
    [c for c in range(num_classes) for _ in range(images_per_class)],
    dtype=torch.long
)

# We'll use a smaller Adam LR to help stability with bigger lr_inner:
distilled_optimizer = optim.Adam([distilled_imgs], lr=0.003)

###################################################
# 4. Single-Step DD Training
###################################################
outer_epochs = 1000         # more outer iterations for better convergence
num_inits = 8               # random initial networks per outer iteration
lr_inner = 0.2              # single-step learning rate on the distilled set
memory_usage = psutil.Process().memory_info().rss / (1024 ** 2)

for epoch in range(outer_epochs):
    distilled_optimizer.zero_grad()
    
    # We'll accumulate the outer loss over multiple random inits
    outer_loss_total = 0.0
    
    for _ in range(num_inits):
        ###################################################
        # (a) Random init
        ###################################################
        net = SimpleCNN()  # new random init each time
        
        ###################################################
        # (b) Single-step (inner) on distilled data
        ###################################################
        loss_distilled = F.cross_entropy(net(distilled_imgs), distilled_labels)
        grads = torch.autograd.grad(loss_distilled, net.parameters(), create_graph=True)
        
        # Build updated fast weights
        fast_weights = []
        for p, g in zip(net.parameters(), grads):
            fast_weights.append(p - lr_inner * g)
        
        # Construct dictionary to mimic net.state_dict()
        weight_keys = [
            'conv1.weight', 'conv1.bias',
            'conv2.weight', 'conv2.bias',
            'fc.weight', 'fc.bias'
        ]
        fast_dict = {k: v for k, v in zip(weight_keys, fast_weights)}
        
        ###################################################
        # (c) Outer loss on real data
        ###################################################
        real_x, real_y = next(iter(real_loader))
        
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
        
        outer_loss_total += outer_loss
    
    # Average across the multiple random inits
    outer_loss_mean = outer_loss_total / num_inits
    outer_loss_mean.backward()
    
    distilled_optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"[Epoch {epoch+1}/{outer_epochs}] Outer Loss: {outer_loss_mean.item():.4f}")

###################################################
# 5. Visualize Distilled Images
###################################################
with torch.no_grad():
    imgs = distilled_imgs.detach().cpu()
    # Scale to [0,1] for nicer display
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())

# Plot them in a grid: 10 classes by 10 images each
fig, axes = plt.subplots(num_classes, images_per_class, figsize=(10, 10))
imgs_np = imgs.numpy()

idx = 0
for c in range(num_classes):
    for j in range(images_per_class):
        ax = axes[c, j]
        ax.imshow(imgs_np[idx][0], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Class {distilled_labels[idx].item()}")
        idx += 1

plt.tight_layout()
plt.show()

utils.save_image(imgs, 'distilled_images.png', nrow=images_per_class, normalize=True)
print("✅ Distilled images saved as 'distilled_images.png'")

###################################################
# 6. Evaluate Distilled Data
###################################################
# We'll do a single-step from random init, then measure test accuracy
test_loader = DataLoader(
    datasets.MNIST(root='./data', train=False, transform=transform),
    batch_size=1024, shuffle=False
)

num_evals = 5
base_seed = 42
accuracies = []

for i in range(num_evals):
    seed = base_seed + i
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    eval_net = SimpleCNN()
    eval_net.train()
    
    # Single-step on distilled set
    loss_dd = F.cross_entropy(eval_net(distilled_imgs), distilled_labels)
    grads_dd = torch.autograd.grad(loss_dd, eval_net.parameters())
    with torch.no_grad():
        for p, g in zip(eval_net.parameters(), grads_dd):
            p.sub_(lr_inner * g)  # same single-step LR used in training
    
    # Evaluate on test set
    eval_net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            preds = eval_net(x_test)
            correct += (preds.argmax(dim=1) == y_test).sum().item()
            total += y_test.size(0)
    
    acc = 100.0 * correct / total
    accuracies.append(acc)
    print(f"[Eval {i+1}/{num_evals}] Seed {seed}, Test Acc: {acc:.2f}%")

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
print(f"\n📌 Final Test Accuracy (Single-Step Distillation) "
      f"over {num_evals} runs: {mean_acc:.2f}% ± {std_acc:.2f}%\n")

###################################################
# 7. Real Subset Baseline (5 epochs)
###################################################
print("🧪 Real Subset Baseline Evaluation")
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

# Evaluate baseline
baseline_net.eval()
correct = total = 0
with torch.no_grad():
    for x_test, y_test in test_loader:
        preds = baseline_net(x_test)
        correct += (preds.argmax(dim=1) == y_test).sum().item()
        total += y_test.size(0)

baseline_acc = 100.0 * correct / total
print(f"📌 Real Subset Baseline Accuracy (5-epoch train): {baseline_acc:.2f}%")

###################################################
# 8. Optional: Save CSV
###################################################
csv_path = "dd_vs_pdd_baseline_results1.csv"
epoch_time = 0.0  # Not precisely measured here.

dd_row = {
    "Method": "DD (Single-Step)",
    "Images_Per_Class": images_per_class,
    "Total_Images": total_distilled,
    "Test_Accuracy_Mean": round(mean_acc, 2),
    "Test_Accuracy_Std": round(std_acc, 2),
    "Training_Time_Per_Epoch": round(epoch_time, 4),
    "Memory_MB": round(memory_usage, 2)
}
baseline_row = {
    "Method": "Real Subset",
    "Images_Per_Class": 100,
    "Total_Images": 1000,
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
print(f"✅ Results appended to {csv_path}")

###################################################
# 9. Save Distilled DD Dataset
###################################################
torch.save({
    'images': distilled_imgs.detach().cpu(),
    'labels': distilled_labels.detach().cpu()
}, "dd_distilled.pt")
print("✅ DD distilled dataset saved as 'dd_distilled.pt'")


save_classwise_tensors(distilled_imgs.detach().cpu(), distilled_labels.cpu(), output_dir="./results", prefix="dd")

