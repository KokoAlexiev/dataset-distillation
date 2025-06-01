import random
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, Subset

# --------------------------------------------------
# 0. Helper Functions for Multi-Step Implementation
# --------------------------------------------------
def forward_with_fast_weights(net, x, fast_weights):
    """
    A manual forward pass that uses 'fast_weights' 
    instead of the net.parameters(). 
    This is needed when we do unrolled inner updates.
    """
    # Because our net is small, let's do it manually:
    # We'll rely on the parameter ordering to match
    idx = 0
    # conv1.weight, conv1.bias
    x = F.conv2d(x, fast_weights[idx], fast_weights[idx + 1], padding=1)
    x = F.relu(x); idx += 2

    # conv2
    x = F.max_pool2d(x, 2)
    x = F.conv2d(x, fast_weights[idx], fast_weights[idx + 1], padding=1)
    x = F.relu(x); idx += 2

    # fc
    x = F.max_pool2d(x, 2)
    x = x.view(x.size(0), -1)
    x = F.linear(x, fast_weights[idx], fast_weights[idx + 1])
    idx += 2

    return x

def single_step_net(net, images, labels, lr):
    """
    Similar to your existing 'one_step_net', 
    trains 'net' for one step on 'images' -> 'labels'.
    """
    net.train()
    loss = F.cross_entropy(net(images), labels)
    grads = torch.autograd.grad(loss, net.parameters())
    with torch.no_grad():
        for p, g in zip(net.parameters(), grads):
            p.sub_(lr * g)



# ---------------------------------------------
# 1. Simple CNN (Same as Baseline for Fairness)
# ---------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # (B, 16, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)               # (B, 16, 14, 14)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # (B, 32, 14, 14)->(B, 32, 7, 7)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -------------------------------------------------
# 2. Load (Subset of) MNIST for Real Data
# -------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# ⚠️ Subset to speed things up (adjust as needed)
subset_indices = torch.randperm(len(mnist_train))[:10000]
mnist_subset = Subset(mnist_train, subset_indices)
real_loader = DataLoader(mnist_subset, batch_size=256, shuffle=True)

# For final test evaluation
test_loader = DataLoader(
    datasets.MNIST(root='./data', train=False, download=True, transform=transform),
    batch_size=1024, shuffle=False
)

# -----------------------------------
# 3. Define PDD Hyperparameters
# -----------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
#num_stages = 5                  # number of progressive stages
#images_per_class_per_stage = 10 # more images => better final accuracy
outer_steps_per_stage = 100     # outer optimization steps per stage
inner_steps = 5                 # multi-step inner unroll (key for PDD!)
lr_inner = 0.05                 # inner learning rate
pdd_lr = 0.01                   # Adam learning rate for distilled images
num_evals = 5                   # how many times to eval with different seeds
base_seed = 42


# parametarize images per class and num stages
parser = argparse.ArgumentParser()
parser.add_argument("--total_images", type=int, default=500)
args = parser.parse_args()

images_per_class_total = args.total_images // num_classes
num_stages = 5
images_per_class_per_stage = images_per_class_total // num_stages




# -----------------------------------------
# 4. Initialize Distilled Images & Labels
# -----------------------------------------
distilled_params = []       # each stage’s Parameter for the images
distilled_labels_full = []  # matching labels for each stage

def infinite_real_loader(loader):
    while True:
        for batch in loader:
            yield batch

real_iter = infinite_real_loader(real_loader)

def init_distilled_stage_from_real(num_imgs):
    # sample `num_imgs` real images from mnist_subset
    # transform them into a single tensor of shape (num_imgs, 1, 28, 28)
    real_iter = iter(real_loader)
    real_batch, _ = next(real_iter)

    # just take as many as needed from this batch
    real_batch = real_batch[:num_imgs].to(device)

    param = nn.Parameter(real_batch.clone(), requires_grad=True)
    return param

def init_distilled_stage():
    shape = (num_classes * images_per_class_per_stage, 1, 28, 28)
    # -> replace with the real-data-based approach
    return init_distilled_stage_from_real(num_classes * images_per_class_per_stage)

# Initialize the 1st stage
distilled_params.append(init_distilled_stage())

init_labels = []
for c in range(num_classes):
    init_labels.extend([c] * images_per_class_per_stage)
distilled_labels_full.append(torch.LongTensor(init_labels).to(device))

# Single Adam optimizer for all parameters
pdd_optimizer = optim.Adam(distilled_params, lr=pdd_lr)

# -----------------------------------------
# 5. Progressive Distillation
# -----------------------------------------
epoch_times = []
memory_usages = []

print("🚀 Starting Progressive Distillation...")

for stage_idx in range(num_stages):
    # If we’re in a later stage, add a new chunk of images & labels
    if stage_idx > 0:
        distilled_params.append(init_distilled_stage())
        pdd_optimizer.add_param_group({'params': distilled_params[-1]})
        labels_this_stage = []
        for c in range(num_classes):
            labels_this_stage.extend([c] * images_per_class_per_stage)
        distilled_labels_full.append(torch.LongTensor(labels_this_stage).to(device))

    # Combine all images/labels so far
    all_distilled_images = torch.cat(distilled_params, dim=0)
    all_distilled_labels = torch.cat(distilled_labels_full, dim=0)
    total_images = len(all_distilled_images)

    print(f"\n[Stage {stage_idx+1}/{num_stages}] Distilled set size: {total_images} images")

    # ⚠️ Fix an "outer batch" once per stage to reduce noise
    # Just sample one batch from real_loader
    outer_batch_x, outer_batch_y = next(iter(real_loader))
    outer_batch_x = outer_batch_x.to(device)
    outer_batch_y = outer_batch_y.to(device)

    # Outer loop
    for outer_step in range(outer_steps_per_stage):
        start_time = time.time()
        pdd_optimizer.zero_grad()

        # 🧠 1) Create fresh net
        net = SimpleCNN().to(device)

        # 🔄 2) Multi-step (inner) update on the distilled set
        #    unrolled gradient steps to simulate "train on distilled data"
        fast_weights = [p.clone() for p in net.parameters()]  # store a copy of the net’s params

        for inner_s in range(inner_steps):
            # forward pass
            y_d = forward_with_fast_weights(net, all_distilled_images, fast_weights)
            loss_inner = F.cross_entropy(y_d, all_distilled_labels)

            # compute grads w.r.t. fast_weights
            grads_inner = torch.autograd.grad(loss_inner, fast_weights, create_graph=True)

            # gradient descent on fast_weights
            fast_weights = [w - lr_inner * g for w, g in zip(fast_weights, grads_inner)]

        # 3) Outer objective: after the net trains on distilled data,
        #    we want it to perform well on real data. So we do:
        preds_real = forward_with_fast_weights(net, outer_batch_x, fast_weights)
        loss_outer = F.cross_entropy(preds_real, outer_batch_y)

        # 4) Backprop into the distilled parameters
        loss_outer.backward()
        pdd_optimizer.step()

        end_time = time.time()

        # Optional logging every 10 steps
        if (outer_step + 1) % 10 == 0:
            # Evaluate quickly with a brand new net & 1-step train on the distilled set
            net_eval = SimpleCNN().to(device)
            single_step_net(net_eval, all_distilled_images, all_distilled_labels, lr_inner)
            
            net_eval.eval()
            with torch.no_grad():
                real_preds_eval = net_eval(outer_batch_x)
            acc = (real_preds_eval.argmax(dim=1) == outer_batch_y).float().mean().item() * 100

            memory_usage = psutil.Process().memory_info().rss / (1024**2)
            epoch_time = end_time - start_time

            print(f"  Stage {stage_idx+1} | Step {outer_step+1}/{outer_steps_per_stage} "
                  f"| Outer Loss: {loss_outer.item():.4f} "
                  f"| Real-acc(1-step): {acc:.2f}% "
                  f"| Time: {epoch_time:.2f}s "
                  f"| Mem: {memory_usage:.2f} MB")

    # Record time/memory from the last step in each stage
    epoch_times.append(epoch_time)
    memory_usages.append(memory_usage)

print("✅ Progressive Distillation Finished!")


# -------------------------
# 6. Visualize Distilled
# -------------------------
all_distilled_images = torch.cat(distilled_params, dim=0).detach().cpu()
all_distilled_images_normed = (all_distilled_images - all_distilled_images.min()) / (
    all_distilled_images.max() - all_distilled_images.min() + 1e-8
)

rows = num_classes
cols = images_per_class_per_stage * num_stages  # show all images per class across stages
fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))

# Ensure axes is 2D
if rows == 1:
    axes = np.expand_dims(axes, axis=0)
if cols == 1:
    axes = np.expand_dims(axes, axis=1)

idx = 0
for c in range(num_classes):
    for s in range(num_stages * images_per_class_per_stage):
        ax = axes[c][s]
        img = all_distilled_images_normed[idx][0].numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        if s == 0:
            ax.set_title(f"Class {c}")
        idx += 1

plt.tight_layout()
plt.show()

# Save final set of images 
utils.save_image(all_distilled_images_normed, 'pdd_distilled_images.png', nrow=10, normalize=True)
print("✅ PDD distilled images saved as 'pdd_distilled_images.png'")

# -------------------------------------------------------------
# 7. Final Test Set Evaluation Over Multiple Seeds (multi-step)
# -------------------------------------------------------------
def multi_step_net_eval(net, images, labels, lr, steps=3):
    """Trains 'net' for 'steps' inner updates on the entire 'images' dataset."""
    net.train()
    fast_params = [p for p in net.parameters()]  # references
    for _ in range(steps):
        y_d = net(images)
        loss = F.cross_entropy(y_d, labels)
        grads = torch.autograd.grad(loss, fast_params, retain_graph=False)
        with torch.no_grad():
            for p, g in zip(fast_params, grads):
                p.sub_(lr * g)

accuracies = []
all_distilled_labels = torch.cat(distilled_labels_full, dim=0)
all_distilled_images = torch.cat(distilled_params, dim=0)

for i in range(num_evals):
    seed = base_seed + i
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    net_eval = SimpleCNN().to(device)

    # Perform multi-step training on final distilled images
    multi_step_net_eval(net_eval, all_distilled_images, all_distilled_labels, lr=lr_inner, steps=inner_steps)

    # Evaluate on test set
    net_eval.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            preds = net_eval(x_test)
            correct += (preds.argmax(dim=1) == y_test).sum().item()
            total += y_test.size(0)

    acc = 100.0 * correct / total
    accuracies.append(acc)
    print(f"[Eval {i+1}/{num_evals}] Test Accuracy (Seed {seed}): {acc:.2f}%")

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
print(f"\n📌 Final PDD Test Accuracy over {num_evals} runs: {mean_acc:.2f}% ± {std_acc:.2f}%")

# -----------------------------------------------------------
# 8. Save PDD Results to CSV (same file as your baseline)
# -----------------------------------------------------------
csv_path = "dd_vs_pdd_baseline_results.csv"
new_row = {
    "Method": "PDD",
    "Architecture": "CNN",
    "Images_Per_Class": images_per_class_per_stage * num_stages,
    "Total_Images": images_per_class_per_stage * num_stages * num_classes,
    "Test_Accuracy_Mean": round(mean_acc, 2),
    "Test_Accuracy_Std": round(std_acc, 2),
    "Training_Time_Per_Epoch": round(np.mean(epoch_times), 4),  # average across stages
    "Memory_MB": round(np.mean(memory_usages), 2)               # average memory usage
}

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.drop_duplicates(inplace=True
else:
    df = pd.DataFrame([new_row])

df.to_csv(csv_path, index=False)
print(f"✅ PDD results appended to {csv_path}")


# Save for the transformer
torch.save({
    'images': all_distilled_images.detach().cpu(),
    'labels': all_distilled_labels.detach().cpu()
}, "pdd_distilled.pt")
