import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torchvision.utils as vutils

###############################################################################
# 1. Utilities
###############################################################################
def save_classwise_tensors(images, labels, output_dir="./results", prefix="distilled"):
    os.makedirs(output_dir, exist_ok=True)
    for c in range(10):
        class_imgs = images[labels == c]
        path = os.path.join(output_dir, f"{prefix}_class_{c}.pt")
        torch.save(class_imgs, path)
        print(f"✅ Saved {class_imgs.size(0)} images to {path}")


def set_seed(seed=0):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def forward_with_fast_weights(net, x, fast_weights):
    """
    Similar to your approach (1): manually forward pass using 'fast_weights'.
    This is for unrolled training to differentiate w.r.t. the distilled images.
    """
    idx = 0
    # conv1
    x = F.conv2d(x, fast_weights[idx], fast_weights[idx+1], padding=1)
    x = F.relu(x)
    idx += 2

    # conv2
    x = F.max_pool2d(x, 2)
    x = F.conv2d(x, fast_weights[idx], fast_weights[idx+1], padding=1)
    x = F.relu(x)
    idx += 2

    # fc
    x = F.max_pool2d(x, 2)
    x = x.view(x.size(0), -1)
    x = F.linear(x, fast_weights[idx], fast_weights[idx+1])
    idx += 2

    return x

###############################################################################
# 2. Simple CNN (MNIST)
###############################################################################
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)

###############################################################################
# 3. Load Real Data (MNIST)
###############################################################################
def get_mnist_loaders(subset_size=10000, batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data  = datasets.MNIST('./data', train=False,  download=True, transform=transform)

    # If you want a real subset to speed up, else skip Subset
    indices = torch.randperm(len(train_data))[:subset_size]
    subset  = Subset(train_data, indices)
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    test_loader  = DataLoader(test_data, batch_size=1024, shuffle=False)
    return train_loader, test_loader

###############################################################################
# 4. Distill in multiple progressive stages
###############################################################################
def pdd_distill(
    train_loader, 
    num_classes=10, 
    num_stages=5, 
    ipc_per_stage=10, 
    inner_steps=5, 
    lr_inner=0.1,
    outer_steps=200,
    lr_distilled=0.01
):
    """
    Implements the PDD approach (2) in a style close to your code (1):
      - For stage i, we generate new images S_i from scratch (or real init).
      - We keep old subsets S_1..S_{i-1} fixed, union them, 
        and do unrolled inner loops on that union + newly created S_i.
      - We do an outer loop that tries to match real-data performance 
        (outer objective).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # We'll store the distilled images & labels across stages
    distilled_params_stages = []
    distilled_labels_stages = []

    # Helper to create initial images for a stage
    def init_stage_images(init_from='random'):
        # shape = (num_classes * ipc_per_stage, 1, 28, 28)
        shape = (num_classes * ipc_per_stage, 1, 28, 28)
        if init_from == 'random':
            params = nn.Parameter(torch.randn(shape, device=device), requires_grad=True)
        else:
            # Optionally: pick real data from loader to init
            real_x, real_y = next(iter(train_loader))
            real_x = real_x[:(num_classes*ipc_per_stage)].to(device)
            params = nn.Parameter(real_x.clone(), requires_grad=True)
        return params

    # For labeling, we do [class0 repeated ipc_per_stage times, class1 repeated..., etc.]
    def make_stage_labels():
        labels = []
        for c in range(num_classes):
            labels.extend([c]*ipc_per_stage)
        return torch.LongTensor(labels).to(device)

    # Start by creating S1
    s1_imgs = init_stage_images(init_from='random')
    s1_labels = make_stage_labels()

    distilled_params_stages.append(s1_imgs)
    distilled_labels_stages.append(s1_labels)

    # Single Adam optimizer that will optimize all stage images
    distilled_optimizer = optim.Adam(distilled_params_stages, lr=lr_distilled)

    for stage_i in range(num_stages):
        if stage_i > 0:
            # Create new stage's images
            stage_imgs = init_stage_images(init_from='random')
            stage_labels = make_stage_labels()
            distilled_params_stages.append(stage_imgs)
            distilled_labels_stages.append(stage_labels)

            # Add to optimizer param groups
            distilled_optimizer.add_param_group({'params': stage_imgs})

        # Now get the union of all images so far
        all_images = torch.cat(distilled_params_stages, dim=0)
        all_labels = torch.cat(distilled_labels_stages, dim=0)

        print(f"\n===== Stage {stage_i+1}/{num_stages} =====")
        print(f"Union set size so far: {all_images.shape[0]} synthetic images")

        # Outer loop for this stage
        for outer_step in range(outer_steps):
            # Get a real batch
            real_x, real_y = next(iter(train_loader))
            real_x, real_y = real_x.to(device), real_y.to(device)

            distilled_optimizer.zero_grad()

            # --------------------------
            # 1) Build a fresh model
            # --------------------------
            model = SimpleCNN().to(device)
            fast_weights = [p for p in model.parameters()]

            # --------------------------
            # 2) Inner unrolled updates on union(S1..Si)
            # --------------------------
            for _ in range(inner_steps):
                preds = forward_with_fast_weights(model, all_images, fast_weights)
                loss_inner = F.cross_entropy(preds, all_labels)
                grads_inner = torch.autograd.grad(loss_inner, fast_weights, create_graph=True)
                fast_weights = [w - lr_inner*g for w, g in zip(fast_weights, grads_inner)]

            # --------------------------
            # 3) Outer objective: match performance on real batch
            # --------------------------
            real_preds = forward_with_fast_weights(model, real_x, fast_weights)
            loss_outer = F.cross_entropy(real_preds, real_y)

            # Backprop on distilled images
            loss_outer.backward()
            distilled_optimizer.step()

            if (outer_step+1) % (outer_steps//5) == 0:
                print(f" Outer step {outer_step+1}/{outer_steps}, loss_outer={loss_outer.item():.4f}")

    return distilled_params_stages, distilled_labels_stages

###############################################################################
# 5. Evaluate PDD dataset
###############################################################################
def evaluate_pdd(
    distilled_params_stages,
    distilled_labels_stages,
    test_loader,
    lr_inner=0.1,
    inner_steps=5
):
    """
    Evaluate by training a new net from scratch on the union of all 
    synthetic subsets (S1..SP) for 'inner_steps'.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SimpleCNN().to(device)

    all_imgs  = torch.cat(distilled_params_stages, dim=0).detach()
    all_lbls  = torch.cat(distilled_labels_stages, dim=0).detach()

    all_imgs  = all_imgs.to(device)
    all_lbls  = all_lbls.to(device)

    # Multi-step training
    for _ in range(inner_steps):
        y = net(all_imgs)
        loss = F.cross_entropy(y, all_lbls)
        grads = torch.autograd.grad(loss, net.parameters())
        with torch.no_grad():
            for p, g in zip(net.parameters(), grads):
                p.sub_(lr_inner*g)

    # Evaluate on test set
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            preds = net(x_test)
            correct += (preds.argmax(dim=1) == y_test).sum().item()
            total   += y_test.size(0)

    acc = 100.0 * correct / total
    return acc

###############################################################################
# 6. Main Running Example
###############################################################################
if __name__ == "__main__":
    set_seed(42)

    train_loader, test_loader = get_mnist_loaders(subset_size=10000, batch_size=256)

    # >>> Hyperparams you can tweak <<<
    NUM_STAGES = 5
    IPC_PER_STAGE = 10
    INNER_STEPS = 5
    LR_INNER = 0.1
    OUTER_STEPS = 300  # bigger => better
    LR_DISTILLED = 0.01

    # 1) Run PDD Distillation
    print("🚀 Starting PDD Distillation ...")
    start_time = time.time()

    all_params, all_labels = pdd_distill(
        train_loader,
        num_classes=10,
        num_stages=NUM_STAGES,
        ipc_per_stage=IPC_PER_STAGE,
        inner_steps=INNER_STEPS,
        lr_inner=LR_INNER,
        outer_steps=OUTER_STEPS,
        lr_distilled=LR_DISTILLED
    )

    duration = time.time() - start_time
    print(f"\n✅ Done. PDD Distillation took {duration/60:.2f} minutes.")

    # 2) Evaluate
    print("🧪 Evaluating Distilled Data ...")
    n_evals = 5
    accuracies = []
    for seed_offset in range(n_evals):
        set_seed(42 + seed_offset)
        acc = evaluate_pdd(all_params, all_labels, test_loader,
                           lr_inner=LR_INNER, inner_steps=INNER_STEPS)
        accuracies.append(acc)
        print(f"  => Run {seed_offset+1}/{n_evals}: Test Accuracy = {acc:.2f}%")

    mean_acc = np.mean(accuracies)
    std_acc  = np.std(accuracies)
    print(f"\n📌 Final (PDD) Test Accuracy over {n_evals} trials: {mean_acc:.2f}% ± {std_acc:.2f}%")

    # 3) (Optional) Visualize or Save Distilled Sets
    #   Save the final union in a .pt file:
    #   Distilled images are the union of each stage’s images in all_params
    #   Distilled labels are the union in all_labels
    
    
    num_classes = 10
    images_per_class = len(all_imgs) // num_classes

    fig, axes = plt.subplots(num_classes, images_per_class, figsize=(images_per_class, num_classes))
    for c in range(num_classes):
        class_idxs = np.where(labels_np == c)[0]
        for i, idx in enumerate(class_idxs[:images_per_class]):
            ax = axes[c, i]
            ax.imshow(imgs_np[idx][0], cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_ylabel(f"Class {c}", fontsize=10)

    plt.tight_layout()
    save_path = "./results/pdd_distilled_grid.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Distilled image grid saved to: {save_path}")

    if not os.path.exists('./results'):
        os.mkdir('./results')

    torch.save({
        'all_params': [p.detach().cpu() for p in all_params],
        'all_labels': [l.detach().cpu() for l in all_labels]
    }, "./results/pdd_distilled.pt")
    print("✅ Distilled PDD images saved to './results/pdd_distilled.pt'")


torch.save({
    'all_params': [p.detach().cpu() for p in all_params],
    'all_labels': [l.detach().cpu() for l in all_labels]
}, "./results/pdd_distilled.pt")
all_imgs = torch.cat(all_params, dim=0).detach().cpu()
all_lbls = torch.cat(all_labels, dim=0).detach().cpu()
save_classwise_tensors(all_imgs, all_lbls, output_dir="./results", prefix="pdd")
