import torch
import os

# Create results directory if it doesn't exist
if not os.path.exists('./results'):
    os.makedirs('./results', exist_ok=True)

# Save the distilled dataset
torch.save({
    'all_params': [p.detach().cpu() for p in all_params],
    'all_labels': [l.detach().cpu() for l in all_labels]
}, "./results/pdd_distilled.pt")
print("✅ Distilled PDD images saved to './results/pdd_distilled.pt'")

# Define the save_classwise_tensors function if it's not already defined
def save_classwise_tensors(images, labels, output_dir="./results", prefix="distilled"):
    os.makedirs(output_dir, exist_ok=True)
    for c in range(10):
        class_imgs = images[labels == c]
        path = os.path.join(output_dir, f"{prefix}_class_{c}.pt")
        torch.save(class_imgs, path)
        print(f"✅ Saved {class_imgs.size(0)} images to {path}")

# Create the combined tensors
all_imgs = torch.cat([p.detach().cpu() for p in all_params], dim=0)
all_lbls = torch.cat([l.detach().cpu() for l in all_labels], dim=0)

# Save individual class tensors
save_classwise_tensors(all_imgs, all_lbls, output_dir="./results", prefix="pdd")