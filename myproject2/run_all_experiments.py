import subprocess
import shutil
import os

# === Setup ===
base_path = os.path.dirname(os.path.abspath(__file__))

# Paths to all scripts
scripts = [
    "baseline/single-step.py",                          # 1. DD-CNN + DD Distilled Save
    "baseline/pdd.py",                                  # 2. PDD-CNN + PDD Distilled Save
    "transformer_on_cnn_generated_data/single_stepDD_transformer.py",  # 3. DD-Transformer
    "transformer_on_cnn_generated_data/evaluate_transformer_on_cnn_pdd.py",  # 4. PDD-Transformer
    "transformer_on_cnn_generated_data/evaluate_transformer_on_real_subset.py",  # 5. Real-Transformer
    "baseline/evaluate_transformer_on_real_subset.py"    # 6. Real Subset - CNN (Optional)
]

print("🚀 Starting full experiment pipeline...\n")

# Run each script in order
for i, rel_path in enumerate(scripts, start=1):
    script_path = os.path.join(base_path, rel_path)
    print(f"▶️ Running [{i}/{len(scripts)}]: {rel_path}")
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {rel_path}:\n{e}")
        break

# Optional: move result CSV to safe location if needed
csv_path = os.path.join(base_path, "dd_vs_pdd_baseline_results.csv")
if os.path.exists(csv_path):
    print("\n📄 Final results saved in:")
    print(csv_path)
else:
    print("\n⚠️ Results CSV not found. Check scripts for save path.")
