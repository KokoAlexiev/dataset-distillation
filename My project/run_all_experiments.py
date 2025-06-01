import subprocess
import os

def run(label, relative_path):
    full_path = os.path.join("My project", relative_path)
    print(f"\n🚀 Running {label}")
    
    if not os.path.exists(full_path):
        print(f"❌ Skipped: {full_path} not found.\n")
        return
    
    result = subprocess.run(["python", full_path], shell=True)
    
    if result.returncode != 0:
        print(f"❌ Error while running {relative_path}\n")
    else:
        print(f"✅ {label} completed.\n")

# Run all experiments
run("DD (CNN, 10 images)", "baseline/single-step.py")
run("DD (Transformer, 10 images)", "transformer_on_cnn_generated_data/single_stepDD_transformer.py")
run("Real Subset (Transformer, 1000 images)", "transformer_on_cnn_generated_data/evaluate_transformer_on_real_subset.py")
run("PDD (CNN, 500 images)", "baseline/pdd.py")
run("Transformer on CNN-PDD (500 images)", "transformer_on_cnn_generated_data/evaluate_transformer_on_cnn_pdd.py")

print("✅ All experiments completed and results saved!")
