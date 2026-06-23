import os
import subprocess
import xgboost as xgb

def run_diagnostics():
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports'))
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, 'gpu_diagnostics_report.txt')

    lines = ["=== GPU Diagnostics Report ===\n"]

    # 1. Check NVIDIA-SMI
    try:
        smi_output = subprocess.check_output(['nvidia-smi'], text=True)
        lines.append("--- NVIDIA-SMI Output ---")
        lines.append(smi_output)
    except Exception as e:
        lines.append("--- NVIDIA-SMI Output ---")
        lines.append(f"Failed to run nvidia-smi: {str(e)}")

    # 2. Check PyTorch CUDA (Optional but helpful if torch is installed)
    try:
        import torch
        lines.append("\n--- PyTorch CUDA Status ---")
        lines.append(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"Device Count: {torch.cuda.device_count()}")
            lines.append(f"Device Name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        lines.append("\n--- PyTorch CUDA Status ---")
        lines.append("PyTorch not installed.")
        
    # 3. Check XGBoost CUDA Support
    lines.append("\n--- XGBoost CUDA Status ---")
    try:
        # Create a dummy dataset
        import numpy as np
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Try to initialize a GPU booster
        model = xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=1)
        model.fit(X, y)
        lines.append("XGBoost GPU support: ENABLED and VERIFIED.")
    except Exception as e:
        lines.append("XGBoost GPU support: FAILED.")
        lines.append(f"Error: {str(e)}")

    # Write Report
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
        
    print(f"Diagnostics complete. Report saved to {report_path}")

if __name__ == "__main__":
    run_diagnostics()
