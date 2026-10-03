import os
import sys
import pandas as pd
import numpy as np
import joblib
import shap
import json
import concurrent.futures
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wfdb
import warnings
import time
import shutil

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'CardioVision_Feature_Pipeline', 'outputs', 'final_dataset', 'dataset.parquet')
MODELS_DIR = os.path.join(BASE_DIR, 'CardioVision_Feature_Pipeline', 'outputs', 'models')
THRESHOLDS_PATH = os.path.join(BASE_DIR, 'CardioVision_Feature_Pipeline', 'outputs', 'thresholds', 'optimal_thresholds.csv')
EXTRACTED_DIR = os.path.join(BASE_DIR, 'Extracted')
FINAL_DIR = os.path.join(BASE_DIR, 'Final_Predicted_Dataset')

os.makedirs(FINAL_DIR, exist_ok=True)

# ---------------------------------------------------------
# Phase 1: Load Resources & Build File Index
# ---------------------------------------------------------
print("Phase 1: Verifying and loading models...")
df = pd.read_parquet(DATASET_PATH)
print(f"Loaded {len(df)} records from {DATASET_PATH}")

thresholds_df = pd.read_csv(THRESHOLDS_PATH)
thresholds = dict(zip(thresholds_df['Target Class'], thresholds_df['Optimal Threshold']))

print("Scanning Extracted/ directory to map patient IDs to files...")
file_index = {}
for root, _, files in os.walk(EXTRACTED_DIR):
    for f in files:
        if f.endswith('.hea'):
            base_name = f.replace('.hea', '')
            file_index[base_name] = os.path.join(root, base_name)
            
print(f"Mapped {len(file_index)} raw ECG records.")

# ---------------------------------------------------------
# Phase 2 & 3: Batch Prediction & SHAP
# ---------------------------------------------------------
print("Phase 3: Running Batch Predictions...")

predictions_dict = {cls: np.zeros(len(df)) for cls in thresholds.keys()}
shap_dict = {cls: [] for cls in thresholds.keys()}
features_list = []

for cls, thresh in thresholds.items():
    model_path = os.path.join(MODELS_DIR, f"{cls.lower()}_model.pkl")
    if not os.path.exists(model_path):
        print(f"Skipping {cls}: Model not found.")
        continue
        
    print(f"Running inference for {cls}...")
    model = joblib.load(model_path)
    model_features = list(model.feature_names_in_)
    features_list = model_features
    
    for col in model_features:
        if col not in df.columns:
            df[col] = 0.0
            
    X_input = df[model_features]
    probs = model.predict_proba(X_input)[:, 1]
    predictions_dict[cls] = probs
    
    base_xgb = model.calibrated_classifiers_[0].estimator
    if hasattr(base_xgb, 'estimator'):
        base_xgb = base_xgb.estimator
    explainer = shap.TreeExplainer(base_xgb)
    shap_vals = explainer.shap_values(X_input)
    shap_dict[cls] = shap_vals

# Prefix the probability columns so they don't clash with ground truth labels
prob_df = pd.DataFrame(predictions_dict)
prob_df = prob_df.add_prefix('prob_')
df = pd.concat([df.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)

# ---------------------------------------------------------
# Phase 4: Output Generation Worker
# ---------------------------------------------------------
print("Phase 4: Generating Final Prediction Dataset...")

def process_record(args):
    idx, row_data = args
    ecg_id = row_data['ecg_id']
    patient_id = row_data['patient_id']
    
    raw_path = file_index.get(ecg_id)
    if not raw_path:
        raw_path = file_index.get(patient_id)
        if not raw_path:
            return None

    predicted_classes = []
    probabilities = {}
    for cls, thresh in thresholds.items():
        prob_col = f"prob_{cls}"
        if prob_col in row_data:
            prob = row_data[prob_col]
            probabilities[cls] = float(prob)
            if prob >= thresh:
                predicted_classes.append(cls)

    if not predicted_classes:
        return 'No Prediction'
        
    try:
        record = wfdb.rdrecord(raw_path, sampto=5000)
        signal = record.p_signal
    except Exception as e:
        return f'WFDB Error: {e}'

    for cls in predicted_classes:
        target_dir = os.path.join(FINAL_DIR, cls, patient_id)
        os.makedirs(target_dir, exist_ok=True)
        
        for ext in ['.hea', '.mat', '.dat']:
            src = f"{raw_path}{ext}"
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(target_dir, f"signal{ext}"))
                
        sv = shap_dict[cls][idx]
        shap_df = pd.DataFrame({'Feature': features_list, 'Importance': sv})
        shap_df = shap_df.sort_values(by='Importance', ascending=False)
        top_shap = shap_df.head(10)
        top_shap.to_csv(os.path.join(target_dir, 'shap_values.csv'), index=False)
        
        pred_json = {
            "record_id": patient_id,
            "dataset": row_data.get('dataset_source', 'Unknown'),
            "predicted": predicted_classes,
            "confidence": probabilities,
            "threshold_used": thresholds,
            "feature_count": len(features_list),
            "shap_summary": top_shap.to_dict(orient='records')
        }
        with open(os.path.join(target_dir, 'prediction.json'), 'w') as f:
            json.dump(pred_json, f, indent=4)
            
        with open(os.path.join(target_dir, 'probabilities.json'), 'w') as f:
            json.dump(probabilities, f, indent=4)
            
        fig_path = os.path.join(target_dir, 'ecg_plot.png')
        if not os.path.exists(fig_path):
            fig = plt.figure(figsize=(15, 8))
            num_leads = min(12, signal.shape[1])
            for i in range(num_leads):
                ax = fig.add_subplot(12, 1, i+1)
                ax.plot(signal[:, i], color='black', linewidth=0.5)
                ax.set_ylabel(record.sig_name[i] if i < len(record.sig_name) else f"Lead {i}")
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            fig.savefig(fig_path, dpi=100)
            plt.close(fig)

    return 'Success'

if __name__ == '__main__':
    results = []
    start_time = time.time()
    
    # Process all records as requested by the user
    subset_df = df
    print(f"Spawning worker pool to process {len(subset_df)} records...")
    
    # Use ThreadPoolExecutor to avoid Windows MemoryError from Process spawn overhead
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        args_list = [(i, row) for i, row in subset_df.iterrows()]
        futures = {executor.submit(process_record, arg): arg for arg in args_list}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            results.append(res)
            if i % 50 == 0:
                print(f"Processed {i}/{len(subset_df)} records...")

    elapsed = time.time() - start_time
    print(f"Completed Phase 4 in {elapsed:.2f} seconds.")

    print("Phase 5: Generating Best 10 Folders...")
    for cls in thresholds.keys():
        best_dir = os.path.join(FINAL_DIR, cls, "Best_10")
        os.makedirs(best_dir, exist_ok=True)
        
        prob_col = f"prob_{cls}"
        top_10 = df.sort_values(by=prob_col, ascending=False).head(10)
        
        for _, row in top_10.iterrows():
            patient_id = row['patient_id']
            src_dir = os.path.join(FINAL_DIR, cls, patient_id)
            dst_dir = os.path.join(best_dir, patient_id)
            if os.path.exists(src_dir):
                if not os.path.exists(dst_dir):
                    shutil.copytree(src_dir, dst_dir)

    print("Phase 6: Generating Summary Reports...")
    prob_cols = [f"prob_{c}" for c in thresholds.keys()]
    out_cols = ['patient_id', 'dataset_source'] + list(thresholds.keys()) + prob_cols
    df[out_cols].to_csv(os.path.join(FINAL_DIR, 'all_predictions.csv'), index=False)

    report = f"""# Prediction Statistics Report
Total ECGs Processed (Demo Limit): {len(subset_df)}
Runtime: {elapsed:.2f} seconds
Model Version: MultiLabelXGBoost v1.0
"""
    with open(os.path.join(FINAL_DIR, 'prediction_statistics.txt'), 'w') as f:
        f.write(report)
        
    print("Pipeline Complete! Output saved to Final_Predicted_Dataset/")
