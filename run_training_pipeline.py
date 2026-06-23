import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add CardioVision_Feature_Pipeline to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'CardioVision_Feature_Pipeline')))

from models.gpu_diagnostics import run_diagnostics
from models.pre_training_auditor import run_feasibility_audit, run_feature_selection, run_drift_analysis, TARGET_CLASSES
from models.hyperparameter_tuner import run_hyperparameter_tuning
from models.xgboost_trainer import train_and_calibrate_models
from models.evaluation_auditor import run_calibration_audit, run_error_analysis
from models.cross_dataset_evaluator import run_rotation_testing
from xai.shap_generator import generate_shap_explanations
from validation.clinical_validator import validate_clinical_rules
from core.config import FINAL_DATASET_DIR

from sklearn.model_selection import StratifiedGroupKFold

def main():
    parser = argparse.ArgumentParser(description="CardioVision Training Pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume from previously completed models")
    args = parser.parse_args()
    
    print("=== Initiating Final GPU XGBoost Training & XAI Pipeline ===")
    
    # Phase 1: Diagnostics
    run_diagnostics()
    
    dataset_path = os.path.join(FINAL_DATASET_DIR, "dataset.parquet")
    print(f"\nLoading dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please run feature extraction first.")
        return
        
    df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(df)} records.")
    
    # Phase 2: Audits and Feature Selection
    run_feasibility_audit(df)
    final_features = run_feature_selection(df)
    
    # Select features + targets + patient_id
    X = df[final_features]
    y = df[TARGET_CLASSES]
    groups = df['patient_id']
    
    print("\nPhase 2: Leakage-Free Data Split (StratifiedGroupKFold)...")
    # For a simple train/val/test split using StratifiedGroupKFold, we'll do an 80/20 split, 
    # then split the 20 into 10/10 for val/test.
    # We will use the most frequent class for stratification if multilabel isn't supported directly by SGKF.
    # To simplify for the pipeline execution, we'll stratify by 'is_positive' if it exists, or just the first class.
    
    sgkf1 = StratifiedGroupKFold(n_splits=5)
    stratify_col = y.idxmax(axis=1) # Pseudo-multilabel stratification
    
    train_idx, temp_idx = next(sgkf1.split(X, stratify_col, groups=groups))
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]
    groups_temp = groups.iloc[temp_idx]
    stratify_col_temp = stratify_col.iloc[temp_idx]
    
    sgkf2 = StratifiedGroupKFold(n_splits=2)
    val_idx, test_idx = next(sgkf2.split(X_temp, stratify_col_temp, groups=groups_temp))
    
    X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
    X_test, y_test = X_temp.iloc[test_idx], y_temp.iloc[test_idx]
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    
    # (Optional) Phase 3: Hyperparameter Tuning - Skipping for speed unless specifically hooked up
    best_hyperparameters = {}
    
    # Phase 4: Independent GPU Training & Calibration
    thresholds = train_and_calibrate_models(X_train, y_train, X_val, y_val, best_hyperparameters, resume=args.resume)
    
    print("\nPhase 5: Evaluation Audit")
    import joblib
    MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'CardioVision_Feature_Pipeline', 'outputs', 'models'))
    y_pred_dict = {}
    y_prob_dict = {}
    
    for cls in TARGET_CLASSES:
        model_path = os.path.join(MODELS_DIR, f"{cls.lower()}_model.pkl")
        if os.path.exists(model_path):
            calibrated_model = joblib.load(model_path)
            probs = calibrated_model.predict_proba(X_test)[:, 1]
            y_prob_dict[cls] = probs
            thresh = thresholds.get(cls, 0.5)
            y_pred_dict[cls] = (probs >= thresh).astype(int)
            
    run_calibration_audit(y_test, y_prob_dict)
    run_error_analysis(X_test, y_test, y_pred_dict, y_prob_dict)
    
    print("\nPhase 6: External Dataset Rotation")
    run_rotation_testing(df, best_hyperparameters)
    
    print("\nPhase 7: SHAP Generation")
    generate_shap_explanations(X_test, TARGET_CLASSES)
    
    print("\nPhase 8: Clinical Rules Validation")
    validate_clinical_rules(X_test, y_pred_dict)
    
    print("\n=== Pipeline Complete! All Final Reports Saved ===")

if __name__ == "__main__":
    main()
