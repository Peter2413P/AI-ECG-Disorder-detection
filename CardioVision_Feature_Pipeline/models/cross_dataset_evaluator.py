import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix
from models.xgboost_trainer import train_and_calibrate_models
from models.pre_training_auditor import run_feature_selection

REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports'))
os.makedirs(REPORTS_DIR, exist_ok=True)

EXPERIMENTS = [
    {"test_source": "PhysioNet", "train_sources": ["PTBXL", "Chapman", "Georgia"]},
    {"test_source": "Georgia", "train_sources": ["PTBXL", "Chapman", "PhysioNet"]},
    {"test_source": "Chapman", "train_sources": ["PTBXL", "Georgia", "PhysioNet"]},
    {"test_source": "PTBXL", "train_sources": ["Chapman", "Georgia", "PhysioNet"]}
]

def evaluate_predictions(y_true, y_prob, threshold):
    """Calculates evaluation metrics prioritizing PR-AUC."""
    y_pred = (y_prob >= threshold).astype(int)
    
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity
    }

def run_rotation_testing(df, best_hyperparameters):
    """Phase 6: External Dataset Rotation Testing"""
    print("Starting External Dataset Rotation Testing...")
    
    results = []
    
    # 1. Apply global feature selection
    final_features = run_feature_selection(df)
    
    target_cols = [c for c in df.columns if c not in final_features and c not in ['ecg_id', 'patient_id', 'dataset_source']]
    
    for exp in EXPERIMENTS:
        test_src = exp['test_source']
        print(f"\n--- Running Experiment: Test = {test_src} ---")
        
        # Split Data
        test_df = df[df['dataset_source'] == test_src]
        train_pool = df[df['dataset_source'].isin(exp['train_sources'])]
        
        # Within train_pool, we need a validation set for calibration and early stopping
        # Using simple random split here for the experiment loop, grouped by patient_id
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(train_pool, groups=train_pool['patient_id']))
        
        train_df = train_pool.iloc[train_idx]
        val_df = train_pool.iloc[val_idx]
        
        X_train, y_train = train_df[final_features], train_df[target_cols]
        X_val, y_val = val_df[final_features], val_df[target_cols]
        X_test, y_test = test_df[final_features], test_df[target_cols]
        
        print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        
        import joblib
        MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models'))
        THRESHOLDS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'thresholds'))
        
        # Load thresholds
        thresh_csv_path = os.path.join(THRESHOLDS_DIR, 'optimal_thresholds.csv')
        thresholds = {}
        if os.path.exists(thresh_csv_path):
            thresh_df = pd.read_csv(thresh_csv_path)
            thresholds = dict(zip(thresh_df['Target Class'], thresh_df['Optimal Threshold']))
        
        for cls in target_cols:
            if cls not in y_test.columns or y_test[cls].sum() == 0:
                continue
                
            model_path = os.path.join(MODELS_DIR, f"{cls.lower()}_model.pkl")
            if not os.path.exists(model_path):
                continue
                
            # Load real model
            model = joblib.load(model_path)
            
            # Predict
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Ensure y_true is integer binary (prevents 'continuous' ValueError)
            y_true_clean = y_test[cls].fillna(0).astype(int)
            
            # Get specific optimal threshold
            opt_thresh = thresholds.get(cls, 0.5)
            
            metrics = evaluate_predictions(y_true_clean, y_prob, threshold=opt_thresh)
            metrics['Test Source'] = test_src
            metrics['Target Class'] = cls
            results.append(metrics)
            
    report_df = pd.DataFrame(results)
    report_df.to_csv(os.path.join(REPORTS_DIR, 'external_generalization_report.csv'), index=False)
    print("Saved external_generalization_report.csv")

if __name__ == "__main__":
    print("Cross Dataset Evaluator ready.")
