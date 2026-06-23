import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models'))
THRESHOLDS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'thresholds'))
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(THRESHOLDS_DIR, exist_ok=True)

def train_and_calibrate_models(X_train, y_train, X_val, y_val, best_hyperparameters, resume=False):
    """Phase 4: Independent GPU Training & Calibration"""
    print("Training Independent GPU Models...")
    
    thresholds = {}
    thresh_csv_path = os.path.join(THRESHOLDS_DIR, 'optimal_thresholds.csv')
    
    if resume and os.path.exists(thresh_csv_path):
        # Load existing thresholds
        try:
            thresh_df = pd.read_csv(thresh_csv_path)
            thresholds = dict(zip(thresh_df['Target Class'], thresh_df['Optimal Threshold']))
            print(f"Loaded {len(thresholds)} existing thresholds from checkpoint.")
        except Exception as e:
            print(f"Could not load thresholds: {e}")
            
    completed_classes = []
    
    for cls in y_train.columns:
        model_path = os.path.join(MODELS_DIR, f"{cls.lower()}_model.pkl")
        
        if resume and os.path.exists(model_path) and cls in thresholds:
            print(f"Model for {cls} already exists. Skipping training (Checkpoint Recovery).")
            completed_classes.append(cls)
            continue
            
        print(f"Training model for {cls}...")
        
        # 1. Setup specific params
        params = best_hyperparameters.get(cls, {})
        params['tree_method'] = 'hist'
        params['device'] = 'cuda'
        params['objective'] = 'binary:logistic'
        
        # Calculate individual scale_pos_weight
        pos_count = y_train[cls].sum()
        neg_count = len(y_train[cls]) - pos_count
        scale_weight = neg_count / max(1, pos_count)
        params['scale_pos_weight'] = scale_weight
        
        # 2. Train Base XGBoost
        # Note: Scikit-learn API required for CalibratedClassifierCV wrapping
        base_model = xgb.XGBClassifier(**params)
        
        # 3. Apply Calibration (Isotonic)
        # Using method='isotonic' and cv='prefit' requires pre-training on train, 
        # but standard CV will split the train set automatically. 
        # Better approach: Train on train, calibrate on validation to avoid overfitting.
        base_model.fit(X_train, y_train[cls])
        
        from sklearn.frozen import FrozenEstimator
        calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(base_model), method='isotonic')
        calibrated_model.fit(X_val, y_val[cls])
        
        # 4. Find Optimal Threshold on Validation Set (max F1)
        val_preds = calibrated_model.predict_proba(X_val)[:, 1]
        precision, recall, _thresholds = precision_recall_curve(y_val[cls], val_preds)
        
        # f1 = 2 * (p*r) / (p+r)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = f1_scores.argmax()
        
        # Guard against edge cases where precision/recall array length differs from thresholds
        optimal_thresh = _thresholds[best_idx] if best_idx < len(_thresholds) else 0.5
        
        thresholds[cls] = optimal_thresh
        
        # 5. Save Model and Incremental Thresholds
        joblib.dump(calibrated_model, model_path)
        print(f"Saved {cls} model to {model_path} (Opt Threshold: {optimal_thresh:.3f})")
        
        # Save thresholds incrementally so they are not lost on crash
        thresh_df = pd.DataFrame(list(thresholds.items()), columns=['Target Class', 'Optimal Threshold'])
        thresh_df.to_csv(thresh_csv_path, index=False)
        
        completed_classes.append(cls)
        
        # Generate Resume Report
        report_path = os.path.join(MODELS_DIR, "training_resume_report.txt")
        with open(report_path, "w") as f:
            f.write("=== Training Resume Report ===\n")
            f.write(f"Total Classes: {len(y_train.columns)}\n")
            f.write(f"Completed: {len(completed_classes)}\n")
            f.write(f"Remaining: {len(y_train.columns) - len(completed_classes)}\n")
            
    print("Saved all optimal thresholds and models.")
    return thresholds

if __name__ == "__main__":
    print("XGBoost Trainer ready.")
