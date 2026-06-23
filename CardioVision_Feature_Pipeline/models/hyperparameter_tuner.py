import os
import json
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models'))
os.makedirs(MODELS_DIR, exist_ok=True)

# Define which classes get Optuna optimization vs Shared Baseline
OPTUNA_CLASSES = ['PAC', 'LBBB', 'IVCD', 'LAE', 'Pacemaker_Rhythm']
BASELINE_CLASSES = ['NSR', 'Sinus_Tachycardia', 'Sinus_Arrhythmia', 'RBBB', 'Persistent_ST_Elevation']

BASELINE_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0.1
}

def optimize_class(target, X_train, y_train_cls, X_val, y_val_cls, n_trials=30):
    """Run Optuna for a specific target class optimizing for PR-AUC."""
    print(f"Running Optuna for difficult class: {target}")
    
    def objective(trial):
        params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0)
        }
        
        # Calculate scale_pos_weight
        pos_count = y_train_cls.sum()
        neg_count = len(y_train_cls) - pos_count
        params['scale_pos_weight'] = neg_count / max(1, pos_count)
        
        # Train model with early stopping
        model = xgb.XGBClassifier(**params, n_estimators=1000, early_stopping_rounds=50)
        model.fit(
            X_train, y_train_cls, 
            eval_set=[(X_val, y_val_cls)], 
            verbose=False
        )
        
        # Predict probabilities
        preds = model.predict_proba(X_val)[:, 1]
        
        # Evaluate PR-AUC
        pr_auc = average_precision_score(y_val_cls, preds)
        return pr_auc
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best PR-AUC for {target}: {study.best_value:.4f}")
    return study.best_params

def run_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """Phase 3: Hyperparameter Optimization"""
    print("Starting Hyperparameter Tuning Phase...")
    best_hyperparameters = {}
    
    for cls in OPTUNA_CLASSES:
        if cls in y_train.columns:
            best_params = optimize_class(cls, X_train, y_train[cls], X_val, y_val[cls], n_trials=20)
            best_hyperparameters[cls] = best_params
            
    for cls in BASELINE_CLASSES:
        best_hyperparameters[cls] = BASELINE_PARAMS
        
    out_path = os.path.join(MODELS_DIR, 'best_hyperparameters.json')
    with open(out_path, 'w') as f:
        json.dump(best_hyperparameters, f, indent=4)
    print(f"Hyperparameters saved to {out_path}")
    
    return best_hyperparameters

if __name__ == "__main__":
    print("Hyperparameter tuner ready.")
