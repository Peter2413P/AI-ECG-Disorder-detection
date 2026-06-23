import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SHAP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'shap'))
CLASS_SHAP_DIR = os.path.join(SHAP_DIR, 'class_shap_reports')
PATIENT_SHAP_DIR = os.path.join(SHAP_DIR, 'patient_explanations')
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models'))

os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(CLASS_SHAP_DIR, exist_ok=True)
os.makedirs(PATIENT_SHAP_DIR, exist_ok=True)

def generate_shap_explanations(X_test, target_classes):
    """Phase 7: SHAP Explainability"""
    print("Generating SHAP Explanations...")
    
    # Due to CalibratedClassifierCV wrapping XGBoost, SHAP needs the inner XGBoost estimator
    # For CalibratedClassifierCV, estimators are stored in calibrated_classifiers_
    
    for cls in target_classes:
        model_path = os.path.join(MODELS_DIR, f"{cls.lower()}_model.pkl")
        if not os.path.exists(model_path):
            continue
            
        print(f"Processing SHAP for {cls}...")
        calibrated_model = joblib.load(model_path)
        
        # Extract base XGBoost model from the first calibrated fold
        # (Assuming prefit was used, there is only one fitted base estimator)
        base_xgb = calibrated_model.calibrated_classifiers_[0].estimator
        
        # Scikit-learn >= 1.6 wraps estimators in FrozenEstimator when cv="prefit"
        if hasattr(base_xgb, 'estimator'):
            base_xgb = base_xgb.estimator
        
        # Initialize Explainer
        explainer = shap.TreeExplainer(base_xgb)
        
        # Calculate SHAP values for a subset to save time (e.g., 500 records)
        X_sample = X_test.sample(min(500, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        # 1. Per-class SHAP Summary
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f"SHAP Summary - {cls}")
        plt.tight_layout()
        plt.savefig(os.path.join(CLASS_SHAP_DIR, f"{cls.lower()}_summary.png"))
        plt.close()
        
        # 2. Top 20 Features
        # Calculate mean absolute SHAP values per feature
        feature_importance = pd.DataFrame({
            'Feature': X_sample.columns,
            'Importance': np.abs(shap_values).mean(0)
        }).sort_values(by='Importance', ascending=False)
        
        top_20 = feature_importance.head(20)
        top_20.to_csv(os.path.join(CLASS_SHAP_DIR, f"{cls.lower()}_top20_features.csv"), index=False)
        
    print("SHAP generation complete.")

if __name__ == "__main__":
    import numpy as np
    print("SHAP Generator ready.")
