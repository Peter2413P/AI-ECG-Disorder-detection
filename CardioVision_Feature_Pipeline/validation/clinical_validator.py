import os
import pandas as pd
import numpy as np

REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports'))
os.makedirs(REPORTS_DIR, exist_ok=True)

def rule_engine_predict(df, cls):
    """
    Mock implementation of Phase 7 Clinical Rule Engine.
    In production, this would apply actual clinical thresholds.
    """
    # Example proxy rules based on the architecture refinement plan
    predictions = np.zeros(len(df))
    
    # Safe getattr/get loop since features might be missing depending on extraction
    if cls == 'RBBB':
        if 'QRS_Dur' in df.columns:
            predictions[df['QRS_Dur'] > 120] = 1
    elif cls == 'Tachycardia':
        if 'HR' in df.columns:
            predictions[df['HR'] > 100] = 1
            
    return predictions

def validate_clinical_rules(X_test, y_pred_dict):
    """Phase 8: Clinical Validation Layer"""
    print("Running Clinical Validation against Rule Engine...")
    results = []
    
    for cls, model_preds in y_pred_dict.items():
        rule_preds = rule_engine_predict(X_test, cls)
        
        agreement = np.mean(model_preds == rule_preds) * 100
        conflict = 100 - agreement
        
        results.append({
            "Target Class": cls,
            "Agreement %": agreement,
            "Conflict %": conflict
        })
        
    report_df = pd.DataFrame(results)
    report_df.to_csv(os.path.join(REPORTS_DIR, 'rule_vs_model_report.csv'), index=False)
    print("Saved rule_vs_model_report.csv")

if __name__ == "__main__":
    print("Clinical Validator ready.")
