import os
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports'))
ERRORS_DIR = os.path.join(REPORTS_DIR, 'error_analysis')
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ERRORS_DIR, exist_ok=True)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculates Expected Calibration Error."""
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        bin_idx = binids == i
        if np.sum(bin_idx) > 0:
            prob_mean = np.mean(y_prob[bin_idx])
            true_mean = np.mean(y_true[bin_idx])
            ece += np.abs(prob_mean - true_mean) * np.sum(bin_idx)
            
    return ece / len(y_true)

def run_calibration_audit(y_true_df, y_prob_dict):
    """Phase 5: Calibration Verification"""
    print("Running Calibration Audit...")
    results = []
    
    for cls in y_true_df.columns:
        if cls in y_prob_dict:
            y_true = y_true_df[cls].values
            y_prob = y_prob_dict[cls]
            
            brier = brier_score_loss(y_true, y_prob)
            ece = expected_calibration_error(y_true, y_prob)
            
            results.append({
                "Target Class": cls,
                "Brier Score": brier,
                "ECE": ece,
                "Status": "REJECT" if ece > 0.15 else "PASS"
            })
            
    report_df = pd.DataFrame(results)
    report_df.to_csv(os.path.join(REPORTS_DIR, 'calibration_report.csv'), index=False)
    print("Saved calibration_report.csv")

def run_error_analysis(X_test, y_true_df, y_pred_dict, y_prob_dict):
    """Phase 5: Error Analysis (FP, FN, Hard Cases)"""
    print("Generating Error Analysis Reports...")
    
    for cls in y_true_df.columns:
        if cls in y_pred_dict:
            y_true = y_true_df[cls].values
            y_pred = y_pred_dict[cls]
            y_prob = y_prob_dict[cls]
            
            # Reconstruct df with predictions
            df = X_test.copy()
            df['true_label'] = y_true
            df['pred_label'] = y_pred
            df['pred_prob'] = y_prob
            
            # False Positives
            fp_df = df[(df['true_label'] == 0) & (df['pred_label'] == 1)]
            fp_df.to_csv(os.path.join(ERRORS_DIR, f'{cls.lower()}_false_positives.csv'), index=False)
            
            # False Negatives
            fn_df = df[(df['true_label'] == 1) & (df['pred_label'] == 0)]
            fn_df.to_csv(os.path.join(ERRORS_DIR, f'{cls.lower()}_false_negatives.csv'), index=False)
            
            # Hard Cases (prob between 0.4 and 0.6)
            hard_df = df[(df['pred_prob'] > 0.4) & (df['pred_prob'] < 0.6)]
            hard_df.to_csv(os.path.join(ERRORS_DIR, f'{cls.lower()}_hard_cases.csv'), index=False)

    print("Error analysis CSVs generated in outputs/reports/error_analysis/")

if __name__ == "__main__":
    print("Evaluation Auditor ready.")
