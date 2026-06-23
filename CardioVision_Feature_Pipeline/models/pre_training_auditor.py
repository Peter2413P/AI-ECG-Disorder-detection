import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports'))
os.makedirs(REPORTS_DIR, exist_ok=True)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.config import TARGET_CLASSES

def run_feasibility_audit(df):
    """Phase 2: Class Feasibility Validation"""
    print("Running Class Feasibility Validation...")
    results = []
    for cls in TARGET_CLASSES:
        if cls in df.columns:
            pos_count = df[cls].sum()
            total = len(df)
            neg_count = total - pos_count
            pos_ratio = pos_count / total
            
            if pos_count > 1000:
                action = "Safe"
            elif pos_count >= 300:
                action = "Moderate"
            else:
                action = "Rare"
                
            if cls == "Pacemaker_Rhythm":
                action = action + " (Flagged for special handling)"
                
            results.append({
                "Target Class": cls,
                "Positive count": pos_count,
                "Negative count": neg_count,
                "Positive ratio": pos_ratio,
                "Recommended action": action
            })
            
    report_df = pd.DataFrame(results)
    report_df.to_csv(os.path.join(REPORTS_DIR, 'class_feasibility_report.csv'), index=False)
    print("Saved class_feasibility_report.csv")

def run_feature_selection(df):
    """Phase 2: Feature Selection (Missingness, Zero Variance, Correlation)"""
    print("Running Feature Selection...")
    # Exclude metadata and labels
    metadata_cols = ['ecg_id', 'patient_id', 'dataset_source']
    feature_cols = [c for c in df.columns if c not in TARGET_CLASSES and c not in metadata_cols]
    
    X = df[feature_cols]
    initial_count = len(feature_cols)
    
    # 1. Missingness filter (>15%)
    missing_pct = X.isnull().mean()
    missing_drops = missing_pct[missing_pct > 0.15].index.tolist()
    X = X.drop(columns=missing_drops)
    
    # 2. Zero variance filter
    variances = X.var()
    zero_var_drops = variances[variances == 0].index.tolist()
    X = X.drop(columns=zero_var_drops)
    
    # 3. Correlation filter (>0.95)
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_drops = [column for column in upper.columns if any(upper[column] > 0.95)]
    X = X.drop(columns=high_corr_drops)
    
    final_count = len(X.columns)
    print(f"Features - Initial: {initial_count}, Removed: {initial_count - final_count}, Final: {final_count}")
    
    report_lines = [
        "=== Feature Selection Report ===",
        f"Initial Features: {initial_count}",
        f"Removed (>15% missing): {len(missing_drops)}",
        f"Removed (Zero variance): {len(zero_var_drops)}",
        f"Removed (Correlation > 0.95): {len(high_corr_drops)}",
        f"Final Retained Features: {final_count}"
    ]
    with open(os.path.join(REPORTS_DIR, 'feature_selection_report.csv'), 'w') as f:
        f.write("\n".join(report_lines))
        
    return X.columns.tolist()

def run_drift_analysis(df, final_features):
    """Phase 2: Dataset Drift Analysis"""
    print("Running Dataset Drift Analysis...")
    sources = df['dataset_source'].unique()
    
    results = []
    
    # Pairwise KS tests across all sources for each final feature
    for feature in final_features:
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                source_a = sources[i]
                source_b = sources[j]
                
                data_a = df[df['dataset_source'] == source_a][feature].dropna()
                data_b = df[df['dataset_source'] == source_b][feature].dropna()
                
                if len(data_a) == 0 or len(data_b) == 0:
                    continue
                    
                stat, p_value = ks_2samp(data_a, data_b)
                
                if stat > 0.3:
                    results.append({
                        "Feature": feature,
                        "Source A": source_a,
                        "Source B": source_b,
                        "KS Statistic": stat,
                        "p-value": p_value,
                        "Status": "FLAGGED (Severe Drift)"
                    })
                    
    drift_df = pd.DataFrame(results)
    if not drift_df.empty:
        drift_df.to_csv(os.path.join(REPORTS_DIR, 'feature_drift_report.csv'), index=False)
        print(f"Saved feature_drift_report.csv ({len(drift_df)} flags found)")
    else:
        print("No severe drift found (KS > 0.3).")

if __name__ == "__main__":
    # Note: In a real run, load the actual dataset.parquet
    # df = pd.read_parquet("CardioVision_Feature_Pipeline/outputs/final_dataset/dataset.parquet")
    print("Pre-training auditor ready. Call these functions from the master script.")
