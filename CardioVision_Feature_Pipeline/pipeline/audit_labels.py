import pandas as pd
import numpy as np
import os
from collections import Counter
from labeling.unified_mapper import TARGET_CLASSES
from core.logger import get_logger

logger = get_logger("AuditLabels")

BASE_DIR = r"d:\College\intern\final\CardioVision_Feature_Pipeline\outputs\final_dataset"

def build_audit():
    dataset_path = os.path.join(BASE_DIR, "labels_dataset.csv")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        return
        
    df = pd.read_csv(dataset_path)
    total_ecgs = len(df)
    
    # Ensure targets are present
    targets_present = [c for c in TARGET_CLASSES if c in df.columns]
    df_labels = df[targets_present]
    
    # 1. Class Distribution & Feasibility
    audit_data = []
    class_counts = df_labels.sum()
    for col in targets_present:
        count = class_counts[col]
        pct = (count / total_ecgs) * 100
        
        if count > 1000:
            feasibility = "Safe (>1000)"
        elif count >= 300:
            feasibility = "Moderate (300-1000)"
        elif count >= 100:
            feasibility = "Rare (100-300)"
        else:
            feasibility = "Critical (<100)"
            
        audit_data.append({
            "Target_Class": col,
            "Total_Samples": int(count),
            "Positive_Pct": round(pct, 2),
            "Negative_Pct": round(100 - pct, 2),
            "Feasibility": feasibility
        })
        
    audit_df = pd.DataFrame(audit_data)
    audit_df.to_csv(os.path.join(BASE_DIR, "dataset_audit_report.csv"), index=False)
    
    # 2. Co-occurrence Matrix
    cooc = df_labels.T.dot(df_labels)
    cooc = cooc.astype(float)
    np.fill_diagonal(cooc.values, np.nan) # Set diagonal to NaN or just keep it as sum
    cooc.to_csv(os.path.join(BASE_DIR, "label_cooccurrence.csv"))
    
    # 3. Multi-Label Analysis
    label_sums = df_labels.sum(axis=1)
    ml_counts = {
        "0_labels": int((label_sums == 0).sum()),
        "1_label": int((label_sums == 1).sum()),
        "2_labels": int((label_sums == 2).sum()),
        "3_plus_labels": int((label_sums >= 3).sum())
    }
    
    # Most common combinations
    # Extract row-wise combos
    def get_active_labels(row):
        return tuple(sorted([col for col in targets_present if row[col] == 1]))
        
    combos = df.apply(get_active_labels, axis=1)
    top_combos = Counter(combos).most_common(10)
    
    # 4. Dataset Source Analysis
    source_counts = df['dataset_source'].value_counts().to_dict()
    
    # 5. Duplicate Analysis
    duplicate_ecgs = int(df['ecg_id'].duplicated().sum())
    duplicate_patients = int(df['patient_id'].duplicated().sum())
    
    # Write Final Summary Markdown
    with open(os.path.join(BASE_DIR, "final_summary_report.md"), 'w') as f:
        f.write("# Unified 12-Class ECG Label Dataset Audit\n\n")
        
        f.write("## 1. Class Feasibility Check\n")
        f.write("| Target Class | Total Samples | Positive Pct | Negative Pct | Feasibility |\n")
        f.write("|---|---|---|---|---|\n")
        for _, row in audit_df.iterrows():
            f.write(f"| {row['Target_Class']} | {row['Total_Samples']} | {row['Positive_Pct']} | {row['Negative_Pct']} | {row['Feasibility']} |\n")
        f.write("\n")
        
        f.write("## 2. Multi-Label Analysis\n")
        f.write(f"- 0 Labels (Normal/Unmapped): {ml_counts['0_labels']}\n")
        f.write(f"- 1 Label: {ml_counts['1_label']}\n")
        f.write(f"- 2 Labels: {ml_counts['2_labels']}\n")
        f.write(f"- 3+ Labels: {ml_counts['3_plus_labels']}\n\n")
        
        f.write("### Most Common Combinations\n")
        for combo, count in top_combos:
            combo_str = " + ".join(combo) if combo else "None"
            f.write(f"- {combo_str}: {count}\n")
            
        f.write("\n## 3. Dataset Source Contributions\n")
        for source, count in source_counts.items():
            f.write(f"- {source}: {count} records\n")
            
        f.write("\n## 4. Duplicate Analysis\n")
        f.write(f"- Duplicate ECG IDs: {duplicate_ecgs}\n")
        f.write(f"- Duplicate Patient IDs: {duplicate_patients}\n\n")
        
        f.write("## 5. Final Recommendation\n")
        f.write("Based on the audit report:\n")
        safe_classes = audit_df[audit_df['Feasibility'].str.contains("Safe")]['Target_Class'].tolist()
        critical = audit_df[audit_df['Feasibility'].str.contains("Critical")]['Target_Class'].tolist()
        
        f.write(f"- Total ECGs Processed: {total_ecgs}\n")
        f.write(f"- Safe Classes for Modeling: {', '.join(safe_classes)}\n")
        if critical:
            f.write(f"- Critical Classes (Need Augmentation/Merge): {', '.join(critical)}\n")
            
    logger.info("Audit complete. Generated dataset_audit_report.csv, label_cooccurrence.csv, and final_summary_report.md")

if __name__ == "__main__":
    build_audit()
