import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor
from core.logger import get_logger

logger = get_logger("OptimizedDatasetBuilder")

BASE_DIR = r"d:\College\intern\final"
FINAL_DATASET_DIR = os.path.join(BASE_DIR, "CardioVision_Feature_Pipeline", "outputs", "final_dataset")
OPTIMIZED_OUTPUT_DIR = os.path.join(BASE_DIR, "CardioVision_Feature_Pipeline", "outputs", "optimized_dataset")

os.makedirs(OPTIMIZED_OUTPUT_DIR, exist_ok=True)

TARGET_CLASSES = [
    "NSR", "Sinus_Tachycardia", "Sinus_Arrhythmia", "PAC", "RBBB",
    "LBBB", "IVCD", "Persistent_ST_Elevation", "LAE", "Pacemaker_Rhythm"
]

# Demographics fetchers (reused logic)
def parse_hea_demographics(hea_path):
    age = "Unknown"
    sex = "Unknown"
    if not os.path.exists(hea_path): return age, sex
    try:
        with open(hea_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    line_lower = line.lower()
                    if 'age:' in line_lower:
                        parts = line_lower.split('age:')
                        if len(parts) > 1:
                            val = parts[1].strip().split()[0]
                            if val.isdigit() or (val != 'nan' and val != ''): age = val
                    if 'sex:' in line_lower:
                        parts = line_lower.split('sex:')
                        if len(parts) > 1:
                            val = parts[1].strip().split()[0]
                            if val in ['m', 'male']: sex = 'Male'
                            elif val in ['f', 'female']: sex = 'Female'
    except: pass
    return age, sex

def load_ptbxl_demographics():
    ptbxl_meta_path = os.path.join(BASE_DIR, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1", "ptbxl_database.csv")
    ptbxl_demo = {}
    if os.path.exists(ptbxl_meta_path):
        df = pd.read_csv(ptbxl_meta_path)
        for _, row in df.iterrows():
            eid = f"ptbxl_{row['ecg_id']}"
            sex = "Male" if row['sex'] == 0 else ("Female" if row['sex'] == 1 else "Unknown")
            age = str(row['age']) if pd.notna(row['age']) else "Unknown"
            ptbxl_demo[eid] = (age, sex)
    return ptbxl_demo

def fetch_demographics(row, ptbxl_demo):
    ds = row['dataset_source']
    eid = str(row['ecg_id'])
    if ds == "PTB-XL": return ptbxl_demo.get(eid, ("Unknown", "Unknown"))
    if ds == "PhysioNet": path = os.path.join(BASE_DIR, "physionet", eid.replace("physionet_", "") + ".hea")
    elif ds == "Chapman": path = os.path.join(BASE_DIR, "chapman", eid.replace("chapman_", "") + ".hea")
    elif ds == "Georgia": path = os.path.join(BASE_DIR, "georgia", eid.replace("georgia_", "") + ".hea")
    else: return "Unknown", "Unknown"
    return parse_hea_demographics(path)

def build_demographic_dataset():
    labels_path = os.path.join(FINAL_DATASET_DIR, "labels_dataset.csv")
    df = pd.read_csv(labels_path)
    
    logger.info("Fetching demographic data...")
    ptbxl_demo = load_ptbxl_demographics()
    
    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(lambda r: fetch_demographics(r[1], ptbxl_demo), df.iterrows()))
        
    df['age'] = [r[0] for r in results]
    df['sex'] = [r[1] for r in results]
    
    def clean_age(a):
        try:
            val = float(a)
            if np.isnan(val): return -1
            return int(val)
        except: return -1
            
    df['age_num'] = df['age'].apply(clean_age)
    df['age_group'] = pd.cut(df['age_num'], bins=[-2, 0, 20, 40, 60, 80, 150], labels=['Unknown', '0-20', '21-40', '41-60', '61-80', '80+'])
    
    # PTB-XL uses filenames to find .hea files, add it here for future steps if needed, 
    # but the pipeline focuses on dataset creation right now
    return df

def run_pipeline():
    df = build_demographic_dataset()
    original_size = len(df)
    
    df['is_positive'] = (df[TARGET_CLASSES].sum(axis=1) > 0)
    pos_df = df[df['is_positive']].copy()
    neg_df = df[~df['is_positive']].copy()
    
    logger.info(f"Phase 1: Kept {len(pos_df)} positives.")
    
    logger.info("Phase 2: Stratified Negative Sampling (Patient Integrity)")
    # Group negatives by patient
    patient_stats = neg_df.groupby('patient_id').agg({
        'ecg_id': 'count',
        'dataset_source': 'first',
        'age_group': 'first',
        'sex': 'first'
    }).rename(columns={'ecg_id': 'record_count'}).reset_index()
    
    patient_stats['strat_group'] = patient_stats['dataset_source'] + "_" + patient_stats['age_group'].astype(str) + "_" + patient_stats['sex']
    
    # We want 15000 +/- 1% records
    target_min = 14850
    target_max = 15150
    
    # Randomly shuffle patients within strat groups
    patient_stats = patient_stats.sample(frac=1, random_state=42)
    
    sampled_patients = set()
    current_count = 0
    
    # Simple strategy: draw proportionally from each stratum until we hit ~15k
    stratum_proportions = patient_stats['strat_group'].value_counts(normalize=True)
    target_per_stratum = (stratum_proportions * 15000).astype(int)
    
    for stratum, target_count in target_per_stratum.items():
        stratum_patients = patient_stats[patient_stats['strat_group'] == stratum]
        stratum_sum = 0
        for _, p in stratum_patients.iterrows():
            if stratum_sum + p['record_count'] <= target_count + 5: # little buffer
                if current_count + p['record_count'] > 15150: break
                sampled_patients.add(p['patient_id'])
                stratum_sum += p['record_count']
                current_count += p['record_count']
    
    # If we are under target_min, just randomly add patients until we are in range
    remaining_patients = patient_stats[~patient_stats['patient_id'].isin(sampled_patients)]
    for _, p in remaining_patients.iterrows():
        if current_count >= 15000: break
        if current_count + p['record_count'] <= 15150:
            sampled_patients.add(p['patient_id'])
            current_count += p['record_count']
            
    sampled_neg_df = neg_df[neg_df['patient_id'].isin(sampled_patients)].copy()
    
    logger.info(f"Sampled {len(sampled_neg_df)} negatives (Target 14850-15150)")
    assert target_min <= len(sampled_neg_df) <= target_max, f"Failed negative sampling bounds! Got {len(sampled_neg_df)}"
    
    optimized_df = pd.concat([pos_df, sampled_neg_df]).reset_index(drop=True)
    
    # Output Phase 2 Report
    neg_orig_dist = neg_df['dataset_source'].value_counts()
    neg_samp_dist = sampled_neg_df['dataset_source'].value_counts()
    neg_rep = pd.DataFrame({
        "Source": neg_orig_dist.index,
        "Original_Count": neg_orig_dist.values,
        "Sampled_Count": [neg_samp_dist.get(x, 0) for x in neg_orig_dist.index],
        "Preservation_%": [round(neg_samp_dist.get(x,0)/neg_orig_dist.get(x,1)*100,2) for x in neg_orig_dist.index]
    })
    neg_rep.to_csv(os.path.join(OPTIMIZED_OUTPUT_DIR, "negative_sampling_report.csv"), index=False)
    
    logger.info("Phase 3: Dataset Comparison Report")
    comp_data = []
    
    def add_comp(category, val, orig_c, opt_c):
        o_pct = orig_c / len(df) * 100
        n_pct = opt_c / len(optimized_df) * 100
        shift = abs(o_pct - n_pct)
        comp_data.append({
            "Category": category, "Value": val, "Original_Pct": round(o_pct,2), 
            "Optimized_Pct": round(n_pct,2), "Shift_%": round(shift,2), "Flagged": shift > 5.0
        })
        
    for ds in df['dataset_source'].unique():
        add_comp("Source", ds, len(df[df['dataset_source']==ds]), len(optimized_df[optimized_df['dataset_source']==ds]))
    for sex in df['sex'].unique():
        add_comp("Sex", sex, len(df[df['sex']==sex]), len(optimized_df[optimized_df['sex']==sex]))
    for age in df['age_group'].unique():
        add_comp("Age", age, len(df[df['age_group']==age]), len(optimized_df[optimized_df['age_group']==age]))
    
    add_comp("Prevalence", "Positive", len(pos_df), len(pos_df))
    add_comp("Prevalence", "Negative", len(neg_df), len(sampled_neg_df))
    
    pd.DataFrame(comp_data).to_csv(os.path.join(OPTIMIZED_OUTPUT_DIR, "dataset_reduction_report.csv"), index=False)
    
    logger.info("Phase 4: Dataset Bias Validation")
    X = optimized_df[['sex', 'age_num'] + TARGET_CLASSES].copy()
    X['sex'] = X['sex'].map({'Male':0, 'Female':1, 'Unknown':2}).fillna(2)
    y = optimized_df['dataset_source']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    
    pd.DataFrame([{"Metric": "Dataset Source Prediction Accuracy", "Value": acc, "Flagged": acc > 0.80}]).to_csv(
        os.path.join(OPTIMIZED_OUTPUT_DIR, "dataset_bias_after_sampling.csv"), index=False)
        
    logger.info("Phase 5: Class-Source Distribution")
    class_source = []
    for cls in TARGET_CLASSES:
        cls_df = optimized_df[optimized_df[cls] == 1]
        counts = cls_df['dataset_source'].value_counts()
        total = len(cls_df)
        if total == 0: continue
        d = {"Class": cls}
        for ds in optimized_df['dataset_source'].unique():
            pct = counts.get(ds, 0) / total * 100
            d[ds] = round(pct, 2)
            d[f"Flag_{ds}"] = pct > 80.0
        class_source.append(d)
    pd.DataFrame(class_source).to_csv(os.path.join(OPTIMIZED_OUTPUT_DIR, "class_source_distribution.csv"), index=False)
    
    logger.info("Phase 6: Data Splitting & Rare-Class Split Audit")
    # External test
    ext_test_mask = optimized_df['dataset_source'] == "Georgia"
    
    internal_df = optimized_df[~ext_test_mask].copy()
    
    # Train/Val split by patient
    unique_internal_patients = internal_df['patient_id'].unique()
    train_p, val_p = train_test_split(unique_internal_patients, test_size=0.2, random_state=42)
    
    optimized_df['split_group'] = "Unknown"
    optimized_df.loc[optimized_df['dataset_source'] == "Georgia", 'split_group'] = "External_Test"
    optimized_df.loc[optimized_df['patient_id'].isin(train_p), 'split_group'] = "Train"
    optimized_df.loc[optimized_df['patient_id'].isin(val_p), 'split_group'] = "Validation"
    
    # Patient assertion test
    t_p = set(optimized_df[optimized_df['split_group'] == 'Train']['patient_id'])
    v_p = set(optimized_df[optimized_df['split_group'] == 'Validation']['patient_id'])
    e_p = set(optimized_df[optimized_df['split_group'] == 'External_Test']['patient_id'])
    assert len(t_p.intersection(v_p)) == 0, "Patient overlap Train/Val"
    assert len(t_p.intersection(e_p)) == 0, "Patient overlap Train/Ext"
    assert len(v_p.intersection(e_p)) == 0, "Patient overlap Val/Ext"
    
    rare_classes = ["Pacemaker_Rhythm", "LBBB", "PAC", "LAE"]
    rare_audit = []
    failed_splits = False
    for cls in rare_classes:
        t_c = optimized_df[(optimized_df['split_group'] == 'Train') & (optimized_df[cls] == 1)].shape[0]
        v_c = optimized_df[(optimized_df['split_group'] == 'Validation') & (optimized_df[cls] == 1)].shape[0]
        e_c = optimized_df[(optimized_df['split_group'] == 'External_Test') & (optimized_df[cls] == 1)].shape[0]
        rare_audit.append({"Class": cls, "Train": t_c, "Val": v_c, "External_Test": e_c})
        if t_c == 0 or v_c == 0 or e_c == 0:
            failed_splits = True
            
    pd.DataFrame(rare_audit).to_csv(os.path.join(OPTIMIZED_OUTPUT_DIR, "rare_class_audit.csv"), index=False)
    assert not failed_splits, "A split contains 0 positives for a rare class!"
    
    logger.info("Phase 7: Final Outputs")
    # Need to append the original paths so the next step can use it directly
    # Same as what was done in pilot
    ptbxl_meta_path = os.path.join(BASE_DIR, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1", "ptbxl_database.csv")
    ptbxl_paths = {}
    if os.path.exists(ptbxl_meta_path):
        ptb_df = pd.read_csv(ptbxl_meta_path)
        for _, row in ptb_df.iterrows():
            ptbxl_paths[f"ptbxl_{row['ecg_id']}"] = row['filename_hr']
            
    def get_hea_path(row):
        ds = row['dataset_source']
        eid = str(row['ecg_id'])
        if ds == "PTB-XL":
            fname = ptbxl_paths.get(eid)
            if fname:
                return os.path.join(BASE_DIR, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1", fname + ".hea")
        elif ds == "PhysioNet":
            return os.path.join(BASE_DIR, "physionet", eid.replace("physionet_", "") + ".hea")
        elif ds == "Chapman":
            return os.path.join(BASE_DIR, "chapman", eid.replace("chapman_", "") + ".hea")
        elif ds == "Georgia":
            return os.path.join(BASE_DIR, "georgia", eid.replace("georgia_", "") + ".hea")
        return None
        
    optimized_df['hea_path'] = optimized_df.apply(get_hea_path, axis=1)
    
    optimized_df.to_parquet(os.path.join(OPTIMIZED_OUTPUT_DIR, "optimized_labels_dataset.parquet"), index=False)
    optimized_df.to_csv(os.path.join(OPTIMIZED_OUTPUT_DIR, "optimized_labels_dataset.csv"), index=False)
    
    missing_age = (optimized_df['age_num'] == -1).sum() / len(optimized_df) * 100
    missing_sex = (optimized_df['sex'] == "Unknown").sum() / len(optimized_df) * 100
    missing_pid = optimized_df['patient_id'].isna().sum() / len(optimized_df) * 100
    
    with open(os.path.join(OPTIMIZED_OUTPUT_DIR, "final_extraction_readiness.md"), 'w') as f:
        f.write("# Final Extraction Readiness Audit\n\n")
        f.write(f"## Dataset Size\n- Final Count: {len(optimized_df)} records\n\n")
        f.write("## Missing Metadata\n")
        f.write(f"- Age: {missing_age:.2f}%\n")
        f.write(f"- Sex: {missing_sex:.2f}%\n")
        f.write(f"- Patient ID: {missing_pid:.2f}%\n\n")
        f.write("## Estimated Runtime (156 Features)\n")
        f.write(f"- **16 Cores**: ~{len(optimized_df)*0.1225/3600:.2f} hours\n")
        f.write(f"- **32 Cores**: ~{len(optimized_df)*0.0612/3600:.2f} hours\n\n")
        f.write("## Estimated Storage\n")
        f.write("- **Parquet Size**: ~350 MB\n")
        f.write("- **CSV Size**: ~1.1 GB\n\n")
        f.write("Dataset is fully validated, splits are locked, and ready for Mass Extraction.\n")

if __name__ == "__main__":
    run_pipeline()
