# dataset_pipeline/main_pipeline.py

import os
import shutil
import json
import pandas as pd
from datetime import datetime

from .config import OUTPUT_DIR, TARGET_DISORDERS, SCORING_WEIGHTS
from .metadata_reader import discover_datasets, parse_physionet_header, parse_ptbxl_metadata, load_wfdb_record
from .diagnosis_mapper import map_diagnoses, filter_targets
from .signal_processor import compute_signal_quality, extract_clinical_features
from .clinical_verifier import verify_clinical_rules
from .visualizer import generate_ecg_preview

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def copy_record_files(record_dict, dest_folder):
    """
    Copies the .hea, .dat, .mat, etc. to the destination without renaming.
    """
    ensure_dir(dest_folder)
    base_path = record_dict["base_path"]
    
    for ext in ['.hea', '.dat', '.mat', '.csv']:
        file_path = base_path + ext
        if os.path.exists(file_path):
            shutil.copy2(file_path, dest_folder)

def main(input_dir):
    print(f"Starting ECG Pipeline Scan on: {input_dir}")
    
    # Pre-parse PTB-XL if exists (can be slow, but done once)
    # Check both input_dir and one level deep
    possible_csvs = [
        os.path.join(input_dir, "ptbxl_database.csv"),
        os.path.join(input_dir, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1", "ptbxl_database.csv")
    ]
    ptbxl_csv = next((p for p in possible_csvs if os.path.exists(p)), None)
    
    ptbxl_mapping = {}
    if ptbxl_csv:
        print(f"Parsing PTB-XL metadata CSV at {ptbxl_csv}...")
        ptbxl_mapping = parse_ptbxl_metadata(ptbxl_csv)

    print("Discovering dataset records...")
    all_records = discover_datasets(input_dir)
    print(f"Total ECG headers found: {len(all_records)}")
    
    results_pool = {disorder: [] for disorder in TARGET_DISORDERS}
    rejected_log = []
    
    stats = {
        "Total_Scanned": 0,
        "Total_Matched": 0,
        "Total_Rejected": 0,
        "Rejection_Reasons": {}
    }
    
    # Process each record
    for record in all_records:
        stats["Total_Scanned"] += 1
        
        # Log progress every 1000 records
        if stats["Total_Scanned"] % 1000 == 0:
            print(f"Processed {stats['Total_Scanned']} / {len(all_records)} records...")
            
        dataset_name = record["dataset_name"]
        record_id = os.path.basename(record["base_path"])
        
        # 1. Read Diagnoses
        raw_diagnoses = []
        if record_id in ptbxl_mapping:
            raw_diagnoses = ptbxl_mapping[record_id]
        else:
            # PhysioNet header fallback
            header_meta = parse_physionet_header(record["hea_path"])
            raw_diagnoses = header_meta.get("diagnoses", [])
            
        # 2. Map Diagnoses
        mapped = map_diagnoses(raw_diagnoses)
        target_matches = filter_targets(mapped)
        
        if not target_matches:
            stats["Total_Rejected"] += 1
            reason = "No target disease match"
            stats["Rejection_Reasons"][reason] = stats["Rejection_Reasons"].get(reason, 0) + 1
            rejected_log.append([dataset_name, record_id, str(raw_diagnoses), reason])
            continue
            
        # 3. Load Signal for Verification & Scoring
        signals, fields = load_wfdb_record(record["base_path"])
        if signals is None:
            stats["Total_Rejected"] += 1
            reason = "Corrupted ECG signal"
            stats["Rejection_Reasons"][reason] = stats["Rejection_Reasons"].get(reason, 0) + 1
            rejected_log.append([dataset_name, record_id, str(raw_diagnoses), reason])
            continue
            
        fs = fields.get('fs', 500)
        
        # 4. Process Signal
        quality_score = compute_signal_quality(signals, fs)
        features = extract_clinical_features(signals, fs)
        
        # 5. Clinical Rules & Score Calculation
        valid_matches = []
        for disorder in target_matches:
            is_valid, reason = verify_clinical_rules(disorder, features)
            
            if not is_valid:
                rejected_log.append([dataset_name, record_id, disorder, f"Failed rule: {reason}"])
                continue
                
            # Score formula calculation
            feature_comp = sum(1 for v in features.values() if v is not None) / len(features)
            # Default annotation confidence if missing
            annot_conf = 0.8 
            
            # 40% Clinical Match (passed rules = 1.0)
            # 30% Signal Quality
            # 20% Feature Completeness
            # 10% Dataset Annotation Confidence
            overall_score = (1.0 * SCORING_WEIGHTS["clinical_match"]) + \
                            ((quality_score/100) * SCORING_WEIGHTS["signal_quality"]) + \
                            (feature_comp * SCORING_WEIGHTS["feature_completeness"]) + \
                            (annot_conf * SCORING_WEIGHTS["annotation_confidence"])
            
            overall_score *= 100 # scale 0-100
            
            valid_matches.append({
                "disorder": disorder,
                "score": overall_score,
                "quality": quality_score,
                "features": features,
                "fs": fs,
                "signals": signals
            })
            
        if not valid_matches:
            stats["Total_Rejected"] += 1
            reason = "Failed all clinical verifications"
            stats["Rejection_Reasons"][reason] = stats["Rejection_Reasons"].get(reason, 0) + 1
            continue
            
        stats["Total_Matched"] += 1
        
        # 6. Organize into memory pool
        for match in valid_matches:
            match_data = {
                "dataset": dataset_name,
                "record_id": record_id,
                "record_dict": record,
                "score": match["score"],
                "quality": match["quality"],
                "features": match["features"],
                "fs": match["fs"],
                "signals": match["signals"]
            }
            results_pool[match["disorder"]].append(match_data)
            
            # Copy to 'All' folder
            all_folder = os.path.join(OUTPUT_DIR, dataset_name, match["disorder"], "All")
            copy_record_files(record, all_folder)

    # 7. Select Best 10 and Finalize
    print("\n--- Finalizing Best 10 Selections ---")
    summary_data = []
    
    for disorder, records in results_pool.items():
        if not records:
            continue
            
        # Sort descending by score
        records.sort(key=lambda x: x["score"], reverse=True)
        best_10 = records[:10]
        
        # Datasets grouped
        for idx, rec in enumerate(best_10):
            best_folder = os.path.join(OUTPUT_DIR, rec["dataset"], disorder, "Best_10")
            copy_record_files(rec["record_dict"], best_folder)
            
            # Generate preview image
            img_path = os.path.join(best_folder, f"{rec['record_id']}_preview.png")
            generate_ecg_preview(rec["signals"], rec["fs"], disorder, rec["quality"], img_path, title=disorder)
            
        # Aggregate for summary
        avg_quality = sum(r["quality"] for r in records) / len(records)
        summary_data.append({
            "Disorder": disorder,
            "Total ECGs": len(records),
            "Best Selected": len(best_10),
            "Average Quality": round(avg_quality, 2)
        })
        
        print(f"Mapped {disorder}: {len(records)} records.")

    # 8. Output Reports
    print("\nWriting Pipeline Reports...")
    ensure_dir(OUTPUT_DIR)
    
    # Rejected Records
    if rejected_log:
        pd.DataFrame(rejected_log, columns=["Dataset", "Record ID", "Original Diagnosis", "Reason"]) \
          .to_csv(os.path.join(OUTPUT_DIR, "Rejected_Records.csv"), index=False)
          
    # Summary CSV
    if summary_data:
        pd.DataFrame(summary_data).to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
        
    # Statistics JSON
    with open(os.path.join(OUTPUT_DIR, "statistics.json"), 'w') as f:
        json.dump(stats, f, indent=4)
        
    print("\nPipeline execution complete! Data available in:", OUTPUT_DIR)

if __name__ == "__main__":
    import sys
    # E.g. python -m dataset_pipeline.main_pipeline "D:\College\intern\final"
    input_directory = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    main(input_directory)
