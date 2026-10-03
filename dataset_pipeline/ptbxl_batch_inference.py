import os
import sys
import shutil
import json
import time
import concurrent.futures
from datetime import datetime
import numpy as np
import wfdb

# Adjust paths to access the backend module
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import the exact components from the backend
from backend.app import run_prediction, PredictionRequest, startup_event
from dataset_pipeline.config import TARGET_DISORDERS

# Configure Directories
PTBXL_DIR = os.path.join(BASE_DIR, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1", "records500")
OUTPUT_DIR = os.path.join(BASE_DIR, "Predicted_PTBXL_Dataset")

# Create base output directory and disorder subdirectories
os.makedirs(OUTPUT_DIR, exist_ok=True)
for disorder in TARGET_DISORDERS:
    os.makedirs(os.path.join(OUTPUT_DIR, disorder), exist_ok=True)

# Scan for all PTB-XL .hea files
print(f"Scanning for ECG records in {PTBXL_DIR}...")
all_files = []
for root, _, files in os.walk(PTBXL_DIR):
    for file in files:
        if file.endswith('.hea'):
            all_files.append(os.path.join(root, file))

print(f"Found {len(all_files)} total .hea records.")

# Initialize the global prediction models that the app uses
print("Initializing Backend Models...")
startup_event()

# Bypassing the heavy local LLM generation for batch processing
import backend.app
backend.app.rag_client = None

print("Models loaded successfully.")

# Tracking statistics
stats = {
    'total_processed': 0,
    'success': 0,
    'failed': 0,
    'multi_label': 0,
    'skipped_nan': 0,
    'disorder_counts': {d: 0 for d in TARGET_DISORDERS}
}

PROCESSED_FILE = os.path.join(OUTPUT_DIR, "processed_records.txt")
processed_set = set()
if os.path.exists(PROCESSED_FILE):
    with open(PROCESSED_FILE, "r") as f:
        for line in f:
            processed_set.add(line.strip())

def process_record(hea_path):
    record_id = os.path.basename(hea_path).replace('.hea', '')
    if record_id in processed_set:
        return record_id, [], True, "ALREADY_PROCESSED"
        
    base_path = hea_path.replace('.hea', '')
    
    # 1. Pre-scan for NaN corruption to prevent pipeline hanging
    try:
        record = wfdb.rdrecord(base_path)
        if np.isnan(record.p_signal).any():
            return record_id, [], False, "CORRUPT_NAN_SIGNAL"
    except Exception as e:
        return record_id, [], False, f"WFDB_READ_ERROR: {e}"

    req = PredictionRequest(filepath=hea_path)
    
    try:
        # Run exactly as a user upload would
        result = run_prediction(req)
        
        predictions = result.get('predictions', {})
        thresholds = result.get('thresholds', {})
        
        predicted_disorders = []
        for disorder, prob in predictions.items():
            if prob >= thresholds.get(disorder, 0.5):
                predicted_disorders.append(disorder)
                
        if not predicted_disorders:
            # Did not cross thresholds for any disorder
            return record_id, [], True, None
            
        # Copy to the corresponding folders
        base_src = hea_path.replace('.hea', '')
        for disorder in predicted_disorders:
            if disorder not in TARGET_DISORDERS:
                continue
                
            target_folder = os.path.join(OUTPUT_DIR, disorder, record_id)
            os.makedirs(target_folder, exist_ok=True)
            
            # Copy original files (.hea, .dat, .mat)
            for ext in ['.hea', '.dat', '.mat']:
                src_file = f"{base_src}{ext}"
                if os.path.exists(src_file):
                    shutil.copy2(src_file, os.path.join(target_folder, f"{record_id}{ext}"))
                    
            # Dump prediction JSON
            meta = {
                "record_id": record_id,
                "dataset": "PTB-XL",
                "predicted": predicted_disorders,
                "confidence": predictions,
                "threshold_used": thresholds,
                "timestamp": datetime.now().isoformat()
            }
            with open(os.path.join(target_folder, "prediction.json"), "w") as f:
                json.dump(meta, f, indent=4)
                
        return record_id, predicted_disorders, True, None
        
    except Exception as e:
        return record_id, [], False, str(e)

if __name__ == '__main__':
    start_time = time.time()
    
    # Process the entire PTB-XL dataset (21,837 files)
    test_files = all_files
    
    print(f"\nStarting batch prediction on {len(test_files)} records...")
    
    # Use ThreadPoolExecutor with 8 workers to prevent neurokit2 C-extension lockups
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_record, f): f for f in test_files}
        
        with open(PROCESSED_FILE, "a") as processed_log:
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                record_id, predicted, success, error = future.result()
                
                if success and error == "ALREADY_PROCESSED":
                    # Silently skip logging already processed records to keep console clean
                    continue
                    
                stats['total_processed'] += 1
                
                # Mark as processed so we can resume later
                processed_log.write(f"{record_id}\n")
                processed_log.flush()
                
                if success:
                    stats['success'] += 1
                    if len(predicted) > 1:
                        stats['multi_label'] += 1
                    for d in predicted:
                        if d in stats['disorder_counts']:
                            stats['disorder_counts'][d] += 1
                            
                    pred_str = ", ".join(predicted) if predicted else "No Prediction"
                    print(f"[{i}/{len(test_files)}] Record: {record_id} | Prediction: {pred_str} | Copied Successfully")
                else:
                    stats['failed'] += 1
                    if error == "CORRUPT_NAN_SIGNAL":
                        stats['skipped_nan'] += 1
                        print(f"[{i}/{len(test_files)}] Record: {record_id} | SKIPPED | Reason: Corrupt NaN Signal")
                    else:
                        print(f"[{i}/{len(test_files)}] Record: {record_id} | FAILED | Error: {error}")
                
    elapsed = time.time() - start_time
    
    print("\n" + "="*40)
    print("FINAL SUMMARY STATISTICS")
    print("="*40)
    print(f"Total ECG records processed : {stats['total_processed']}")
    print(f"Successfully predicted      : {stats['success']}")
    print(f"Failed records              : {stats['failed']}")
    print(f"Multi-label predictions     : {stats['multi_label']}")
    print(f"Total execution time        : {elapsed:.2f} seconds")
    print("\nRecords per Disorder:")
    for d, count in stats['disorder_counts'].items():
        print(f"  - {d}: {count}")
