import os
import shutil
import csv
import concurrent.futures
from .config import OUTPUT_DIR

def copy_record_files(record_dict, dest_folder):
    """
    Copies the .hea, .dat, .mat, etc. to the destination without renaming.
    If duplicates exist, appends the dataset name.
    """
    os.makedirs(dest_folder, exist_ok=True)
    base_path = record_dict["base_path"]
    dataset = record_dict["dataset_name"]
    
    copied_files = []
    
    for ext in ['.hea', '.dat', '.mat', '.csv']:
        file_path = base_path + ext
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            dest_path = os.path.join(dest_folder, filename)
            
            # Handle duplicates across datasets
            if os.path.exists(dest_path):
                filename = f"{dataset}_{filename}"
                dest_path = os.path.join(dest_folder, filename)
                
            shutil.copy2(file_path, dest_path)
            copied_files.append(dest_path)
            
    return copied_files

def write_metadata_row(record_dict, disorder, features):
    """Writes metadata for the extracted record."""
    dest_folder = os.path.join(OUTPUT_DIR, disorder)
    os.makedirs(dest_folder, exist_ok=True)
    meta_path = os.path.join(dest_folder, "metadata.csv")
    
    file_exists = os.path.exists(meta_path)
    
    with open(meta_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Dataset", "Record ID", "Diagnosis", "Original File Location", 
                "Heart_Rate", "QRS_Duration"
            ])
            
        writer.writerow([
            record_dict["dataset_name"],
            record_dict["record_id"],
            disorder,
            record_dict["base_path"],
            round(features["heart_rate"], 2),
            round(features["qrs_duration"], 2)
        ])

def commit_batch_to_disk(batch_records, batch_features, valid_masks):
    """
    Consumer I/O task: takes the batched results and multithreads the file copying
    to prevent bottlenecking the GPU loop.
    """
    # Create flat list of tasks
    tasks = []
    
    for i, record in enumerate(batch_records):
        for disorder, mask in valid_masks.items():
            if mask[i]: # If record i passed the rule for this disorder
                tasks.append((record, disorder, {k: v[i] for k, v in batch_features.items()}))
                
    if not tasks:
        return 0
        
    def _task_worker(task):
        rec, dis, feats = task
        dest_folder = os.path.join(OUTPUT_DIR, dis)
        copy_record_files(rec, dest_folder)
        write_metadata_row(rec, dis, feats)
        return True
        
    # Execute file I/O aggressively across CPU threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(_task_worker, tasks))
        
    return len(tasks)
