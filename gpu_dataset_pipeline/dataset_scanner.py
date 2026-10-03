import os
import glob
import pandas as pd
import ast
import multiprocessing
import concurrent.futures
from .config import TARGET_DISORDERS, DIAGNOSIS_MAP, EXCLUSIONS

def parse_ptbxl_metadata(csv_path):
    """Parses PTB-XL CSV to map filenames to SCP codes."""
    try:
        df = pd.read_csv(csv_path, index_col='ecg_id')
        mapping = {}
        for _, row in df.iterrows():
            filename = os.path.basename(str(row['filename_hr']))
            try:
                scp_dict = ast.literal_eval(row['scp_codes'])
                mapping[filename] = list(scp_dict.keys())
            except:
                mapping[filename] = []
        return mapping
    except Exception:
        return {}

def parse_physionet_header(header_path):
    """Parses standard WFDB .hea for Dx fields."""
    diagnoses = []
    try:
        with open(header_path, 'r') as f:
            for line in f:
                if 'Dx:' in line:
                    dx_str = line.split('Dx:')[1].strip()
                    diagnoses = [dx.strip() for dx in dx_str.split(',')]
    except Exception:
        pass
    return diagnoses

def map_diagnoses(raw_dx_list):
    """Maps raw strings/codes to our target 12 disorders, applying exclusions."""
    mapped = set()
    for raw in raw_dx_list:
        clean_raw = str(raw).lower().strip()
        
        # Check Exclusions first
        if any(ex in clean_raw for ex in EXCLUSIONS):
            continue
            
        # Check specific mappings
        for key, target in DIAGNOSIS_MAP.items():
            if key == clean_raw or key in clean_raw:
                mapped.add(target)
                break
    return list(mapped)

def scan_single_file(hea_path, ptbxl_mapping):
    """Reads a single file's metadata and maps it. Designed for multiprocessing."""
    base_path = os.path.splitext(hea_path)[0]
    has_mat = os.path.exists(base_path + '.mat')
    has_dat = os.path.exists(base_path + '.dat')
    
    if not (has_mat or has_dat):
        return None
        
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(hea_path))) # approximate
    record_id = os.path.basename(base_path)
    
    # Get raw diagnoses
    if record_id in ptbxl_mapping:
        raw_dx = ptbxl_mapping[record_id]
        dataset_name = "PTB_XL"
    else:
        raw_dx = parse_physionet_header(hea_path)
        
    mapped_targets = map_diagnoses(raw_dx)
    
    # Only return if it matches at least one target
    if not mapped_targets:
        return None
        
    return {
        "base_path": base_path,
        "hea_path": hea_path,
        "data_path": base_path + ('.mat' if has_mat else '.dat'),
        "dataset_name": dataset_name,
        "record_id": record_id,
        "raw_dx": raw_dx,
        "mapped_targets": mapped_targets
    }

def discover_datasets_multiprocessed(input_dirs):
    """
    Finds all .hea files and uses a ProcessPool to quickly extract and filter metadata.
    """
    all_hea_files = []
    ptbxl_mapping = {}
    
    for input_dir in input_dirs:
        # Check for PTB-XL CSV
        possible_csvs = [
            os.path.join(input_dir, "ptbxl_database.csv"),
            os.path.join(input_dir, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1", "ptbxl_database.csv")
        ]
        ptbxl_csv = next((p for p in possible_csvs if os.path.exists(p)), None)
        if ptbxl_csv and not ptbxl_mapping:
            ptbxl_mapping = parse_ptbxl_metadata(ptbxl_csv)
            
        pattern = os.path.join(input_dir, '**', '*.hea')
        all_hea_files.extend(glob.glob(pattern, recursive=True))
        
    print(f"Scanning {len(all_hea_files)} ECG headers via CPU multiprocessing...")
    
    valid_records = []
    
    # Use ThreadPoolExecutor for I/O bound header reading
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
        futures = {executor.submit(scan_single_file, path, ptbxl_mapping): path for path in all_hea_files}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                valid_records.append(res)
                
    return valid_records
