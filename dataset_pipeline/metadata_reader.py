# dataset_pipeline/metadata_reader.py

import os
import glob
import pandas as pd
import wfdb

def load_wfdb_record(filepath):
    """
    Load a WFDB record (.hea / .dat or .mat).
    """
    try:
        # wfdb.rdsamp expects the path without the extension
        record_name = os.path.splitext(filepath)[0]
        signals, fields = wfdb.rdsamp(record_name)
        return signals, fields
    except Exception as e:
        return None, None

def parse_ptbxl_metadata(csv_path):
    """
    Parse the PTB-XL ptbxl_database.csv
    Returns a dictionary mapping filename to diagnoses (SCP codes).
    """
    try:
        df = pd.read_csv(csv_path, index_col='ecg_id')
        # scp_codes is a string representation of a dictionary. We extract keys.
        # Example: "{'NORM': 100.0, 'LMI': 0.0}" -> ["NORM", "LMI"]
        import ast
        mapping = {}
        for ecg_id, row in df.iterrows():
            # Use os.path.basename to just get '00001_hr' instead of 'records500/00000/00001_hr'
            filename = os.path.basename(str(row['filename_hr'])) 
            try:
                scp_dict = ast.literal_eval(row['scp_codes'])
                mapping[filename] = list(scp_dict.keys())
            except:
                mapping[filename] = []
        return mapping
    except Exception as e:
        return {}

def parse_physionet_header(header_path):
    """
    Parses a PhysioNet .hea file to extract dx (diagnoses) and demographics.
    """
    metadata = {
        "age": None,
        "gender": None,
        "diagnoses": []
    }
    try:
        with open(header_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    # Parse demographics
                    if 'Age:' in line:
                        age_str = line.split('Age:')[1].strip()
                        metadata['age'] = int(age_str) if age_str.isdigit() else None
                    if 'Sex:' in line:
                        metadata['gender'] = line.split('Sex:')[1].strip()
                    if 'Dx:' in line:
                        dx_str = line.split('Dx:')[1].strip()
                        metadata['diagnoses'] = [dx.strip() for dx in dx_str.split(',')]
    except Exception as e:
        pass
    
    return metadata

def discover_datasets(input_dir):
    """
    Scans the input directory and returns a list of all ECG files (.hea, .mat, .dat)
    along with their associated metadata if identifiable.
    """
    ecg_files = []
    
    # Recursively find all .hea files as they are the standard WFDB header
    search_pattern = os.path.join(input_dir, '**', '*.hea')
    for hea_path in glob.iglob(search_pattern, recursive=True):
        base_path = os.path.splitext(hea_path)[0]
        
        # Check if corresponding .mat or .dat exists
        has_mat = os.path.exists(base_path + '.mat')
        has_dat = os.path.exists(base_path + '.dat')
        
        if has_mat or has_dat:
            ecg_files.append({
                "base_path": base_path,
                "hea_path": hea_path,
                "data_path": base_path + ('.mat' if has_mat else '.dat'),
                "dataset_name": os.path.basename(os.path.dirname(hea_path))
            })
            
    return ecg_files
