import os
import glob
import pandas as pd
import json
import ast
from tqdm import tqdm
from labeling.unified_mapper import map_labels, save_mapping_to_json, TARGET_CLASSES
from core.logger import get_logger

logger = get_logger("BuildLabelDataset")

BASE_DIR = r"d:\College\intern\final"
PTBXL_META_PATH = r"d:\College\intern\final\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1\ptbxl_database.csv"

def load_snomed_to_text():
    """
    Loads SNOMED to text dictionary from the existing csv if available.
    """
    snomed_path = r"d:\College\intern\ECG_DATASETS\ecg_all_snomed_codes.csv"
    snomed_dict = {}
    if os.path.exists(snomed_path):
        try:
            df = pd.read_csv(snomed_path)
            for _, row in df.iterrows():
                code = str(row['SNOMED Code']).strip()
                name = str(row['Disorder Name']).strip()
                snomed_dict[code] = name
        except Exception as e:
            logger.warning(f"Could not load snomed codes: {e}")
    return snomed_dict

def parse_hea_file(hea_path):
    """
    Extracts age, sex, dx codes from a .hea file.
    """
    codes = []
    patient_id = None
    with open(hea_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("#Dx:"):
                dx_str = line.split("#Dx:")[1].strip()
                codes = [c.strip() for c in dx_str.split(',') if c.strip()]
            elif line.startswith("#Age:"):
                # We can grab age if needed, but sticking to labels
                pass
            # Just use base filename as patient_id if not present
    
    ecg_id = os.path.basename(hea_path).replace('.hea', '')
    return ecg_id, codes

def build_dataset():
    snomed_to_text = load_snomed_to_text()
    
    records = []
    mapping_reports = []
    
    datasets = {
        "Chapman": os.path.join(BASE_DIR, "chapman"),
        "Georgia": os.path.join(BASE_DIR, "georgia"),
        "PhysioNet": os.path.join(BASE_DIR, "physionet")
    }
    
    logger.info("Parsing PhysioNet-style datasets...")
    for ds_name, ds_path in datasets.items():
        if not os.path.exists(ds_path):
            logger.warning(f"Path not found for {ds_name}: {ds_path}")
            continue
            
        hea_files = glob.glob(os.path.join(ds_path, "*.hea"))
        for hea_path in tqdm(hea_files, desc=f"Parsing {ds_name}"):
            ecg_id, original_codes = parse_hea_file(hea_path)
            
            original_diagnosis = [snomed_to_text.get(c, "Unknown") for c in original_codes]
            mapped_dict, mapped_classes, unmapped = map_labels(original_codes)
            
            for c in original_codes:
                target = CLINICAL_CROSSWALK.get(c, "UNMAPPED") if c in globals().get("CLINICAL_CROSSWALK", {}) else "UNMAPPED"
                # wait we need CLINICAL_CROSSWALK imported
                pass # Handled below
            
            row = {
                "ecg_id": f"{ds_name.lower()}_{ecg_id}",
                "patient_id": f"{ds_name.lower()}_{ecg_id}",
                "dataset_source": ds_name,
                "original_codes": json.dumps(original_codes),
                "original_diagnosis": json.dumps(original_diagnosis),
                **mapped_dict
            }
            records.append(row)
            
            for code in original_codes:
                mapping_reports.append({"Code": code, "Original_Diagnosis": snomed_to_text.get(code, "Unknown"), "Dataset": ds_name})

    logger.info("Parsing PTB-XL metadata...")
    if os.path.exists(PTBXL_META_PATH):
        try:
            ptbxl_df = pd.read_csv(PTBXL_META_PATH)
            for _, row in tqdm(ptbxl_df.iterrows(), total=len(ptbxl_df), desc="Parsing PTB-XL"):
                ecg_id = str(row['ecg_id'])
                patient_id = str(row['patient_id'])
                scp_dict_str = row['scp_codes']
                try:
                    scp_dict = ast.literal_eval(scp_dict_str)
                    original_codes = list(scp_dict.keys())
                except:
                    original_codes = []
                    
                # PTB-XL provides some statements, but SCP codes themselves are the diagnosis keys
                original_diagnosis = original_codes # For SCP, the code is often the abbreviation
                
                mapped_dict, mapped_classes, unmapped = map_labels(original_codes)
                
                row_dict = {
                    "ecg_id": f"ptbxl_{ecg_id}",
                    "patient_id": f"ptbxl_{patient_id}",
                    "dataset_source": "PTB-XL",
                    "original_codes": json.dumps(original_codes),
                    "original_diagnosis": json.dumps(original_diagnosis),
                    **mapped_dict
                }
                records.append(row_dict)
                
                for code in original_codes:
                    mapping_reports.append({"Code": code, "Original_Diagnosis": code, "Dataset": "PTB-XL"})
        except Exception as e:
            logger.error(f"Failed to parse PTB-XL: {e}")

    df_final = pd.DataFrame(records)
    
    # Save datasets
    output_dir = os.path.join(BASE_DIR, "CardioVision_Feature_Pipeline", "outputs", "final_dataset")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "labels_dataset.csv")
    parquet_path = os.path.join(output_dir, "labels_dataset.parquet")
    
    df_final.to_csv(csv_path, index=False)
    df_final.to_parquet(parquet_path, index=False)
    logger.info(f"Saved Unified Dataset to {csv_path} and .parquet")
    
    # Generate Mapping Validation Report
    from labeling.unified_mapper import CLINICAL_CROSSWALK
    df_mapping = pd.DataFrame(mapping_reports).drop_duplicates(subset=['Code'])
    df_mapping['Mapped_To'] = df_mapping['Code'].apply(lambda c: CLINICAL_CROSSWALK.get(str(c).strip().upper(), "UNMAPPED"))
    
    mapping_report_path = os.path.join(output_dir, "mapping_validation_report.csv")
    df_mapping.to_csv(mapping_report_path, index=False)
    
    # Save JSON mapping
    json_path = os.path.join(output_dir, "label_mapping.json")
    save_mapping_to_json(json_path)

    total_codes = len(df_mapping)
    unmapped = len(df_mapping[df_mapping['Mapped_To'] == "UNMAPPED"])
    mapped = total_codes - unmapped
    mapping_pct = (mapped / total_codes) * 100 if total_codes > 0 else 0
    
    logger.info("--- Mapping Validation ---")
    logger.info(f"Total diagnosis codes found: {total_codes}")
    logger.info(f"Mapped successfully: {mapped}")
    logger.info(f"Unmapped: {unmapped}")
    logger.info(f"Mapping percentage: {mapping_pct:.2f}%")

if __name__ == "__main__":
    build_dataset()
