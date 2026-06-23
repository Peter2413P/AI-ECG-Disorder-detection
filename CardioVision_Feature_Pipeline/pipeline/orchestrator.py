import os
import json
import pandas as pd
from tqdm import tqdm
from loaders.physionet_loader import PhysioNetLoader
from processing.preprocessor import ECGPreprocessor
from processing.delineator import ECGDelineator
from processing.sqi import assess_signal_quality
from features.extractor import FeatureExtractor
from labeling.mapper import LabelMapper
from core.logger import get_logger
from core.config import CLEANED_DIR, DELINEATION_DIR, FEATURES_DIR, FINAL_DATASET_DIR

logger = get_logger("PipelineOrchestrator")

class PipelineOrchestrator:
    def __init__(self):
        self.loader = PhysioNetLoader()
        self.preprocessor = ECGPreprocessor()
        self.delineator = ECGDelineator()
        self.extractor = FeatureExtractor()
        self.mapper = LabelMapper()
        
        # Ensure output dirs exist
        os.makedirs(CLEANED_DIR, exist_ok=True)
        os.makedirs(DELINEATION_DIR, exist_ok=True)
        os.makedirs(FEATURES_DIR, exist_ok=True)
        os.makedirs(FINAL_DATASET_DIR, exist_ok=True)

    def process_record(self, record_path, dataset_source, labels, patient_id=None):
        """
        Processes a single ECG record through the entire pipeline.
        Saves intermediate outputs and returns the final feature dictionary.
        """
        # 1. Load
        record = self.loader.load_record(record_path, dataset_source, labels, patient_id)
        if not record:
            return None
            
        ecg_id = record['ecg_id']
            
        # 2. Preprocess & SQI
        cleaned_signal = self.preprocessor.process(record)
        sqi = assess_signal_quality(cleaned_signal, record['fs'])
        if sqi < 0.2:
            logger.warning(f"Record {ecg_id} failed SQI assessment (SQI={sqi:.2f}). Skipping.")
            return None
            
        # Save intermediate cleaned signal
        np_save_path = os.path.join(CLEANED_DIR, f"{ecg_id}_cleaned.npy")
        import numpy as np
        np.save(np_save_path, cleaned_signal)
        
        # 3. Delineate
        delineations = self.delineator.delineate_record(record)
        
        # Save intermediate delineations
        delin_save_path = os.path.join(DELINEATION_DIR, f"{ecg_id}_delineation.json")
        with open(delin_save_path, 'w') as f:
            json.dump(delineations, f)
            
        # 4. Extract Features
        features = self.extractor.extract_features(record)
        
        # Save intermediate features
        feat_save_path = os.path.join(FEATURES_DIR, f"{ecg_id}_features.json")
        with open(feat_save_path, 'w') as f:
            json.dump(features, f)
            
        # 5. Map Labels
        mapped_labels = self.mapper.process(record)
        
        # 6. Combine for final row
        final_row = {
            "ecg_id": ecg_id,
            "patient_id": record['patient_id'],
            "dataset_source": record['dataset_source'],
            **features,
            **mapped_labels
        }
        
        return final_row

    def run_pipeline(self, metadata_df):
        """
        Runs the pipeline over a dataframe of metadata containing paths and labels.
        Expects columns: 'hea_path', 'dataset_source', 'labels' (list or comma-separated string), 'patient_id'
        """
        logger.info(f"Starting pipeline for {len(metadata_df)} records...")
        results = []
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            try:
                # Handle labels if string
                labels = row.get('labels', row.get('original_codes', '[]'))
                if isinstance(labels, str):
                    if labels.startswith('['):
                        import json
                        try:
                            labels = json.loads(labels)
                        except Exception:
                            # Fallback if json parsing fails
                            labels = [l.strip() for l in labels.strip('[]').replace('"', '').replace("'", '').split(',')]
                    else:
                        labels = [l.strip() for l in labels.split(',')]
                    
                final_row = self.process_record(
                    record_path=row['hea_path'],
                    dataset_source=row['dataset_source'],
                    labels=labels,
                    patient_id=row.get('patient_id', None)
                )
                
                if final_row:
                    results.append(final_row)
                    
            except Exception as e:
                logger.error(f"Failed to process record at index {idx}: {str(e)}")
                
        # Generate Final Dataset
        if not results:
            logger.error("No valid records processed!")
            return None
            
        final_df = pd.DataFrame(results)
        
        parquet_path = os.path.join(FINAL_DATASET_DIR, "dataset.parquet")
        csv_path = os.path.join(FINAL_DATASET_DIR, "dataset.csv")
        
        final_df.to_parquet(parquet_path, index=False)
        final_df.to_csv(csv_path, index=False)
        logger.info(f"Pipeline complete. Final dataset saved to {parquet_path}")
        
        return final_df
