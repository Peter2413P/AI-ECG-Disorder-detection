import pandas as pd
from core.logger import get_logger
from core.config import TARGET_CLASSES

logger = get_logger("DatasetAudit")

class DatasetAudit:
    def audit_dataset(self, df):
        """
        Runs rigorous checks on the final dataset to ensure integrity before modeling.
        """
        logger.info("--- Starting Strict Dataset Audit ---")
        issues_found = False
        
        # 1. Class Counts & Check exactly 12 classes
        missing_classes = [c for c in TARGET_CLASSES if c not in df.columns]
        if missing_classes:
            logger.error(f"Missing target columns: {missing_classes}")
            issues_found = True
            
        logger.info("Class Distributions:")
        for c in TARGET_CLASSES:
            if c in df.columns:
                count = df[c].sum()
                logger.info(f"  {c}: {count}")
                if count == 0:
                    logger.error(f"  CRITICAL: 0 samples found for {c}!")
                    issues_found = True
                    
        # 2. Duplicate ECG IDs
        if 'ecg_id' in df.columns:
            dupes = df['ecg_id'].duplicated().sum()
            if dupes > 0:
                logger.error(f"Found {dupes} duplicate ECG IDs!")
                issues_found = True
        else:
            logger.error("Missing 'ecg_id' column!")
            issues_found = True
            
        # 3. Patient Leakage potential (just warn if no patient_id)
        if 'patient_id' not in df.columns:
            logger.warning("No 'patient_id' column found. Cannot guarantee leak-free cross-validation.")
            
        # 4. Source Tracking
        if 'dataset_source' in df.columns:
            sources = df['dataset_source'].value_counts()
            logger.info(f"Dataset Sources:\n{sources}")
        else:
            logger.error("Missing 'dataset_source' column! Needed for bias tracking.")
            issues_found = True
            
        if not issues_found:
            logger.info("--- Audit Passed: Dataset is ready for XAI Pipeline ---")
        else:
            logger.error("--- Audit Failed: Resolve issues before training ---")
