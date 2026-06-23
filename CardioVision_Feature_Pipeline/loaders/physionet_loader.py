import wfdb
import scipy.io as sio
import numpy as np
import os
from .base_loader import BaseLoader
from core.logger import get_logger

logger = get_logger("PhysioNetLoader")

class PhysioNetLoader(BaseLoader):
    def load_record(self, hea_path, dataset_source, labels=None, patient_id=None):
        """
        Load ECG record from WFDB formats (.hea + .dat/.mat)
        Returns unified dictionary.
        """
        if not os.path.exists(hea_path):
            logger.error(f"Header file not found: {hea_path}")
            return None
            
        base_path = os.path.splitext(hea_path)[0]
        ecg_id = os.path.basename(base_path)
        
        try:
            # Read header
            record = wfdb.rdheader(base_path)
            fs = record.fs
            
            # Read signal
            if os.path.exists(base_path + ".dat"):
                record_data = wfdb.rdrecord(base_path)
                signal = record_data.p_signal.T # shape (leads, samples)
            elif os.path.exists(base_path + ".mat"):
                mat_data = sio.loadmat(base_path + ".mat")
                signal = mat_data['val']
                # Scale if needed, typically mat files are in ADU and need scaling
                if hasattr(record, 'adc_gain') and record.adc_gain is not None:
                    gains = np.array(record.adc_gain)
                    # Handle zero gain to avoid division by zero
                    gains[gains == 0] = 1
                    signal = signal.astype(np.float64) / gains[:, np.newaxis]
            else:
                logger.error(f"Signal data file not found for: {base_path}")
                return None
                
            return {
                "ecg_id": ecg_id,
                "patient_id": patient_id if patient_id else ecg_id,
                "dataset_source": dataset_source,
                "signal": signal,
                "fs": fs,
                "labels": labels if labels else []
            }
            
        except Exception as e:
            logger.error(f"Error loading {hea_path}: {str(e)}")
            return None
