import neurokit2 as nk
import numpy as np
from core.logger import get_logger

logger = get_logger("ECGDelineator")

class ECGDelineator:
    def delineate_lead(self, cleaned_signal_1d, fs):
        """
        Detects R-peaks and delineates P, QRS, and T waves for a single lead.
        Returns a dictionary of boundaries.
        """
        try:
            # 1. R-Peak detection (Pan-Tompkins)
            _, rpeaks = nk.ecg_peaks(cleaned_signal_1d, sampling_rate=fs)
            rpeaks_indices = rpeaks['ECG_R_Peaks']
            
            if len(rpeaks_indices) == 0:
                logger.warning("No R-peaks detected.")
                return None
                
            # 2. Wave delineation (DWT method)
            _, waves = nk.ecg_delineate(cleaned_signal_1d, rpeaks_indices, sampling_rate=fs, method="dwt")
            
            boundaries = {
                'R_Peaks': rpeaks_indices.tolist(),
                'P_Onsets': waves.get('ECG_P_Onsets', []),
                'P_Offsets': waves.get('ECG_P_Offsets', []),
                'QRS_Onsets': waves.get('ECG_R_Onsets', []), # R_Onsets corresponds to QRS onset
                'QRS_Offsets': waves.get('ECG_R_Offsets', []), # R_Offsets corresponds to QRS offset
                'T_Onsets': waves.get('ECG_T_Onsets', []),
                'T_Offsets': waves.get('ECG_T_Offsets', [])
            }
            
            # Convert NaNs to None for JSON serialization
            for key, val in boundaries.items():
                if isinstance(val, list):
                    boundaries[key] = [int(v) if not np.isnan(v) else None for v in val]
                    
            return boundaries
            
        except Exception as e:
            logger.error(f"Delineation error: {str(e)}")
            return None

    def delineate_record(self, ecg_record):
        """
        Delineates all leads in an ECG record.
        """
        signal = ecg_record.get('cleaned_signal', ecg_record['signal'])
        fs = ecg_record['fs']
        num_leads = signal.shape[0]
        
        delineation_results = {}
        for i in range(num_leads):
            # Delineate each lead independently
            boundaries = self.delineate_lead(signal[i], fs)
            delineation_results[f"Lead_{i}"] = boundaries
            
        ecg_record['delineation'] = delineation_results
        return delineation_results
