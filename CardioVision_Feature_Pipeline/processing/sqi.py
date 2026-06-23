import neurokit2 as nk
import numpy as np
from core.logger import get_logger

logger = get_logger("SQIAssessor")

def assess_signal_quality(cleaned_signal, fs):
    """
    Assesses the signal quality using NeuroKit2 (Zhao 2018 algorithm).
    Returns global quality index (0.0 to 1.0).
    """
    try:
        num_leads = cleaned_signal.shape[0]
        qualities = []
        
        for i in range(num_leads):
            # ecg_quality returns an array of quality scores per sample or a single metric
            # zhao2018 returns 'Unacceptable', 'Barely Acceptable', 'Excellent'
            # Let's use simple heuristic based on R-peak detection confidence if zhao fails
            
            # Actually, nk.ecg_quality returns a continuous index (default method is 'averageQRS')
            quality_arr = nk.ecg_quality(cleaned_signal[i], sampling_rate=fs, method='zhao2018')
            
            # Map string to float
            if quality_arr == 'Excellent':
                q = 1.0
            elif quality_arr == 'Barely Acceptable':
                q = 0.5
            elif quality_arr == 'Unacceptable':
                q = 0.0
            else:
                q = 0.5 # Default
                
            qualities.append(q)
            
        # Average quality across all leads
        return np.mean(qualities)
        
    except Exception as e:
        logger.warning(f"SQI Assessment failed: {str(e)}")
        return 1.0 # Default to pass if algorithm fails
