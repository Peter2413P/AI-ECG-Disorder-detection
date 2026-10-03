import numpy as np
from .config import TARGET_DISORDERS

def verify_batch_rules(features, target_disorder):
    """
    Applies vectorized numpy boolean masking to evaluate rules across the entire batch instantly.
    Returns a boolean array of shape (B,) where True means the record passed.
    """
    hr = features["heart_rate"] # Shape (B,)
    qrs = features["qrs_duration"] # Shape (B,)
    
    B = len(hr)
    
    # Initialize all as passing, then apply strict exclusionary masks
    passed = np.ones(B, dtype=bool)
    
    if target_disorder == "Normal_Sinus_Rhythm":
        # Reject if HR < 50 or HR > 110
        passed &= (hr >= 50) & (hr <= 110)
        
    elif target_disorder == "Sinus_Tachycardia":
        # Reject if HR < 95
        passed &= (hr >= 95)
        
    elif target_disorder in ["RBBB", "LBBB"]:
        # Broad QRS required
        passed &= (qrs > 110)
        
    # We can add more vectorized rules here as needed.
    # The point of this module is that evaluating 1,024 records takes ~0.001s.
    
    return passed
