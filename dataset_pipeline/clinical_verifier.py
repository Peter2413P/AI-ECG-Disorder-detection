# dataset_pipeline/clinical_verifier.py

def verify_clinical_rules(disorder, features):
    """
    Verifies if the extracted morphological features support the assigned disorder.
    Returns (True, "Valid") if supported, or (False, "Reason") if contradicted.
    """
    
    hr = features.get("heart_rate")
    qrs = features.get("qrs_duration")
    p_wave = features.get("p_wave_present")
    
    if disorder == "Normal_Sinus_Rhythm":
        if hr is not None and (hr < 50 or hr > 110): # Slight buffer for edge cases
            return False, f"Heart rate ({hr:.1f}) out of NSR bounds (50-110)."
        if not p_wave:
            return False, "Missing P waves, contradicts NSR."
            
    elif disorder == "Sinus_Tachycardia":
        if hr is not None and hr < 95:
            return False, f"Heart rate ({hr:.1f}) too low for Tachycardia."
            
    elif disorder == "RBBB" or disorder == "LBBB":
        # Bundle branch blocks require widened QRS
        if qrs is not None and qrs < 110: 
            return False, f"QRS duration ({qrs:.1f}ms) too narrow for BBB (>110ms required)."
            
    elif disorder == "Ventricular_Fibrillation_Flutter":
        if p_wave:
            # VF typically has no organized P waves, but neurokit might misidentify noise. 
            # We won't strictly reject, but we could lower confidence.
            pass
            
    # Default to passing if features are missing or disorder lacks strict rules implemented here.
    return True, "Valid morphology"
