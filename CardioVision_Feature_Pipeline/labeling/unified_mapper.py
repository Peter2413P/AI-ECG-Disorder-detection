import json
import os
import pandas as pd

# The exact 12 Target Classes
TARGET_CLASSES = [
    "NSR",
    "Sinus_Tachycardia",
    "Sinus_Arrhythmia",
    "PAC",
    "RBBB",
    "LBBB",
    "IVCD",
    "WPW",
    "Persistent_ST_Elevation",
    "LAE",
    "VF_Flutter",
    "Pacemaker_Rhythm"
]

# Clinical Crosswalk: Maps SNOMED / SCP codes or standard strings to our 12 classes
CLINICAL_CROSSWALK = {
    # 1. Normal Sinus Rhythm (NSR)
    "426783006": "NSR",
    "164861001": "NSR", # Normal ECG
    "NORM": "NSR",      # PTB-XL SCP
    "SR": "NSR",
    "NSR": "NSR",
    
    # 2. Sinus Tachycardia
    "426177001": "Sinus_Tachycardia",
    "STACH": "Sinus_Tachycardia", # PTB-XL
    
    # 3. Sinus Arrhythmia
    "427084000": "Sinus_Arrhythmia",
    "427393009": "Sinus_Arrhythmia",
    "SARRH": "Sinus_Arrhythmia", # PTB-XL
    
    # 4. PAC (Premature Atrial Contraction)
    "284470004": "PAC",
    "54329005": "PAC", # Atrial Premature Depolarization
    "63593006": "PAC", # Supraventricular Premature Beat
    "PAC": "PAC",      # PTB-XL
    "SVPB": "PAC",
    "SVPC": "PAC",
    
    # 5. RBBB (Right Bundle Branch Block)
    "59118001": "RBBB",   # RBBB
    "713427006": "RBBB",  # Complete RBBB
    "713426002": "RBBB",  # Incomplete RBBB
    "CRBBB": "RBBB",      # PTB-XL Complete RBBB
    "IRBBB": "RBBB",      # PTB-XL Incomplete RBBB
    "RBBB": "RBBB",
    
    # 6. LBBB (Left Bundle Branch Block)
    "164909002": "LBBB",  # LBBB
    "251120003": "LBBB",  # Incomplete LBBB
    "CLBBB": "LBBB",      # PTB-XL Complete LBBB
    "ILBBB": "LBBB",      # PTB-XL Incomplete LBBB
    "LBBB": "LBBB",
    
    # 7. IVCD (Intraventricular Conduction Delay / QRS Widening)
    "111975006": "IVCD",  # Prolonged QRS
    "164947007": "IVCD",  # Prolonged QT Interval (sometimes grouped, but IVCD is specifically QRS widening. Let's keep 164947007 separate if it's strictly QT, but usually IVCD is 713426002 or prolonged QRS 111975006)
    "164942001": "IVCD",  # QT Prolongation - wait, IVCD is strictly QRS widening. We'll map Prolonged QRS.
    "6374002": "IVCD",    # Bundle Branch Block (unspecified)
    "IVCD": "IVCD",       # PTB-XL
    
    # 8. WPW (Delta Wave)
    "164951009": "WPW",   # Delta Wave
    "74390002": "WPW",    # WPW
    "195101003": "WPW",   # Pre-excitation Syndrome
    "WPW": "WPW",         # PTB-XL
    
    # 9. Persistent ST Elevation
    "164930006": "Persistent_ST_Elevation", # ST Elevation
    "57054005": "Persistent_ST_Elevation",  # Acute MI (often STEMI)
    "164865005": "Persistent_ST_Elevation", # MI (grouping STEMI equivalents)
    "STE": "Persistent_ST_Elevation",       # PTB-XL
    "AMI": "Persistent_ST_Elevation",       # Acute MI PTB-XL
    "IMI": "Persistent_ST_Elevation",       # Inferior MI
    "ASMI": "Persistent_ST_Elevation",      # Anteroseptal MI
    "ALMI": "Persistent_ST_Elevation",      # Anterolateral MI
    "INJAS": "Persistent_ST_Elevation",     # Subendocardial injury PTB-XL
    "INJAL": "Persistent_ST_Elevation",
    "INJIN": "Persistent_ST_Elevation",
    "INJLA": "Persistent_ST_Elevation",
    
    # 10. Left Atrial Enlargement (LAE)
    "67741000119109": "LAE", # Left Atrial Abnormality
    "446813000": "LAE",      # Left Atrial Hypertrophy
    "67198005": "LAE",       # Biatrial Enlargement
    "LAE": "LAE",            # PTB-XL Left Atrial Enlargement
    "LAO/LAE": "LAE",        # PTB-XL
    
    # 11. Ventricular Fibrillation / Flutter
    "164884008": "VF_Flutter", # VFib
    "164896001": "VF_Flutter", # VTach (often grouped in lethal arrhythmias, but strictly VFib here)
    "11157007": "VF_Flutter",  # Ventricular Trigeminy (maybe not, wait, VFib is specific)
    "VF": "VF_Flutter",        # PTB-XL
    "VFLUT": "VF_Flutter",     # PTB-XL
    
    # 12. Pacemaker Rhythm
    "10370003": "Pacemaker_Rhythm",
    "698252002": "Pacemaker_Rhythm", # Artificial Pacemaker
    "PACE": "Pacemaker_Rhythm",      # PTB-XL
    "PM": "Pacemaker_Rhythm"         # PTB-XL
}

def map_labels(original_codes):
    """
    Given a list of original codes (SNOMED or SCP), map them to the 12 target classes.
    Returns:
        dict: Mapping {class_name: 1/0}
        list: mapped_classes
        list: unmapped_codes
    """
    mapped_dict = {c: 0 for c in TARGET_CLASSES}
    mapped_classes = set()
    unmapped_codes = []
    
    for code in original_codes:
        code_upper = str(code).strip().upper()
        
        # Check direct mapping
        if code_upper in CLINICAL_CROSSWALK:
            target = CLINICAL_CROSSWALK[code_upper]
            mapped_dict[target] = 1
            mapped_classes.add(target)
        else:
            # Maybe try to clean it
            clean_code = code_upper.replace(" ", "")
            if clean_code in CLINICAL_CROSSWALK:
                target = CLINICAL_CROSSWALK[clean_code]
                mapped_dict[target] = 1
                mapped_classes.add(target)
            else:
                unmapped_codes.append(str(code))
                
    return mapped_dict, list(mapped_classes), unmapped_codes

def save_mapping_to_json(output_path):
    # Output the reverse mapping for documentation
    reverse_mapping = {c: [] for c in TARGET_CLASSES}
    for code, target in CLINICAL_CROSSWALK.items():
        reverse_mapping[target].append(code)
        
    with open(output_path, 'w') as f:
        json.dump(reverse_mapping, f, indent=4)
