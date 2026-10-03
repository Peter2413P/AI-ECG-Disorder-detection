# dataset_pipeline/config.py

import os

# --- Target Disorders ---
TARGET_DISORDERS = [
    "Normal_Sinus_Rhythm",
    "Sinus_Tachycardia",
    "Sinus_Arrhythmia",
    "PAC",
    "RBBB",
    "LBBB",
    "IVCD",
    "Delta_Wave_WPW",
    "Persistent_ST_Elevation",
    "Left_Atrial_Enlargement",
    "Ventricular_Fibrillation_Flutter",
    "Pacemaker_Rhythm"
]

# --- Diagnosis Mapping ---
# A universal mapping dictionary. Keys are lowercase for case-insensitive matching.
DIAGNOSIS_MAP = {
    # Normal Sinus Rhythm
    "normal": "Normal_Sinus_Rhythm",
    "nsr": "Normal_Sinus_Rhythm",
    "sinus rhythm": "Normal_Sinus_Rhythm",
    "426783006": "Normal_Sinus_Rhythm",
    "norm": "Normal_Sinus_Rhythm",
    "sry": "Normal_Sinus_Rhythm",

    # Sinus Tachycardia
    "sinus tachycardia": "Sinus_Tachycardia",
    "tachycardia": "Sinus_Tachycardia",
    "427084000": "Sinus_Tachycardia",
    "stach": "Sinus_Tachycardia",

    # Sinus Arrhythmia
    "sinus arrhythmia": "Sinus_Arrhythmia",
    "427322008": "Sinus_Arrhythmia",
    "sarrh": "Sinus_Arrhythmia",

    # PAC
    "pac": "PAC",
    "premature atrial contraction": "PAC",
    "premature atrial complex": "PAC",
    "apc": "PAC",
    "atrial premature beat": "PAC",
    "284470004": "PAC",

    # RBBB
    "rbbb": "RBBB",
    "right bundle branch block": "RBBB",
    "crbbb": "RBBB",
    "713427006": "RBBB",
    "59118001": "RBBB",

    # LBBB
    "lbbb": "LBBB",
    "left bundle branch block": "LBBB",
    "clbbb": "LBBB",
    "164909002": "LBBB",

    # IVCD
    "ivcd": "IVCD",
    "intraventricular conduction delay": "IVCD",
    "nonspecific intraventricular conduction block": "IVCD",
    "713426002": "IVCD",
    "164947007": "IVCD",

    # WPW
    "wpw": "Delta_Wave_WPW",
    "wolff-parkinson-white": "Delta_Wave_WPW",
    "delta wave": "Delta_Wave_WPW",
    "74390002": "Delta_Wave_WPW",

    # Persistent ST Elevation
    "persistent st elevation": "Persistent_ST_Elevation",
    "st elevation": "Persistent_ST_Elevation",
    "ste": "Persistent_ST_Elevation",
    "164865005": "Persistent_ST_Elevation",

    # LAE
    "lae": "Left_Atrial_Enlargement",
    "left atrial enlargement": "Left_Atrial_Enlargement",
    "left atrial hypertrophy": "Left_Atrial_Enlargement",
    "445118002": "Left_Atrial_Enlargement",

    # VF / VFL
    "vf": "Ventricular_Fibrillation_Flutter",
    "vfl": "Ventricular_Fibrillation_Flutter",
    "ventricular fibrillation": "Ventricular_Fibrillation_Flutter",
    "ventricular flutter": "Ventricular_Fibrillation_Flutter",
    "164896001": "Ventricular_Fibrillation_Flutter",
    "71908006": "Ventricular_Fibrillation_Flutter",

    # Pacemaker Rhythm
    "pacemaker rhythm": "Pacemaker_Rhythm",
    "paced rhythm": "Pacemaker_Rhythm",
    "pacemaker": "Pacemaker_Rhythm",
    "pm": "Pacemaker_Rhythm",
    "10339000": "Pacemaker_Rhythm",
}

# --- Scoring Weights ---
SCORING_WEIGHTS = {
    "clinical_match": 0.40,
    "signal_quality": 0.30,
    "feature_completeness": 0.20,
    "annotation_confidence": 0.10
}

# Add default output structure variables
OUTPUT_DIR = os.path.join(os.getcwd(), "Extracted_ECG_Disorders_Full")
