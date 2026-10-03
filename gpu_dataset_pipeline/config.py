import os
import torch

# Base I/O
OUTPUT_DIR = os.path.join(os.getcwd(), "Extracted")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardware & Tensor Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adaptive Batch Sizing
if torch.cuda.is_available():
    BATCH_SIZE = 512 
else:
    BATCH_SIZE = 128 # Smaller batch for CPU memory constraints

# Uniform Signal Length (10 seconds @ 500Hz)
TARGET_FS = 500
TARGET_LENGTH = 5000
NUM_LEADS = 12

# Target Disorders
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

# Strict diagnosis exclusion words
EXCLUSIONS = [
    "mi", "myocardial infarction", "nstemi", "stemi", "pvc", 
    "av block", "junctional", "cardiomyopathy", "hypertrophy", 
    "ischemia", "bradycardia", "pericarditis", "myocarditis"
]

# Mapping dictionary for strict filtering
DIAGNOSIS_MAP = {
    # 1. NSR
    "nsr": "Normal_Sinus_Rhythm",
    "normal sinus rhythm": "Normal_Sinus_Rhythm",
    "normal": "Normal_Sinus_Rhythm",
    "sinus rhythm": "Normal_Sinus_Rhythm",
    "426783006": "Normal_Sinus_Rhythm",

    # 2. Sinus Tachycardia
    "sinus tachycardia": "Sinus_Tachycardia",
    "tachycardia": "Sinus_Tachycardia",
    "stach": "Sinus_Tachycardia",

    # 3. Sinus Arrhythmia
    "sinus arrhythmia": "Sinus_Arrhythmia",
    "sarrh": "Sinus_Arrhythmia",

    # 4. PAC
    "pac": "PAC",
    "premature atrial contraction": "PAC",
    "svcb": "PAC",

    # 5. RBBB
    "rbbb": "RBBB",
    "right bundle branch block": "RBBB",
    "crbbb": "RBBB",

    # 6. LBBB
    "lbbb": "LBBB",
    "left bundle branch block": "LBBB",
    "clbbb": "LBBB",

    # 7. IVCD
    "ivcd": "IVCD",
    "intraventricular conduction delay": "IVCD",

    # 8. WPW
    "wpw": "Delta_Wave_WPW",
    "wolff-parkinson-white": "Delta_Wave_WPW",
    "delta wave": "Delta_Wave_WPW",

    # 9. Persistent ST Elevation
    "persistent st elevation": "Persistent_ST_Elevation",
    "st elevation": "Persistent_ST_Elevation",
    "ste": "Persistent_ST_Elevation",

    # 10. LAE
    "lae": "Left_Atrial_Enlargement",
    "left atrial enlargement": "Left_Atrial_Enlargement",
    "p mitrale": "Left_Atrial_Enlargement",

    # 11. Ventricular Fibrillation / Flutter
    "vf": "Ventricular_Fibrillation_Flutter",
    "vfl": "Ventricular_Fibrillation_Flutter",
    "ventricular fibrillation": "Ventricular_Fibrillation_Flutter",
    "ventricular flutter": "Ventricular_Fibrillation_Flutter",

    # 12. Pacemaker Rhythm
    "pacemaker": "Pacemaker_Rhythm",
    "paced rhythm": "Pacemaker_Rhythm",
    "pacemaker rhythm": "Pacemaker_Rhythm"
}
