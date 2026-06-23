"""
Centralized configuration for the ECG Feature Pipeline.
"""

# Signal Parameters
TARGET_FS = 500  # Resample all signals to 500 Hz for consistency
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Target Classes
TARGET_CLASSES = [
    "Normal_Sinus_Rhythm",
    "Sinus_Tachycardia",
    "Sinus_Arrhythmia",
    "PAC",
    "RBBB",
    "LBBB",
    "IVCD",
    "Delta_Wave",
    "Persistent_ST_Elevation",
    "Left_Atrial_Enlargement",
    "Ventricular_Fibrillation_Flutter",
    "Pacemaker_Rhythm"
]

# Preprocessing Parameters
BASELINE_FILTER_CUTOFF = 0.5  # Hz
POWERLINE_FREQ_1 = 50.0 # Hz
POWERLINE_FREQ_2 = 60.0 # Hz
LOWPASS_CUTOFF = 40.0  # Hz

# Paths
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # this is core/
PIPELINE_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

DATA_DIR = os.path.join(PIPELINE_ROOT, "data")
OUTPUTS_DIR = os.path.join(PIPELINE_ROOT, "outputs")
CLEANED_DIR = os.path.join(OUTPUTS_DIR, "cleaned_signals")
DELINEATION_DIR = os.path.join(OUTPUTS_DIR, "delineation_results")
FEATURES_DIR = os.path.join(OUTPUTS_DIR, "feature_tables")
FINAL_DATASET_DIR = os.path.join(OUTPUTS_DIR, "final_dataset")
