import numpy as np
from core.config import TARGET_CLASSES
from core.logger import get_logger

logger = get_logger("LabelMapper")

# Example crosswalk based on SNOMED / SCP codes mapped to the 12 targets
LABEL_CROSSWALK = {
    "Normal_Sinus_Rhythm": ["SR", "NORM", "NSR", "426783006"],
    "Sinus_Tachycardia": ["STACH", "427084000", "ST"],
    "Sinus_Arrhythmia": ["SARRH", "427393009", "SA"],
    "PAC": ["PAC", "284470004", "SVPB", "Premature atrial contraction"],
    "RBBB": ["CRBBB", "IRBBB", "713427006", "164907000", "59110001", "RBBB"],
    "LBBB": ["CLBBB", "ILBBB", "164909002", "251146004", "LBBB"],
    "IVCD": ["IVCD", "164947007", "713426002"],
    "Delta_Wave": ["WPW", "74390002", "Wolff-Parkinson-White"],
    "Persistent_ST_Elevation": ["STE_", "164865005", "STEMI"],
    "Left_Atrial_Enlargement": ["LAO/LAE", "445118002", "LAE"],
    "Ventricular_Fibrillation_Flutter": ["VF", "VFLUT", "164890007", "426749004", "Vfib"],
    "Pacemaker_Rhythm": ["PACE", "PM", "10370003", "Paced rhythm"]
}

class LabelMapper:
    def __init__(self):
        self.target_classes = TARGET_CLASSES
        # Create reverse mapping for fast lookup O(1)
        self.source_to_target = {}
        for target, sources in LABEL_CROSSWALK.items():
            for src in sources:
                self.source_to_target[src.upper()] = target
                
    def map_labels(self, source_labels):
        """
        Converts a list of source labels (SNOMED, SCP, text) into a binary vector of length 12.
        Returns a dictionary for easy merging with feature vectors.
        """
        vector = {target: 0 for target in self.target_classes}
        
        if not source_labels:
            return vector
            
        for label in source_labels:
            label_upper = str(label).upper()
            if label_upper in self.source_to_target:
                target_class = self.source_to_target[label_upper]
                vector[target_class] = 1
                
        return vector

    def process(self, ecg_record):
        """
        Updates ecg_record with mapped labels.
        """
        mapped_dict = self.map_labels(ecg_record['labels'])
        ecg_record['mapped_labels'] = mapped_dict
        return mapped_dict
