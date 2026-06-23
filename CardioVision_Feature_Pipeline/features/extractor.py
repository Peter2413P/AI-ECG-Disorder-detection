import numpy as np
from core.registry import get_all_feature_names
from core.config import LEAD_NAMES
from .rhythm import extract_rhythm_features
from .morphology import extract_morphology_features
from .entropy_freq import extract_entropy_freq_features
from .pacemaker import detect_pacemaker_spikes

class FeatureExtractor:
    def __init__(self):
        self.feature_names = get_all_feature_names()

    def extract_features(self, ecg_record):
        """
        Orchestrates feature extraction across all leads and globally.
        Returns a flat dictionary matching feature_names.
        """
        raw_signal = ecg_record['signal']
        cleaned_signal = ecg_record.get('cleaned_signal', raw_signal)
        fs = ecg_record['fs']
        delineations = ecg_record.get('delineation', {})
        
        flat_features = {k: np.nan for k in self.feature_names}
        
        # 1. Global Features (Using Lead II usually index 1)
        lead_ii_idx = LEAD_NAMES.index('II') if 'II' in LEAD_NAMES else 0
        lead_ii_delin = delineations.get(f"Lead_{lead_ii_idx}", {})
        
        # Rhythm
        rhythm_feats = extract_rhythm_features(lead_ii_delin.get('R_Peaks'), fs)
        for k, v in rhythm_feats.items():
            if f"Global_{k}" in flat_features:
                flat_features[f"Global_{k}"] = v
                
        # Global Entropy & Freq (on Lead II)
        ent_freq_feats = extract_entropy_freq_features(cleaned_signal[lead_ii_idx], fs)
        for k, v in ent_freq_feats.items():
            if f"Global_{k}" in flat_features:
                flat_features[f"Global_{k}"] = v
                
        # Pacemaker (Global max over all leads for simplicity, or just Lead II)
        pm_feats = detect_pacemaker_spikes(raw_signal[lead_ii_idx], fs)
        for k, v in pm_feats.items():
            if f"Global_{k}" in flat_features:
                flat_features[f"Global_{k}"] = v

        # 2. Lead-Specific Features
        for i, lead in enumerate(LEAD_NAMES):
            lead_delin = delineations.get(f"Lead_{i}", {})
            morph_feats = extract_morphology_features(cleaned_signal[i], lead_delin, fs)
            
            for k, v in morph_feats.items():
                col_name = f"Lead_{lead}_{k}"
                if col_name in flat_features:
                    flat_features[col_name] = v
                    
        # 3. Global Axis (needs Lead I and aVF)
        # Using R-wave amplitudes
        lead_i_idx = LEAD_NAMES.index('I')
        lead_avf_idx = LEAD_NAMES.index('aVF')
        r_i = flat_features.get(f"Lead_I_r_amplitude", 0)
        r_avf = flat_features.get(f"Lead_aVF_r_amplitude", 0)
        if not np.isnan(r_i) and not np.isnan(r_avf) and r_i != 0:
            flat_features["Global_qrs_axis"] = np.degrees(np.arctan(r_avf / r_i))

        ecg_record['features'] = flat_features
        return flat_features
