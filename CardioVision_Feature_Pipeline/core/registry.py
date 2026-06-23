"""
Feature Registry
Maps feature groups to specific extracted features.
"""

FEATURES = {
    "rhythm": [
        "heart_rate",
        "rr_mean",
        "rr_std",
        "rmssd",
        "pnn50"
    ],
    "conduction": [
        "pr_interval",
        "qrs_duration",
        "qt_interval",
        "qtc"
    ],
    "p_wave": [
        "p_amplitude",
        "p_duration",
        "p_area"
    ],
    "qrs_wave": [
        "r_amplitude",
        "s_amplitude",
        "qrs_area",
        "qrs_energy",
        "rs_ratio"
    ],
    "st_segment": [
        "st_deviation",
        "st_slope"
    ],
    "t_wave": [
        "t_amplitude",
        "t_duration",
        "t_symmetry"
    ],
    "axis": [
        "qrs_axis",
        "t_axis"
    ],
    "entropy": [
        "shannon_entropy",
        "sample_entropy",
        "approximate_entropy"
    ],
    "frequency": [
        "dominant_frequency",
        "spectral_entropy",
        "band_power"
    ],
    "pacemaker": [
        "spike_count",
        "spike_amplitude",
        "spike_width",
        "spike_qrs_delay",
        "paced_beat_percentage"
    ]
}

def get_all_feature_names(include_global=True, include_per_lead=True):
    """
    Returns a flattened list of all feature column names expected in the final dataset.
    """
    from core.config import LEAD_NAMES
    
    all_features = []
    
    # Global features (Rhythm, Pacemaker, Entropy, Frequency are often calculated globally or on Lead II)
    global_groups = ["rhythm", "pacemaker", "entropy", "frequency", "axis"]
    if include_global:
        for group in global_groups:
            if group in FEATURES:
                for f in FEATURES[group]:
                    all_features.append(f"Global_{f}")
                    
    # Lead-specific features
    lead_groups = ["conduction", "p_wave", "qrs_wave", "st_segment", "t_wave"]
    if include_per_lead:
        for lead in LEAD_NAMES:
            for group in lead_groups:
                if group in FEATURES:
                    for f in FEATURES[group]:
                        all_features.append(f"Lead_{lead}_{f}")
                        
    return all_features
