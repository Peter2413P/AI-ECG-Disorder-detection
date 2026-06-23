import numpy as np

def extract_rhythm_features(rpeaks_indices, fs):
    """
    Extract global rhythm features based on R-peaks from Lead II (or globally).
    """
    features = {
        "heart_rate": np.nan,
        "rr_mean": np.nan,
        "rr_std": np.nan,
        "rmssd": np.nan,
        "pnn50": np.nan
    }
    
    if rpeaks_indices is None or len(rpeaks_indices) < 2:
        return features
        
    valid_rpeaks = [r for r in rpeaks_indices if r is not None and not np.isnan(r)]
    if len(valid_rpeaks) < 2:
        return features
        
    # RR intervals in milliseconds
    rr_intervals = np.diff(valid_rpeaks) / fs * 1000.0
    
    features["rr_mean"] = np.mean(rr_intervals)
    features["rr_std"] = np.std(rr_intervals)
    features["heart_rate"] = 60000.0 / features["rr_mean"] if features["rr_mean"] > 0 else np.nan
    
    if len(rr_intervals) > 1:
        rr_diffs = np.diff(rr_intervals)
        features["rmssd"] = np.sqrt(np.mean(rr_diffs**2))
        features["pnn50"] = np.sum(np.abs(rr_diffs) > 50) / len(rr_diffs) * 100.0
        
    return features
