import numpy as np
import scipy.signal as signal
import neurokit2 as nk

def extract_entropy_freq_features(signal_1d, fs):
    """
    Extract entropy and frequency domain features.
    """
    features = {
        "shannon_entropy": np.nan,
        "sample_entropy": np.nan,
        "dominant_frequency": np.nan,
        "spectral_entropy": np.nan,
        "band_power": np.nan
    }
    
    if len(signal_1d) < fs:
        return features
        
    try:
        # Entropy
        features["shannon_entropy"] = nk.entropy_shannon(signal_1d)[0]
        # Sample entropy is slow on long signals, might want to downsample or use subset
        features["sample_entropy"] = nk.entropy_sample(signal_1d[::5])[0] 
        
        # Frequency
        f, Pxx = signal.welch(signal_1d, fs, nperseg=1024)
        features["dominant_frequency"] = f[np.argmax(Pxx)]
        
        # Spectral Entropy
        Pxx_norm = Pxx / np.sum(Pxx)
        features["spectral_entropy"] = -np.sum(Pxx_norm * np.log2(Pxx_norm + 1e-12))
        
        # Band Power (e.g., 2-10 Hz ratio to 10-40 Hz)
        low_band = np.logical_and(f >= 2, f <= 10)
        high_band = np.logical_and(f > 10, f <= 40)
        if np.sum(Pxx[high_band]) > 0:
            features["band_power"] = np.sum(Pxx[low_band]) / np.sum(Pxx[high_band])
            
    except Exception:
        pass
        
    return features
