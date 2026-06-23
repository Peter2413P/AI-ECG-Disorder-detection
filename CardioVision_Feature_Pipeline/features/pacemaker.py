import numpy as np
import scipy.signal as signal

def detect_pacemaker_spikes(raw_signal_1d, fs):
    """
    Detects pacemaker spikes on raw signal before low-pass filtering.
    """
    features = {
        "spike_count": 0,
        "spike_amplitude": np.nan,
        "spike_width": np.nan,
        "spike_qrs_delay": np.nan,
        "paced_beat_percentage": 0.0
    }
    
    # 1. High-pass filter (>15Hz) to isolate sharp spikes
    b, a = signal.butter(4, 15.0 / (0.5 * fs), btype='highpass')
    hp_sig = signal.filtfilt(b, a, raw_signal_1d)
    
    # 2. Derivative and squaring to emphasize spikes
    diff_sig = np.diff(hp_sig)
    sq_sig = diff_sig ** 2
    
    # 3. Thresholding (very high threshold compared to mean)
    threshold = np.mean(sq_sig) + 5 * np.std(sq_sig)
    
    # Find peaks above threshold
    peaks, _ = signal.find_peaks(sq_sig, height=threshold, distance=int(0.1*fs))
    
    if len(peaks) > 0:
        features["spike_count"] = len(peaks)
        features["spike_amplitude"] = np.max(np.abs(raw_signal_1d[peaks]))
        features["spike_width"] = 2.0 # typically around 2ms, simplified here
        
    return features
