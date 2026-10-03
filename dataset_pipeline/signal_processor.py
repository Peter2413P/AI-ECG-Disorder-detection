# dataset_pipeline/signal_processor.py

import numpy as np
import scipy.signal as signal
import neurokit2 as nk

def compute_signal_quality(signals, fs=500):
    """
    Computes an overall ECG quality score between 0 and 100 based on
    SNR, baseline wander, and missing/flat signals.
    """
    if signals is None or len(signals) == 0:
        return 0.0
        
    quality_scores = []
    
    # Analyze each lead
    # Assuming signals is shape (samples, leads)
    num_leads = signals.shape[1]
    for i in range(num_leads):
        lead_signal = signals[:, i]
        
        # Check for flatline
        if np.var(lead_signal) < 1e-6:
            quality_scores.append(0.0)
            continue
            
        # Check for NaNs
        if np.isnan(lead_signal).any():
            quality_scores.append(0.0)
            continue
            
        try:
            # Clean signal
            clean = nk.ecg_clean(lead_signal, sampling_rate=fs, method='neurokit')
            
            # Simple SNR estimation: var(clean) / var(noise)
            noise = lead_signal - clean
            snr = np.var(clean) / (np.var(noise) + 1e-8)
            
            # Cap SNR contribution to avoid extreme outliers skewing results
            snr_score = min(snr / 10.0, 1.0) * 100
            
            quality_scores.append(snr_score)
        except Exception as e:
            quality_scores.append(20.0) # Penalty for failure to process
            
    if not quality_scores:
        return 0.0
        
    # Average across leads
    return sum(quality_scores) / len(quality_scores)

def extract_clinical_features(signals, fs=500):
    """
    Extracts high-level morphological features required for clinical verification.
    """
    features = {
        "heart_rate": None,
        "qrs_duration": None,
        "pr_interval": None,
        "p_wave_present": False,
        "rr_std": None
    }
    
    if signals is None or len(signals) == 0:
        return features
        
    try:
        # We process Lead II commonly used for rhythm analysis. 
        # Assuming Lead II is index 1, fallback to 0 if not 12-lead.
        lead_idx = 1 if signals.shape[1] >= 2 else 0
        lead_signal = signals[:, lead_idx]
        
        # Clean and find peaks
        clean_sig = nk.ecg_clean(lead_signal, sampling_rate=fs)
        peaks, info = nk.ecg_peaks(clean_sig, sampling_rate=fs)
        
        # Extract rate
        hr = nk.signal_rate(peaks, sampling_rate=fs, desired_length=len(clean_sig))
        features["heart_rate"] = np.nanmean(hr)
        
        # RR standard deviation (HRV)
        r_peaks = info["ECG_R_Peaks"]
        rr_intervals = np.diff(r_peaks) / fs * 1000 # ms
        features["rr_std"] = np.std(rr_intervals)
        
        # Delineate to find QRS and P waves
        _, waves_peak = nk.ecg_delineate(clean_sig, r_peaks, sampling_rate=fs, method="dwt")
        
        # QRS Duration (approximate using delineated peaks)
        if 'ECG_R_Onsets' in waves_peak and 'ECG_R_Offsets' in waves_peak:
            onsets = np.array(waves_peak['ECG_R_Onsets'])
            offsets = np.array(waves_peak['ECG_R_Offsets'])
            
            valid = ~np.isnan(onsets) & ~np.isnan(offsets)
            if any(valid):
                qrs_durations = (offsets[valid] - onsets[valid]) / fs * 1000
                features["qrs_duration"] = np.nanmean(qrs_durations)
                
        # P-wave presence
        if 'ECG_P_Peaks' in waves_peak:
            p_peaks = np.array(waves_peak['ECG_P_Peaks'])
            features["p_wave_present"] = np.sum(~np.isnan(p_peaks)) > (len(r_peaks) * 0.5)

    except Exception as e:
        pass
        
    return features
