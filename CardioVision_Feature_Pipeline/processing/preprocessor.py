import neurokit2 as nk
import scipy.signal as signal
import numpy as np
from core.logger import get_logger
from core.config import TARGET_FS, BASELINE_FILTER_CUTOFF, POWERLINE_FREQ_1, POWERLINE_FREQ_2, LOWPASS_CUTOFF

logger = get_logger("ECGPreprocessor")

class ECGPreprocessor:
    def __init__(self, target_fs=TARGET_FS):
        self.target_fs = target_fs

    def resample_signal(self, ecg_signal, original_fs):
        if original_fs == self.target_fs:
            return ecg_signal
            
        num_leads = ecg_signal.shape[0]
        num_samples = int(ecg_signal.shape[1] * self.target_fs / original_fs)
        resampled_signal = np.zeros((num_leads, num_samples))
        
        for i in range(num_leads):
            # Using scipy.signal.resample for fast resampling
            resampled_signal[i] = signal.resample(ecg_signal[i], num_samples)
            
        return resampled_signal

    def clean_signal(self, ecg_signal, fs):
        """
        Applies baseline wander removal, powerline notch filtering, and high-frequency noise removal.
        """
        num_leads = ecg_signal.shape[0]
        cleaned_signal = np.zeros_like(ecg_signal)
        
        for i in range(num_leads):
            lead_sig = ecg_signal[i]
            
            # Baseline wander removal (High-pass filter)
            b, a = signal.butter(5, BASELINE_FILTER_CUTOFF / (0.5 * fs), btype='highpass')
            lead_sig = signal.filtfilt(b, a, lead_sig)
            
            # Powerline interference (Notch filter)
            for freq in [POWERLINE_FREQ_1, POWERLINE_FREQ_2]:
                b_notch, a_notch = signal.iirnotch(freq, 30.0, fs)
                lead_sig = signal.filtfilt(b_notch, a_notch, lead_sig)
                
            # Low-pass filter for EMG noise
            b_low, a_low = signal.butter(5, LOWPASS_CUTOFF / (0.5 * fs), btype='lowpass')
            lead_sig = signal.filtfilt(b_low, a_low, lead_sig)
            
            cleaned_signal[i] = lead_sig
            
        return cleaned_signal

    def normalize_signal(self, ecg_signal):
        """
        Z-score normalization per lead.
        """
        num_leads = ecg_signal.shape[0]
        normalized_signal = np.zeros_like(ecg_signal)
        
        for i in range(num_leads):
            lead_sig = ecg_signal[i]
            std = np.std(lead_sig)
            if std > 0:
                normalized_signal[i] = (lead_sig - np.mean(lead_sig)) / std
            else:
                normalized_signal[i] = lead_sig
                
        return normalized_signal

    def process(self, ecg_record):
        """
        Main preprocessing pipeline.
        Updates ecg_record in place and returns cleaned signal.
        """
        raw_signal = ecg_record['signal']
        original_fs = ecg_record['fs']
        
        # Resample
        resampled = self.resample_signal(raw_signal, original_fs)
        ecg_record['fs'] = self.target_fs
        
        # Clean
        cleaned = self.clean_signal(resampled, self.target_fs)
        
        # Normalize
        normalized = self.normalize_signal(cleaned)
        
        ecg_record['cleaned_signal'] = normalized
        return normalized
