import matplotlib.pyplot as plt
import numpy as np

def plot_raw_ecg(signal, fs, lead_names=None, title="Raw ECG"):
    """
    Plot raw 12-lead ECG signals.
    """
    if lead_names is None:
        from core.config import LEAD_NAMES
        lead_names = LEAD_NAMES
        
    num_leads = signal.shape[0]
    time = np.arange(signal.shape[1]) / fs
    
    fig, axes = plt.subplots(num_leads, 1, figsize=(15, 2 * num_leads), sharex=True)
    if num_leads == 1:
        axes = [axes]
        
    for i in range(num_leads):
        axes[i].plot(time, signal[i])
        axes[i].set_ylabel(lead_names[i])
        axes[i].grid(True)
        
    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_cleaned_ecg(raw_signal, cleaned_signal, fs, lead_idx=1, title="Cleaned vs Raw ECG"):
    """
    Plot cleaned vs raw signal for a specific lead (default Lead II).
    """
    time = np.arange(raw_signal.shape[1]) / fs
    plt.figure(figsize=(15, 5))
    plt.plot(time, raw_signal[lead_idx], label='Raw', alpha=0.5)
    plt.plot(time, cleaned_signal[lead_idx], label='Cleaned', linewidth=1.5)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_r_peaks(cleaned_signal, r_peaks, fs, lead_idx=1, title="R-Peaks Detection"):
    """
    Plot R-peaks over the cleaned signal.
    """
    time = np.arange(cleaned_signal.shape[1]) / fs
    plt.figure(figsize=(15, 5))
    plt.plot(time, cleaned_signal[lead_idx])
    plt.scatter(r_peaks / fs, cleaned_signal[lead_idx][r_peaks], color='red', zorder=5, label='R-Peaks')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_delineation(cleaned_signal, boundaries, fs, lead_idx=1, title="Wave Delineation"):
    """
    Plot boundaries (P onset/offset, QRS onset/offset, T onset/offset).
    boundaries: dict of lists with indices
    """
    time = np.arange(cleaned_signal.shape[1]) / fs
    plt.figure(figsize=(15, 5))
    plt.plot(time, cleaned_signal[lead_idx], color='black', alpha=0.7)
    
    colors = {
        'P_Onsets': 'green', 'P_Offsets': 'lightgreen',
        'QRS_Onsets': 'red', 'QRS_Offsets': 'darkred',
        'T_Onsets': 'blue', 'T_Offsets': 'lightblue'
    }
    
    for key, color in colors.items():
        if key in boundaries:
            valid_indices = [idx for idx in boundaries[key] if not np.isnan(idx)]
            plt.scatter(np.array(valid_indices) / fs, cleaned_signal[lead_idx][valid_indices], color=color, label=key, zorder=5)
            
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_st_segment(cleaned_signal, qrs_offsets, t_onsets, fs, lead_idx=1, title="ST Segment"):
    """
    Highlight ST segments between QRS offset and T onset.
    """
    time = np.arange(cleaned_signal.shape[1]) / fs
    plt.figure(figsize=(15, 5))
    plt.plot(time, cleaned_signal[lead_idx], color='gray', alpha=0.5)
    
    for q_off, t_on in zip(qrs_offsets, t_onsets):
        if not np.isnan(q_off) and not np.isnan(t_on) and t_on > q_off:
            q_off, t_on = int(q_off), int(t_on)
            plt.plot(time[q_off:t_on], cleaned_signal[lead_idx][q_off:t_on], color='red', linewidth=2)
            
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.grid(True)
    plt.show()
