# dataset_pipeline/visualizer.py

import os
import numpy as np
import matplotlib.pyplot as plt

def generate_ecg_preview(signals, fs, disorder, quality_score, output_path, title):
    """
    Generates a high-quality 12-lead standard clinical ECG plot.
    """
    if signals is None or len(signals) == 0:
        return
        
    num_samples = signals.shape[0]
    num_leads = signals.shape[1]
    
    # We assume standard 12-lead shape. If it's single lead or less than 12, plot what we have.
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    plot_leads = min(num_leads, 12)
    
    fig, axes = plt.subplots(plot_leads, 1, figsize=(15, 2 * plot_leads), sharex=True)
    if plot_leads == 1:
        axes = [axes]
        
    time = np.arange(num_samples) / fs
    
    for i in range(plot_leads):
        ax = axes[i]
        
        # Plot signal
        ax.plot(time, signals[:, i], color='black', linewidth=1.2)
        
        # Clinical red grid
        ax.minorticks_on()
        ax.grid(which='major', color='#ff9999', linestyle='-', linewidth=0.8)
        ax.grid(which='minor', color='#ffcccc', linestyle='-', linewidth=0.3)
        
        # Labels
        lead_name = leads[i] if num_leads == 12 else f"Lead {i}"
        ax.set_ylabel(lead_name, fontsize=12, fontweight='bold', rotation=0, labelpad=20)
        
        # Hide top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    # Bottom axis label
    axes[-1].set_xlabel('Time (seconds)', fontsize=12)
    
    # Super title
    fig.suptitle(f"Disorder: {disorder.replace('_', ' ')}\nQuality Score: {quality_score:.1f}/100", 
                 fontsize=16, fontweight='bold', y=0.98)
                 
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
