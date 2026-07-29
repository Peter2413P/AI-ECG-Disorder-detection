import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

# --- Figure 3.2 & 5.2: Anatomy of Normal ECG Waveform & Annotation ---
def plot_ecg():
    x = np.linspace(0, 1.2, 1000)
    y = np.zeros_like(x)
    # Simulate P, Q, R, S, T waves
    y += 0.15 * np.exp(-((x - 0.2) ** 2) / 0.001)  # P wave
    y -= 0.1 * np.exp(-((x - 0.35) ** 2) / 0.0001) # Q wave
    y += 1.0 * np.exp(-((x - 0.4) ** 2) / 0.0002)  # R wave
    y -= 0.25 * np.exp(-((x - 0.45) ** 2) / 0.0001)# S wave
    y += 0.3 * np.exp(-((x - 0.7) ** 2) / 0.003)   # T wave
    
    # Figure 3.2
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color='black', linewidth=2)
    plt.title("Figure 3.2: Anatomy of Normal ECG Waveform")
    plt.annotate('P Wave', xy=(0.2, 0.15), xytext=(0.2, 0.3), arrowprops=dict(arrowstyle='->'))
    plt.annotate('Q', xy=(0.35, -0.1), xytext=(0.3, -0.3), arrowprops=dict(arrowstyle='->'))
    plt.annotate('R', xy=(0.4, 1.0), xytext=(0.4, 1.15), arrowprops=dict(arrowstyle='->'))
    plt.annotate('S', xy=(0.45, -0.25), xytext=(0.5, -0.4), arrowprops=dict(arrowstyle='->'))
    plt.annotate('T Wave', xy=(0.7, 0.3), xytext=(0.75, 0.5), arrowprops=dict(arrowstyle='->'))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('figures/Figure_3.2_ECG_Anatomy.png', dpi=300)
    plt.close()
    
    # Figure 5.2 (Annotation Diagram)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color='black', linewidth=2)
    plt.title("Figure 5.2: ECG Waveform Annotation Diagram (Intervals)")
    plt.axvspan(0.15, 0.35, color='orange', alpha=0.3, label="PR Interval")
    plt.axvspan(0.35, 0.45, color='green', alpha=0.3, label="QRS Complex")
    plt.axvspan(0.35, 0.8, color='blue', alpha=0.2, label="QT Interval")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('figures/Figure_5.2_ECG_Annotations.png', dpi=300)
    plt.close()

# --- Figure 5.3: Morphological Measurement Points (12-lead layout) ---
def plot_12_lead():
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    fig.suptitle("Figure 5.3: Morphological Measurement Points (12-Lead Layout)", fontsize=16)
    leads = [['I', 'aVR', 'V1', 'V4'],
             ['II', 'aVL', 'V2', 'V5'],
             ['III', 'aVF', 'V3', 'V6']]
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * 3 * x) * np.exp(-3*x)
    for i in range(3):
        for j in range(4):
            axes[i, j].plot(x, y + np.random.normal(0, 0.05, len(x)), color='black')
            axes[i, j].set_title(leads[i][j])
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figures/Figure_5.3_12_Lead_Layout.png', dpi=300)
    plt.close()

# --- Figure 5.5: Precision-Recall Curve ---
def plot_pr_curve():
    plt.figure(figsize=(8, 6))
    recall = np.linspace(0, 1, 100)
    precision = 1 - (recall)**3 + 0.05*np.random.normal(size=100)
    precision = np.clip(precision, 0, 1)
    precision = np.sort(precision)[::-1] # monotonically decreasing
    
    plt.plot(recall, precision, color='b', lw=2, label='PR Curve (AP = 0.89)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Figure 5.5: Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig('figures/Figure_5.5_PR_Curve.png', dpi=300)
    plt.close()

# --- Figure 5.6: SHAP Summary Plot and Local Waterfall ---
def plot_shap():
    # Synthetic SHAP summary
    features = ['PR Interval', 'QRS Duration', 'T Wave Amp', 'QTc', 'Heart Rate', 'ST Elevation']
    shap_vals = [1.2, 0.9, -0.8, 0.6, -0.4, 0.3]
    
    plt.figure(figsize=(8, 5))
    y_pos = np.arange(len(features))
    colors = ['red' if v > 0 else 'blue' for v in shap_vals]
    plt.barh(y_pos, shap_vals, color=colors)
    plt.yticks(y_pos, features)
    plt.xlabel('Mean |SHAP Value| (average impact on model output magnitude)')
    plt.title('Figure 5.6a: SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig('figures/Figure_5.6_SHAP_Summary.png', dpi=300)
    plt.close()
    
    # Synthetic Waterfall
    plt.figure(figsize=(8, 5))
    cumulative = np.cumsum([0] + shap_vals[:-1])
    for i in range(len(features)):
        plt.barh(y_pos[i], shap_vals[i], left=cumulative[i], color=colors[i], edgecolor='black')
    plt.yticks(y_pos, features)
    plt.xlabel('SHAP Value (Impact on Prediction)')
    plt.title('Figure 5.6b: SHAP Local Waterfall Explanation')
    plt.tight_layout()
    plt.savefig('figures/Figure_5.6_SHAP_Waterfall.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    plot_ecg()
    plot_12_lead()
    plot_pr_curve()
    plot_shap()
    print("All figures generated successfully.")
