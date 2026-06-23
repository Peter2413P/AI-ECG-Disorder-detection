# ECG Feature Extraction Pipeline Architecture

This document details the end-to-end preprocessing, delineation, and feature extraction pipeline required to transform raw 12-lead ECG signals into a structured tabular dataset. This tabular structure is optimized for tree-based machine learning models (XGBoost, LightGBM) and robust SHAP explainability.

---

## PHASE 1: ECG PREPROCESSING

A rigorous preprocessing pipeline is critical because tabular features extracted from noisy signals will result in garbage-in, garbage-out (GIGO).

| Step | Purpose | Affects | Implementation & Parameters |
| :--- | :--- | :--- | :--- |
| **1. Baseline Wander Removal** | Removes low-frequency drifts caused by patient respiration or movement. | ST-segment elevation/depression, P-wave detection. | **NeuroKit2**: High-pass filter (Butterworth, order=5, cutoff=0.5 Hz). |
| **2. Powerline Interference** | Removes 50Hz/60Hz noise from electrical mains. | T-wave symmetry, QRS morphology. | **SciPy**: Notch filter (Q=30) at 50Hz and 60Hz. |
| **3. High-Frequency Noise Filtering** | Removes muscle artifacts (EMG noise). | QRS delineation, Entropy metrics. | **NeuroKit2**: Low-pass filter (Butterworth, order=5, cutoff=40 Hz). |
| **4. Signal Normalization** | Ensures all signals have a common scale regardless of the hardware. | Amplitudes (P, QRS, T). | **NumPy**: Z-score normalization (`(x - mean) / std`) or Min-Max scaling per lead. |
| **5. Lead Consistency Checking** | Validates the array shape and ordering. | Feature Mapping to specific regions (e.g., V1 vs V6). | **NumPy**: Shape assertion `(12, N_samples)`, ensuring standard order (I, II, III, aVR, aVL, aVF, V1-V6). |
| **6. Signal Quality Assessment** | Drops/Flags unreadable ECGs before extraction. | Prevents model collapse on noise. | **NeuroKit2**: `nk.ecg_quality()` using the zhao2018 algorithm. Drop if Quality Index < 0.5. |

**Recommended Library Stack**: `NeuroKit2` for high-level ECG cleaning (`nk.ecg_clean`), `SciPy` for custom filters, and `NumPy` for matrix manipulation.

---

## PHASE 2: ECG DELINEATION

Delineation is the process of identifying the exact boundaries (onset and offset) of the P, QRS, and T waves. This step is mandatory to compute clinical durations (like the PR interval).

1. **R-Peak Detection**:
   - **Algorithm**: Pan-Tompkins algorithm or Modified Christov algorithm.
   - **Accuracy**: Extremely high (>99% on clean data).

2. **Wave Delineation (Onsets/Offsets)**:
   - **Algorithm**: Discrete Wavelet Transform (DWT) or NeuroKit2's modified CWT-based delineation (`nk.ecg_delineate`).
   - **Accuracy**: Moderate to High. P-wave onsets are notoriously difficult to detect in noisy signals. 
   - **Implementation limitation**: If a delineation algorithm fails to find a T-wave offset, that feature must be returned as `NaN`.

---

## PHASE 3: FEATURE EXTRACTION DESIGN

Once delineated, clinical features are extracted per lead and globally.

### A. Rhythm Features (Global across Lead II/V1)
- **Heart Rate**: Beats per minute. Formula: `60 / Mean(RR)`. Importance: Detects Bradycardia/Tachycardia.
- **Mean / Median RR**: Average duration between R-peaks (seconds). Importance: Baseline rhythm tracking.
- **RR Standard Deviation (SDRR)**: Variation in RR intervals. Importance: Detects Arrhythmia / AFib.
- **RMSSD**: Root mean square of successive differences between normal heartbeats. Importance: Short-term HRV.
- **pNN50**: Percentage of successive RR intervals >50ms. Importance: Parasympathetic activity.

### B. Conduction Features (Extracted usually from Lead II or globally)
- **PR Interval**: Time from P onset to QRS onset. Importance: WPW (<120ms) or AV Blocks (>200ms).
- **QRS Duration**: Time from QRS onset to QRS offset. Importance: Identifies LBBB / RBBB / IVCD (>120ms).
- **QT Interval**: Time from QRS onset to T offset. Importance: Repolarization abnormalities.
- **QTc (Corrected)**: Bazett’s formula: `QT / sqrt(RR)`. Importance: Standardized QT assessment.

### C. P-Wave Features
- **P Duration**: P onset to P offset. Importance: Left Atrial Enlargement (>120ms).
- **P Amplitude**: Max voltage in P wave. Importance: Right Atrial Enlargement (>2.5mm).
- **P Area**: Integral of P wave. Importance: Overall atrial depolarization energy.

### D. QRS Features
- **R Amplitude**: Max positive voltage of QRS. Importance: Hypertrophy.
- **S Amplitude**: Max negative voltage after R peak. Importance: Bundle branch blocks.
- **R/S Ratio**: R amplitude divided by S amplitude. Importance: Axis shifts, RVH.

### E. ST Segment Features
- **ST Elevation / Depression**: Voltage level exactly 60ms after the J-point relative to baseline. Importance: Myocardial Infarction / Ischemia.
- **ST Slope**: Rate of change of the ST segment. Importance: Differentiating benign early repolarization from true ischemia.

### F. T Wave Features
- **T Amplitude**: Max voltage of T wave. Importance: Hyperkalemia, Ischemia.
- **T Symmetry**: Skewness of the T wave. Importance: Ischemic detection (symmetric inverted T-waves).

### G. Axis Features
- **QRS Axis**: Calculated via the net amplitudes in leads I and aVF. Formula: `arctan(Net_aVF / Net_I)`. Importance: Fascicular blocks, ventricular hypertrophy.

### H. Entropy & I. Frequency Domain Features
- **Shannon/Sample Entropy**: Measures complexity/unpredictability of the signal. Importance: Very high entropy indicates VFib or AFib.
- **Dominant Frequency**: Found via FFT peak. Importance: Flutter waves (~250-350 bpm).

---

## PHASE 4: DISORDER-FEATURE MAPPING

| Target Disorder | Primary Features | Secondary Features | Important Leads |
| :--- | :--- | :--- | :--- |
| **Normal Sinus Rhythm** | HR (60-100), Regular RR | Normal P-wave duration | II, V1 |
| **Sinus Tachycardia** | HR (> 100) | Shortened RR intervals | II |
| **Sinus Arrhythmia** | SDRR (High), RMSSD (High) | Normal P-wave / QRS | II |
| **PAC** | RR variance (Premature beat) | Abnormal P morphology | II, V1 |
| **RBBB** | QRS Duration (> 120ms) | RSR' pattern, Wide S wave | V1, V2, I, V6 |
| **LBBB** | QRS Duration (> 120ms) | Broad/notched R wave, Absent Q | I, aVL, V5, V6 |
| **IVCD** | QRS Duration (110 - 120ms) | Overall QRS widening | All Leads |
| **Delta Wave (WPW)** | PR Interval (< 120ms) | Initial QRS slurring (Delta) | II, V1-V6 |
| **Persistent ST Elevation** | ST Elevation > 1mm | T wave inversion | Leads mapping to injury |
| **Left Atrial Enlargement** | P Duration (> 120ms) | Biphasic P wave (negative terminal) | II, V1 |
| **Ventricular Fibrillation**| Very high Sample Entropy | Absent P and QRS | All Leads |
| **Pacemaker Rhythm** | High-frequency pre-QRS spikes | Wide QRS | All Leads |

---

## PHASE 5: FEATURE VECTOR DESIGN

The final tabular feature vector for a single ECG record will concatenate global metrics and lead-specific metrics.

**Structure**:
```python
[
  # Global Rhythm (10 features)
  HR, RR_mean, RR_std, RMSSD, pNN50, QRS_axis, Overall_Entropy, ...
  
  # Lead I (15 features)
  Lead_I_PR_interval, Lead_I_QRS_duration, Lead_I_P_amp, Lead_I_ST_dev, ...
  
  # Lead II (15 features)
  Lead_II_PR_interval, Lead_II_QRS_duration, ...
  
  # ... Repeated for all 12 Leads
]
```

**Estimations**:
- **Total Features**: ~10 Global + (15 metrics × 12 leads) = **~190 features per ECG**.
- **Dimensionality**: `N_samples x 190`. 
- **Storage**: Highly efficient. A dataset of 100,000 ECGs stored in a CSV/Parquet file of shape `(100000, 190)` requires less than 200 MB, vastly reducing I/O bottlenecks compared to deep learning on raw waveforms.

---

## PHASE 6: DATASET GENERATION WORKFLOW

The final output generated by the preprocessing script will be a wide tabular dataset (CSV or Parquet).

### Table Schema
| ECG_ID | Patient_ID | HR | RR_std | Lead_V1_QRS_dur | ... | NSR | PAC | RBBB | ... |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| P_001 | 991 | 75 | 0.05 | 95ms | ... | 1 | 0 | 0 | ... |
| P_002 | 992 | 110 | 0.02 | 160ms | ... | 0 | 0 | 1 | ... |

### Handling Edge Cases
- **Missing Values**: If a lead is too noisy and delineation fails (e.g., T-wave undetected), the feature extractor returns `np.nan` for `Lead_X_T_amp`. XGBoost/LightGBM handles NaNs natively during training, learning that a missing T-wave is itself informative.
- **Noisy ECGs**: If global `Signal_Quality < 0.2`, the row is discarded before feature extraction, ensuring model purity.

---

## PHASE 7: EXPLAINABLE AI READINESS (SHAP)

Because the model trains on explicitly defined clinical features, we achieve 1:1 parity with clinical guidelines via SHAP.

### Example Explanation Strategies

**1. RBBB Explanation**
- **Prediction**: Right Bundle Branch Block (89% Confidence)
- **Top SHAP Features**: `Lead_V1_QRS_duration`, `Lead_V1_R_amplitude`, `Lead_I_S_amplitude`.
- **Template**: "RBBB detected due to prolonged QRS duration (>120ms) combined with high terminal R-wave amplitude in Lead V1 (RSR' pattern)."

**2. Left Atrial Enlargement (LAE) Explanation**
- **Prediction**: LAE (75% Confidence)
- **Top SHAP Features**: `Lead_II_P_duration`, `Lead_V1_P_area`.
- **Template**: "LAE detected primarily due to prolonged P-wave duration in Lead II and abnormal P-wave area in V1."

**3. Ventricular Fibrillation Explanation**
- **Prediction**: VFib (99% Confidence)
- **Top SHAP Features**: `Global_Sample_Entropy`, `Lead_II_RR_std`.
- **Template**: "Life-threatening arrhythmia detected due to extreme signal entropy and lack of organized RR intervals."

By mapping SHAP values back to these specific column names, the frontend application can present the clinician with the exact physiological reason for the AI's diagnosis, satisfying regulatory (FDA/MDR) requirements for transparent medical AI.
