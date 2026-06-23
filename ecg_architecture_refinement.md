# ECG Architecture Refinement & Validation Plan

This document refines the proposed 190-feature extraction pipeline, introducing robust validation layers, enhanced specific-disorder feature engineering (Pacemaker, VF), and a unified XAI architecture. The total feature count of ~190 is preserved and slightly enhanced with targeted metrics.

---

## PHASE 1 — DATASET SOURCE TRACKING

When aggregating PTB-XL, Chapman, Georgia, and PhysioNet datasets, tracking provenance is crucial to prevent the model from learning "dataset-specific" artifacts rather than clinical pathology.

### Design Strategy
1. **Source Encoding**: Include a strict categorical column `dataset_source` in the final tabular dataset (e.g., `['PTB-XL', 'Chapman', 'Georgia', 'PhysioNet_Other']`).
2. **Bias Detection Method**: 
   - After extracting the ~190 features, train a simple classifier to predict `dataset_source` from the features. 
   - If the model achieves high accuracy (>80%), it indicates the features contain strong dataset biases (e.g., all Chapman data has slightly lower T-wave amplitudes due to their specific machines).
3. **Evaluation Strategy**: 
   - Perform "Leave-One-Dataset-Out" (LODO) cross-validation. Train on PTB-XL + Chapman, and test on Georgia. This proves the model generalizes across hardware and demographics.

---

## PHASE 2 — FEATURE VALIDATION LAYER

An automated sanity check must run immediately after the 190 features are extracted and before any ML training begins.

### Automated Validation Output Example

| Feature | Mean | Std Dev | Min | Max | Missing % | Outliers (Count) | Clinical Sanity Check |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `QRS_Duration` | 98 ms | 20 ms | 40 ms| 350 ms| 0.5% | 120 (>160ms) | Valid: 60-120ms normal, up to 200ms in blocks. |
| `PR_Interval` | 160 ms| 35 ms | 0 ms | 600 ms| 2.1% | 45 (0ms errors)| Valid: 120-200ms normal. 0ms indicates extraction failure. |
| `HR` | 75 bpm| 18 bpm | 10 bpm| 300 bpm| 0.1% | 8 (Extreme Tachy)| Valid: 60-100 bpm normal. >250 usually noise. |

### Handling Extraction Failures
- If an algorithm cannot find a P-wave, `PR_Interval` must explicitly be set to `NaN` (not 0).
- If `Missing %` for a feature exceeds 15% across the dataset, the extraction algorithm for that specific wave must be tuned, or the feature dropped.

---

## PHASE 3 — WPW (DELTA WAVE) VERIFICATION

WPW (Delta Wave) is notoriously difficult to classify because noise in the PR segment can mimic a slurred upstroke. With 3,389 suspected WPW samples, manual verification of a subset is necessary.

### Verification Pipeline
1. **Sample Count Confirmation**: Group the dataset by exact original SCP/SNOMED codes (e.g., `WPW`, `74390002`). Ensure no unrelated pre-excitation syndromes were incorrectly mapped.
2. **Feature-Based Triage (Automated filtering)**:
   - Filter the 3,389 WPW samples where the extracted `PR_Interval` > 140ms. True WPW classically has PR < 120ms. If PR > 140ms, these are high-risk **False Positives**.
3. **Visual Validation Workflow**:
   - Randomly sample 100 "True Positives" (PR < 120ms) and 100 "High Risk" (PR > 140ms).
   - Plot Lead II and Lead V1 using `matplotlib`. Overlay the delineator's QRS-onset marker.
   - Visually confirm the *slurred upstroke* (Delta wave).

---

## PHASE 4 — PACEMAKER FEATURE ENHANCEMENT

Standard QRS delineation often fails on Pacemaker rhythms because the sharp pacemaker spike throws off Pan-Tompkins algorithms.

### Enhanced Features to Add (Included in the 190 total)
- `Spike_Count`: Number of high-frequency vertical spikes detected per 10 seconds.
- `Spike_Amplitude`: Voltage of the highest spike (usually > 2mV, much faster than QRS).
- `Spike_Width`: Duration of the spike (typically < 2 ms).
- `Spike_to_QRS_Interval`: Time from spike to ventricular depolarization.
- `Percent_Paced_Beats`: Ratio of paced beats to native beats.

### Extraction Strategy
Before standard low-pass filtering (which will erase the pacemaker spike), run a high-pass filter (>15 Hz) and a derivative operator to locate extreme, near-instantaneous voltage changes indicating the artificial spike.

---

## PHASE 5 — VF / FLUTTER FEATURE ENHANCEMENT

Ventricular Fibrillation/Flutter lacks distinct P-QRS-T complexes. Standard features (`PR_Interval`) will return `NaN`. We must rely on frequency and complexity metrics.

### Enhanced Features to Add
- **Rhythm Features**:
  - `RR_Entropy`: Sample entropy of RR intervals (Highest in VFib).
  - `RR_Irregularity_Index`: Coefficient of variation of RR intervals.
- **Frequency Features (FFT)**:
  - `Dominant_Frequency`: In Flutter, this peaks clearly at ~4-6 Hz (250-350 bpm).
  - `Spectral_Entropy`: Flat/broad spectrum in VFib vs. narrow peaks in normal rhythm.
  - `Band_Power_Ratio`: Ratio of power in 2-10 Hz band vs. 10-40 Hz band.

*These features will replace missing morphological features during extreme arrhythmias.*

---

## PHASE 6 — MULTI-LABEL OUTPUT DESIGN

An ECG can concurrently exhibit multiple pathologies (e.g., LBBB + AFib + PAC).

### Output Vector
`[NSR, Tachy, Arrhythmia, PAC, RBBB, LBBB, IVCD, WPW, STE, LAE, VF_Flutter, Pacemaker]`

### Architecture
- **Structure**: A single XGBoost or LightGBM model trained using `MultiOutputClassifier` or native multi-label objectives (`binary:logistic` applied to each column independently).
- **Class Imbalance Strategy**: Use `scale_pos_weight` dynamically calculated for each of the 12 columns to heavily penalize missing minority classes (like LBBB or VFib).
- **Target Matrix**: A binary matrix of shape `(N_samples, 12)` where `1` indicates presence.

---

## PHASE 7 — CLINICAL RULE ENGINE

The rule engine runs in parallel with the ML model. It provides explicit medical text templates when certain features cross established cardiological thresholds.

| Target Disorder | Feature Thresholds | Human-Readable Explanation Template |
| :--- | :--- | :--- |
| **RBBB** | `QRS_Dur` > 120ms AND `R_amp_V1` > `S_amp_V1` | "RBBB indicated by prolonged QRS duration ({QRS_Dur} ms) and an RSR' equivalent pattern in Lead V1." |
| **LBBB** | `QRS_Dur` > 120ms AND `S_amp_V1` high AND `Q_amp_V6` = 0 | "LBBB indicated by prolonged QRS duration ({QRS_Dur} ms) with a broad, deep S-wave in V1 and absent Q-waves in lateral leads." |
| **WPW** | `PR_Interval` < 120ms AND `QRS_Dur` > 110ms | "Pre-excitation (WPW) indicated by shortened PR interval ({PR_Interval} ms) and widened QRS complex suggestive of a delta wave." |
| **LAE** | `P_Dur_II` > 120ms AND `P_Area_V1` negative | "Left Atrial Enlargement indicated by prolonged P-wave duration ({P_Dur_II} ms) in Lead II and pronounced terminal negative P-wave in V1." |
| **Tachycardia**| `HR` > 100 bpm | "Sinus Tachycardia classified primarily due to elevated heart rate ({HR} bpm)." |
| **Pacemaker** | `Spike_Count` > 0 | "Pacemaker rhythm detected due to the presence of {Spike_Count} high-frequency artificial spikes preceding ventricular depolarization." |

---

## PHASE 8 — PROBABILITY CALIBRATION

Raw ML outputs (e.g., 0.85) from tree models are often poorly calibrated and do not represent true clinical probabilities. 

### Strategy
After training the Multi-Label XGBoost model, apply **Isotonic Regression** to calibrate the probabilities for each of the 12 classes independently using a held-out validation set.
- **Why Isotonic?**: We have enough data (>30,000 samples) to support non-parametric Isotonic Regression, which performs better than Platt Scaling (Sigmoid) on highly skewed tree outputs.
- **Benefit**: A calibrated 90% confidence score means that out of 100 ECGs given that score, exactly 90 truly have the disorder.

---

## PHASE 9 — PREDICTION EVIDENCE PACKAGE (JSON)

The final API output fuses the ML predictions, SHAP evidence, Rule Engine matches, and raw measurements into a single unified JSON payload.

```json
{
  "prediction": {
    "disorder": "RBBB",
    "confidence_calibrated": 0.94
  },
  "clinical_rules_matched": [
    "QRS Duration > 120ms",
    "R/S Ratio > 1 in V1"
  ],
  "top_shap_features": [
    {"feature": "QRS_Duration_Global", "importance_weight": +4.2},
    {"feature": "Lead_V1_R_Amplitude", "importance_weight": +2.8}
  ],
  "ecg_measurements": {
    "Heart_Rate_bpm": 76,
    "PR_Interval_ms": 155,
    "QRS_Duration_ms": 142,
    "QTc_Interval_ms": 410
  },
  "explanation_text": "RBBB indicated by prolonged QRS duration (142 ms) and an RSR' equivalent pattern in Lead V1."
}
```

---

## PHASE 10 — FINAL ARCHITECTURE REVIEW

### System Flow
`Raw ECG (.mat)` → `NeuroKit2 Preprocessing` → `DWT Delineation` → `~190 Tabular Feature Extraction` → `Automated Feature Validation Layer` → `Multi-Label XGBoost` → `Isotonic Calibration` → `SHAP + Rule Engine Fusion` → `Clinical JSON Payload`.

### Weaknesses & Risks Identified
1. **Delineation Failure on High Noise**: If baseline wander is extreme, the delineator will fail to find T-waves. **Mitigation**: The `Feature Validation Layer` will output `NaN`, and XGBoost will natively handle the missing split.
2. **WPW False Positives**: Noise simulating a delta wave. **Mitigation**: Relying on Phase 3 visual verification during dataset build.
3. **Pacemaker Signal Loss**: Standard low-pass filters destroy pacemaker spikes. **Mitigation**: Extract pacemaker spike features *before* low-pass smoothing (Phase 4).

### Final Recommendation
The architecture is now highly robust, mathematically sound, and ready for production code. The inclusion of the **Feature Validation Layer**, **Isotonic Calibration**, and the dual-pronged **SHAP + Clinical Rule Engine** ensures the system meets the highest standards for Explainable Medical AI.
