# CardioVision AI - Complete Technical Documentation

## 1. Project Overview

**Purpose**: CardioVision is an advanced, GPU-accelerated Medical AI system designed for the multi-label classification of 12 distinct cardiac disorders from raw 12-lead Electrocardiogram (ECG) signals. It bridges the gap between black-box machine learning and clinical cardiology by fusing XGBoost predictions with strict electrophysiological Rule Engines, SHAP explainability, and a Phi-3 backed Retrieval-Augmented Generation (RAG) system for automated clinical report generation.

**Architecture Workflow**:
```text
Raw ECG Input (.mat, .hea)
      ↓
Multiprocessed Dataset Scanner (CPU)
      ↓
GPU Signal Preprocessing (PyTorch Tensors)
      ↓
Vectorized Feature Extraction (190+ Features)
      ↓
Clinical Rule Engine (Batched Boolean Masks)
      ↓
Multi-Label XGBoost Classifier
      ↓
Isotonic Probability Calibration
      ↓
SHAP Explainer (Feature Importance)
      ↓
JSON Unified Payload Construction
      ↓
RAG + Phi-3 Clinical Report Generation
      ↓
FastAPI Backend
      ↓
React Interactive UI
```

**Technologies**:
- **Core ML**: XGBoost, Scikit-Learn, SHAP
- **Signal Processing**: PyTorch (GPU Batched), NeuroKit2, WFDB, SciPy
- **Backend**: FastAPI, Uvicorn
- **Frontend**: React (Vite)
- **LLM / RAG**: ChromaDB, Phi-3
- **Concurrency**: `concurrent.futures`, PyTorch DataLoaders

---

## 2. Dataset Pipeline

The project aggregates four primary databases: **PTB-XL**, **Chapman-Shaoxing**, **Georgia (CinC)**, and the **PhysioNet Challenge**.

**Workflow**:
- **Discovery**: A CPU-based `ProcessPoolExecutor` rapidly scans directories for `.hea` files.
- **Metadata Mapping**: Parses `ptbxl_database.csv` (using `ast.literal_eval` for SCP codes) and raw `.hea` diagnosis strings.
- **Exclusion Logic**: Immediately drops records containing confounding pathologies (e.g., STEMI, NSTEMI, PVCs).
- **Multiple Diagnoses**: Maps records to one or more of the 12 target classes. If a record has "NSR + RBBB", it is duplicated into both target directories.
- **Copying**: Uses a multithreaded Consumer thread-pool to copy `.mat`, `.dat`, and `.hea` files into an `Extracted/` directory without overwriting files by prepending the dataset name.

```python
# snippet from dataset_scanner.py
def map_diagnoses(raw_dx_list):
    mapped = set()
    for raw in raw_dx_list:
        clean_raw = str(raw).lower().strip()
        if any(ex in clean_raw for ex in EXCLUSIONS): continue
        for key, target in DIAGNOSIS_MAP.items():
            if key in clean_raw: mapped.add(target)
    return list(mapped)
```

---

## 3. ECG Signal Reading

The pipeline relies on the `wfdb` library to read header and signal data.

- **Format**: `wfdb.rdsamp()` reads the binary matrix.
- **Dimensions**: Signals are transposed to shape `(Num_Leads, Signal_Length)` which corresponds to `(12, L)`.
- **Normalization**: Records are truncated or zero-padded dynamically to exactly **10 seconds (5,000 points)** at a **500Hz** sampling rate to ensure fixed-size PyTorch tensors for batching.

```python
# snippet from gpu_processor.py
signals, fields = wfdb.rdsamp(record["base_path"])
signals = signals.T 
if signals.shape[1] > 5000:
    signals = signals[:, :5000]
elif signals.shape[1] < 5000:
    pad_width = 5000 - signals.shape[1]
    signals = np.pad(signals, ((0, 0), (0, pad_width)), mode='constant')
tensor = torch.tensor(signals, dtype=torch.float32)
```

---

## 4. ECG Preprocessing

The preprocessing pipeline is strictly optimized for GPUs using PyTorch, eschewing slow CPU-bound loops.

1. **Precision**: Casts to `torch.float16` for VRAM efficiency.
2. **Baseline Wander Correction**: Subtracts the mean of each lead independently across the batch matrix.
3. **Bandpass Filtering**: Utilizes Fast Fourier Transforms (`torch.fft.rfft`) to mute frequencies outside the physiological range (e.g., dropping <1Hz to remove respiratory wander, dropping >40Hz to remove muscle artifact).

---

## 5. R Peak Detection & Frequency Mapping

Rather than relying on iterative CPU delineators (like Pan-Tompkins) for mass extraction, the GPU pipeline leverages Spectral Density to identify heart rate and QRS boundaries.

- **HR via FFT**: Isolates the 1-3 Hz band (60-180 BPM), finds the dominant frequency bin using `torch.argmax(power)`, and converts to BPM.
- **NeuroKit2 Fallback**: In the sequential pipeline, `nk.ecg_peaks` and `nk.ecg_delineate` are used to find exact P, Q, R, S, T indices.

---

## 6. Feature Extraction

Over 190 tabular features are extracted per record to feed the XGBoost model.

- **Heart Rate (HR)**: Extracted via Spectral Dominant Frequency or RR intervals. Critical for Tachycardia.
- **QRS Duration**: Estimated via High-Frequency / Low-Frequency energy ratios on GPU, or exact temporal mapping via Wavelet transforms. Crucial for Bundle Branch Blocks.
- **PR Interval**: Measures AV node conduction delay. Short PR (<120ms) is the hallmark of WPW.
- **Variance / Energy**: Lead-specific voltage amplitude (`torch.var`, `torch.max`).
- **Pacemaker Spikes**: Derived via high-pass filtering (>15Hz) to isolate rapid, near-instantaneous voltage jumps before standard smoothing destroys the spike.

---

## 7. Clinical Validation (Rule Engine)

Clinical rule validation is the safeguard against noisy datasets. It is applied *before* model training (to clean data) and *after* prediction (to explain predictions).

Implemented using vectorized Boolean PyTorch/NumPy masks:
- **NSR**: Requires `50 <= HR <= 110` and presence of P-waves.
- **Tachycardia**: Requires `HR >= 95`.
- **RBBB/LBBB**: Requires widened QRS `> 110ms`.

```python
# snippet from batched_verifier.py
if target_disorder == "Normal_Sinus_Rhythm":
    passed &= (hr >= 50) & (hr <= 110)
elif target_disorder in ["RBBB", "LBBB"]:
    passed &= (qrs > 110)
```

---

## 8. Feature Selection

Feature selection happens intrinsically via the XGBoost algorithm, which penalizes uninformative features during tree construction via regularization (`alpha`, `lambda`). Low importance features (evaluated later via SHAP) are dropped dynamically across different `max_depth` branches.

---

## 9. Model Training

- **Algorithm**: `MultiLabelXGBoost` utilizing the `binary:logistic` objective for each of the 12 outputs.
- **Imbalance**: Uses dynamically calculated `scale_pos_weight` arrays to handle severe minority classes (like WPW).
- **Calibration**: Applies `IsotonicCalibrator` via Scikit-Learn. Tree models output uncalibrated margins; Isotonic Regression warps these outputs on a held-out 15% validation set so that an 85% score genuinely represents an 85% statistical probability.

```python
# snippet from train.py
model = MultiLabelXGBoost()
model.fit(X_train, y_train)

calibrator = IsotonicCalibrator(model)
calibrator.fit(X_calib, y_calib)
```

---

## 10. Training Workflow

```text
dataset.parquet 
      ↓
train_test_split (70% Train, 15% Calib, 15% Test)
      ↓
MultiLabelXGBoost.fit(X_train, y_train)
      ↓
IsotonicCalibrator.fit(X_calib, y_calib)
      ↓
model_evaluator(y_test, y_pred_calib)
      ↓
SHAPExplainer.fit(X_train)
      ↓
Save .pkl artifacts to /outputs/models/
```

---

## 11. Threshold Optimization

By default, binary classifiers use a 0.5 threshold. However, for severe pathologies (like VFib), high sensitivity (Recall) is favored.
- **ROC Analysis**: Computes the Receiver Operating Characteristic curve.
- **Youden's J Statistic**: `Sensitivity + Specificity - 1` is maximized to find the optimal per-class probability threshold, minimizing False Negatives for lethal rhythms.

---

## 12. Model Evaluation

Evaluated comprehensively using:
- **ROC AUC**: Area under the curve for each of the 12 diseases.
- **F1 Score**: Harmonic mean of Precision and Recall.
- **Confusion Matrix**: Tracked heavily during cross-dataset generalization (e.g. training on PTB-XL, testing on Georgia) to identify hardware-specific bias.

---

## 13. SHAP Explainability

SHAP (SHapley Additive exPlanations) opens the black box.
- **TreeExplainer**: Optimized specifically for XGBoost.
- **Local Explanations**: For a single patient, SHAP returns an array of floating point weights showing exactly which features pushed the prediction up or down (e.g., `Lead_V1_R_Amp: +2.4`).
- **Integration**: Sent directly to the frontend for Heatmap rendering.

---

## 14. Clinical Rule Engine Implementation

Parallel to SHAP, explicit Boolean rules are evaluated on the incoming patient features to construct text explanations:

- **WPW**: `if PR_Interval < 120 and QRS > 110: return "Pre-excitation indicated by shortened PR and widened QRS delta wave."`
- **LAE**: `if P_Dur_II > 120 and P_Area_V1 < 0: return "Left Atrial Enlargement indicated by prolonged P-wave."`
- **Pacemaker**: `if Spike_Count > 0: return "Pacemaker rhythm detected due to high-frequency spikes."`

---

## 15. Prediction Pipeline (End-to-End User Upload)

1. **Upload**: User uploads `ecg.mat` via React UI.
2. **FastAPI**: Endpoint receives file, triggers `predict.py`.
3. **Extraction**: GPU pipeline standardizes to 500Hz/5000pts and extracts 190 tabular features.
4. **Prediction**: XGBoost generates 12 raw probabilities.
5. **Calibration**: Isotonic Calibrator adjusts probabilities to clinical confidence percentages.
6. **XAI**: SHAP computes feature impacts; Rule Engine checks morphological thresholds.
7. **RAG**: Features + SHAP are sent to Phi-3 + ChromaDB to write a plain-English doctor's note.
8. **JSON Response**: Unified payload returned to React.

---

## 16. Backend API

Built using `FastAPI`.
- **Endpoint**: `POST /api/predict`
- **Input**: `multipart/form-data` containing `.mat` and `.hea` files.
- **Output**: Strict JSON schema.

---

## 17. Frontend Integration

Built using **React** and **Vite**.
- **Charts**: Interactive 12-lead plotting.
- **Heatmaps**: Uses SHAP weights to colorize specific leads/segments (e.g. Lead V1 turns red if it contributed heavily to an RBBB prediction).
- **Report Display**: Renders the Markdown generated by Phi-3 alongside clinical confidence gauges.

---

## 18. RAG Pipeline

Retrieval-Augmented Generation ensures the LLM doesn't hallucinate.
- **Knowledge Base**: Textbooks and ACC/AHA guidelines chunked and embedded into **ChromaDB**.
- **Retriever**: Queries ChromaDB using the predicted disorder (e.g. "RBBB criteria").
- **Context Generation**: Injects the retrieved medical facts directly into the Phi-3 prompt.

---

## 19. Phi-3 Integration

Phi-3 (a lightweight LLM) acts as the reporting cardiologist.
- **Prompt Building**: Combines the Patient Features, Model Confidence, SHAP values, Rule Engine triggers, and RAG context.
- **Output**: Generates a strictly formatted, empathetic, and clinically accurate summary report explaining the diagnosis.

---

## 20. Output JSON Example

```json
{
  "prediction": {
    "disorder": "RBBB",
    "confidence_calibrated": 0.94
  },
  "clinical_rules_matched": [
    "QRS Duration > 120ms",
    "RSR' pattern in V1"
  ],
  "top_shap_features": [
    {"feature": "QRS_Duration", "importance_weight": 4.2},
    {"feature": "Lead_V1_R_Amplitude", "importance_weight": 2.8}
  ],
  "ecg_measurements": {
    "Heart_Rate_bpm": 76,
    "PR_Interval_ms": 155,
    "QRS_Duration_ms": 142
  },
  "llm_report": "The AI model detected a Right Bundle Branch Block with 94% confidence, primarily driven by a widened QRS duration of 142ms..."
}
```

---

## 21. Important Source Files

| File | Purpose | Important Functions |
| ---- | ------- | ------------------- |
| `gpu_dataset_pipeline/orchestrator.py` | Runs GPU Producer-Consumer pipeline | `run_pipeline`, `render_progress` |
| `gpu_dataset_pipeline/gpu_processor.py` | PyTorch Tensor preprocessing | `process_signal_batch` |
| `gpu_dataset_pipeline/batched_verifier.py`| Vectorized clinical rules | `verify_batch_rules` |
| `train.py` | XGBoost training and calibration | `MultiLabelXGBoost.fit()` |
| `backend/app.py` | FastAPI application endpoints | `predict_ecg()` |

---

## 22. Important Classes

- **`BatchedECGDataset`** (`gpu_processor.py`): Subclasses `torch.utils.data.Dataset`. Reads raw WFDB files and returns standardized `(12, 5000)` float32 tensors.
- **`MultiLabelXGBoost`** (`xgboost_classifier.py`): Wraps the XGBoost library to handle 12 discrete binary classification targets simultaneously using matrix scaling.
- **`IsotonicCalibrator`** (`calibration.py`): Wraps Scikit-Learn's `IsotonicRegression` to warp ML margins into true probabilities.
- **`SHAPExplainer`** (`shap_explainer.py`): Initializes `shap.TreeExplainer` upon training and serves `.shap_values()` during inference.

---

## 23. End-to-End Workflow Summary

CardioVision represents a state-of-the-art hybrid AI architecture. It begins by aggressively aggregating tens of thousands of ECG records using a highly-optimized PyTorch Producer-Consumer pipeline, converting chaotic disk I/O into rapid, batched tensor matrix operations on the GPU. During this phase, strict morphologic Rule Engines mathematically reject noisy or false-positive datasets (e.g. rejecting Tachycardia labels if the HR tensor is <95). 

This pristine tabular feature data is fed into a Multi-Label XGBoost classifier. Because tree-based models lack inherent probability accuracy, Isotonic Calibration is applied to guarantee the outputs reflect true clinical confidence. 

At inference time, a patient uploads a `.mat` file via the React UI. FastAPI routes this to the preprocessing backend, extracting 190+ features and querying the calibrated XGBoost model. The model's decision is instantly audited by two systems: SHAP (which calculates exact mathematical feature impacts) and the Clinical Rule Engine (which verifies that physiological laws were met). These numeric justifications are pushed through a ChromaDB RAG retrieval system into a Phi-3 LLM, which formats the complex arrays into a coherent, human-readable medical report. Finally, the React UI renders this JSON payload, coloring the ECG waveform using the SHAP heatmaps and presenting the cardiologist with a perfectly transparent, rule-verified diagnosis.
