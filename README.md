# CardioVision Feature Pipeline & Unified ECG Dataset Audit

Welcome to the **CardioVision Feature Pipeline** repository. This project focuses on building a robust, explainable Machine Learning pipeline for multi-label classification of 12 distinct ECG abnormalities using a harmonized dataset from multiple public sources.

## Overview

The primary goal of this project is to develop a highly interpretable, unified feature extraction and classification system for electrocardiogram (ECG) data. The pipeline aggregates over 85,000 ECG records, extracts ~190 clinically relevant morphological and rhythm features, and classifies them into 12 target disorders with explainable outputs using SHAP and a Clinical Rule Engine.

## Datasets
The unified dataset is constructed from four major public sources:
- **PhysioNet** (43,101 records)
- **PTB-XL** (21,837 records)
- **Georgia** (10,344 records)
- **Chapman** (10,247 records)

**Total ECGs Processed**: 85,529

## Target Classes
We focus on a strict 12-class categorization of common to critical ECG pathologies:
1. Normal Sinus Rhythm (NSR)
2. Sinus Tachycardia
3. Sinus Arrhythmia
4. Premature Atrial Contraction (PAC)
5. Right Bundle Branch Block (RBBB)
6. Left Bundle Branch Block (LBBB)
7. Intraventricular Conduction Delay (IVCD)
8. Wolff-Parkinson-White (WPW)
9. Persistent ST Elevation
10. Left Atrial Enlargement (LAE)
11. Ventricular Fibrillation/Flutter (VF/Flutter)
12. Pacemaker Rhythm

## System Architecture

Our pipeline ensures transparency and clinically-aligned logic at every step:
1. **Preprocessing**: Data cleaning and baseline wander removal using `NeuroKit2`.
2. **Delineation**: P-QRS-T wave detection utilizing Discrete Wavelet Transforms (DWT).
3. **Feature Extraction**: Generation of ~190 tabular features covering wave amplitudes, durations, and intervals (including specific enhancements for Pacemaker rhythms and VF/Flutter).
4. **Validation Layer**: Automated sanity checking of feature values against established clinical norms.
5. **Classification Model**: A Multi-Label XGBoost or LightGBM model to predict concurrent disorders.
6. **Probability Calibration**: Applying Isotonic Regression for clinically true probability scores.
7. **Explainability Engine**: Fusion of SHAP importance weights and deterministic Clinical Rules (e.g., detecting RBBB via QRS prolongation + RSR' in V1) into a single, comprehensive JSON payload.

## Core Directories

- `/CardioVision_Feature_Pipeline`: Contains the main source code for the ML pipeline.
  - `/core`: Shared utilities and configurations.
  - `/data`: Scripts for data ingestion and standardization.
  - `/features`: Algorithms for extracting the 190 tabular features (e.g., `morphology.py`).
  - `/labeling`: Logic to unify diagnosis codes (e.g., SNOMED/SCP to the 12 target classes).
  - `/loaders`: Utilities to load waveforms from raw `.mat` or CSV files.
  - `/models`: Implementation of the XGBoost classifier and calibration routines.
  - `/processing`: Preprocessing and delineation scripts (`preprocessor.py`).
  - `/validation`: Automated sanity checks for the extracted features.
  - `/xai`: Explainable AI components, including SHAP analysis and the clinical rule engine.
  - `/pipeline`: Orchestration scripts to run the end-to-end process.

## Documentation

- `ecg_architecture_refinement.md`: Details the architectural phases, evaluation strategies, and JSON output design.
- `final_summary_report.md`: Audit report of class feasibility, multi-label analysis, and dataset contributions.
- `ecg_feature_pipeline_design.md`: Core design specifications for the feature pipeline.
- `pipeline walkthrough`: Guide to running the extraction and training pipeline.

## Getting Started

### 1. Environment Setup

It is recommended to use a Python virtual environment. Create and activate it using:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

Install the necessary core libraries for ECG processing, model training, and Explainable AI:

```bash
pip install pandas numpy neurokit2 scikit-learn xgboost shap matplotlib pyarrow wfdb
```

### 3. Running the Pipeline

The feature extraction pipeline processes the raw data to build a final `dataset.parquet` file ready for XGBoost. 

Create a `main.py` file in the root directory to initiate the job:

```python
import pandas as pd
from CardioVision_Feature_Pipeline.pipeline.orchestrator import PipelineOrchestrator

if __name__ == "__main__":
    # 1. Load your dataset index (e.g., list of ECG IDs and paths)
    df = pd.read_csv("ecg_filtered_dataset.csv")
    
    # 2. Initialize and run the pipeline
    orchestrator = PipelineOrchestrator()
    orchestrator.run_pipeline(df)
```

Run the pipeline from the terminal to begin extracting features across all chunks:

```bash
python main.py
```

**Checkpoint & Resume Feature**: 
If the extraction is interrupted (e.g., computer crash or manual stop), you can safely resume from the last completed chunk without reprocessing old data:
```bash
python main.py --resume
```

Check the `CardioVision_Feature_Pipeline/outputs/feature_chunks/` directory to view your saved chunks and `resume_report.txt`, and `outputs/final_dataset/` for the merged dataset.

### 4. Training the Model (GPU-Accelerated & XAI Pipeline)

The model training process has been upgraded to a fully automated, GPU-accelerated 8-phase architecture that includes hyperparameter optimization, rigorous cross-dataset evaluation, and SHAP explainability.

The 8 phases executed are:
1. **CUDA Diagnostics**: Verifies NVIDIA-SMI and XGBoost GPU support.
2. **Feature Selection & Drift Analysis**: Drops features with >15% missingness, zero variance, and correlation > 0.95. Flags dataset drift (KS > 0.3).
3. **Hyperparameter Tuning**: Uses Optuna to optimize difficult classes (PAC, LBBB, IVCD, LAE, Pacemaker) while applying efficient baselines to others.
4. **GPU Training & Calibration**: Trains 10 independent `XGBClassifier` models with `device="cuda"` and applies Isotonic Regression for probability calibration.
5. **Error Analysis**: Generates false positive, false negative, and hard case CSV reports per class.
6. **Cross-Dataset Generalization**: Evaluates models across 4 rotation experiments (train on 3 sources, test on 1) prioritizing PR-AUC.
7. **SHAP Explainability**: Generates global, per-class, and top-20 feature SHAP summaries.
8. **Clinical Validation**: Compares model predictions against deterministic clinical rules.

To execute the entire training and XAI pipeline, simply run the master orchestration script from the root directory:

```bash
# Ensure your virtual environment is activated
python run_training_pipeline.py
```

**Checkpoint & Resume Feature**:
Training 12 XGBoost models is computationally intensive. The pipeline actively saves checkpoints after each class completes. If training is interrupted, you can resume exactly where you left off by running:
```bash
python run_training_pipeline.py --resume
```

All generated reports, calibration thresholds, SHAP plots, and independent `.pkl` models will be securely organized within the `CardioVision_Feature_Pipeline/outputs/` directory structure.

### 5. Running Predictions (Inference)

To evaluate a single patient's ECG record and generate predictions using the 12 trained XGBoost models, you can use the newly included inference script.

Run a test prediction on a random patient sample from your dataset:
```bash
python predict.py --sample
```

This will print the positive disorder predictions, their probabilities against the clinical thresholds, and the key underlying ECG feature values. You can take this exact output and feed it into any LLM along with the `CardioVision_Feature_Pipeline/xai/llm_system_prompt.txt` to automatically generate a clinical AI report!

## 🚀 CardioVision Full-Stack App (RAG + React + FastAPI)

We have wrapped the pipeline into a complete, locally-hosted **Explainable AI Platform** featuring a **Phi3 RAG** integration!

### Prerequisites
1. You must have [Ollama](https://ollama.com/) installed.
2. Run `ollama pull phi3` in your terminal to download the language model.
3. Node.js must be installed to run the React frontend.
4. Install all Python dependencies by running:
```bash
pip install -r requirements.txt
```

### Step 1: Build the Medical Vector Database
Before starting the backend, you must build the local ChromaDB vector database so the RAG system can retrieve medical context for the LLM:
```bash
python backend/rag/chroma_ingest.py
```

### Step 2: Start the FastAPI Backend
Start the high-performance Python inference API:
```bash
uvicorn backend.app:app --reload
```
The backend will run on `http://127.0.0.1:8000`.

### Step 3: Start the React Frontend
Open a new terminal and start the Vite React server:
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` in your browser. You can now drag and drop a pre-extracted `.csv` of patient features (or raw `.mat` files to simulate extraction) and instantly view dynamic SHAP plots and the AI-generated medical report!
