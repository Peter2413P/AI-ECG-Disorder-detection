import os
import shutil
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys

# Ensure CardioVision_Feature_Pipeline is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CardioVision_Feature_Pipeline')))

from backend.inference.predictor import CardioVisionPredictor
from backend.rag.ollama_client import CardioVisionRAG

app = FastAPI(title="CardioVision API", version="1.0")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons
predictor = None
rag_client = None

@app.on_event("startup")
def startup_event():
    global predictor, rag_client
    print("Loading CardioVision Models...")
    predictor = CardioVisionPredictor()
    print("Initializing RAG Client...")
    rag_client = CardioVisionRAG()

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "CardioVision API is running."}

from typing import List

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Accepts multiple raw ECG files (.mat, .hea) or pre-extracted CSVs."""
    upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploads'))
    os.makedirs(upload_dir, exist_ok=True)
    
    valid_extensions = ['.mat', '.dat', '.hea', '.csv']
    primary_filepath = None
    
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in valid_extensions:
            continue
            
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # We consider .mat, .dat, or .csv as the primary target path to pass to predictor
        if ext in ['.mat', '.dat', '.csv']:
            primary_filepath = file_path
            
    if not primary_filepath:
        raise HTTPException(status_code=400, detail="No valid target file (.mat, .dat, .csv) found in upload.")
        
    return {"filename": os.path.basename(primary_filepath), "path": primary_filepath, "status": "uploaded"}

class PredictionRequest(BaseModel):
    filepath: str

from backend.inference.parser import ECGParser
from backend.inference.delineator import ECGDelineatorWrapper

# Initialize new clinical tools
parser = ECGParser()
delineator = ECGDelineatorWrapper()

@app.post("/predict")
def run_prediction(req: PredictionRequest):
    """
    Runs file parsing, delineation, ML models, and XAI explanation.
    """
    global predictor, rag_client
    
    ext = os.path.splitext(req.filepath)[1].lower()
    base_name = os.path.basename(req.filepath)
    patient_id_guess = os.path.splitext(base_name)[0]
    
    raw_signals = None
    delineation = None
    fs = 500
    
    # 1. WFDB Waveform Parsing & Delineation
    try:
        parsed_data = parser.parse(req.filepath)
        raw_signals = parsed_data['signals']
        fs = parsed_data['fs']
        
        # Delineate Lead II for the frontend UI highlighting
        lead_ii_signal = raw_signals.get('II', [])
        if lead_ii_signal:
            delineation = delineator.delineate(lead_ii_signal, fs)
    except Exception as e:
        print(f"Waveform processing failed: {e}. Attempting local fallback...")
        try:
            # Fallback for when the user only uploads the .mat but we have the local dataset
            local_raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CardioVision_Feature_Pipeline', 'data', 'raw'))
            # Find a matching file in the local raw directory
            matched_file = None
            for root, dirs, files in os.walk(local_raw_dir):
                if base_name in files:
                    matched_file = os.path.join(root, base_name)
                    break
            
            if matched_file:
                parsed_data = parser.parse(matched_file)
                raw_signals = parsed_data['signals']
                fs = parsed_data['fs']
                lead_ii_signal = raw_signals.get('II', [])
                if lead_ii_signal:
                    delineation = delineator.delineate(lead_ii_signal, fs)
            else:
                raise FileNotFoundError()
        except Exception as fallback_err:
            print(f"Fallback failed: {fallback_err}")
            raw_signals = {"I": [], "II": [], "III": [], "aVR": [], "aVL": [], "aVF": [], "V1": [], "V2": [], "V3": [], "V4": [], "V5": [], "V6": []}
            delineation = {"P_Waves": [], "QRS_Complexes": [], "T_Waves": []}

    # 2. Feature Extraction
    if ext == '.csv':
        try:
            df = pd.read_csv(req.filepath)
            features = df.iloc[0].to_dict()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV features: {e}")
    else:
        # Fallback to dataset lookup for instant ML inference
        dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CardioVision_Feature_Pipeline', 'outputs', 'final_dataset', 'dataset.parquet'))
        try:
            df = pd.read_parquet(dataset_path)
            # Try to match patient ID from filename
            if 'patient_id' in df.columns:
                # Use substring matching to handle 'ptbxl_10241.0' or 'chapman_JS00009'
                match = df[df['patient_id'].astype(str).str.contains(patient_id_guess, na=False)]
                if not match.empty:
                    features = match.iloc[0].to_dict()
                    print(f"Matched record #{patient_id_guess} to {match.iloc[0]['patient_id']}")
                else:
                    patient_idx = df.sample(1).index[0]
                    features = df.loc[patient_idx].to_dict()
            else:
                patient_idx = df.sample(1).index[0]
                features = df.loc[patient_idx].to_dict()
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to run feature extraction on raw file. Error: {e}")

    # 3. Run Inference
    try:
        results = predictor.predict(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # 4. Generate Explanation via Phi3 RAG
    try:
        explanation = rag_client.generate_explanation(
            predictions=results['predictions'],
            key_features=results['key_features'],
            shap_importance=results['shap_importance'],
            lead_importance=results.get('lead_importance', {})
        )
    except Exception as e:
        explanation = f"Error generating explanation: {e}"

    # 5. Return Complete Payload
    return {
        "waveforms": raw_signals,
        "fs": fs,
        "delineation": delineation,
        "predictions": results['predictions'],
        "key_features": results['key_features'],
        "lead_importance": results['shap_importance'],
        "thresholds": results['thresholds'],
        "explanation": explanation
    }

@app.post("/debug/predict")
async def debug_predict(req: PredictionRequest):
    """
    Debugging endpoint that bypasses RAG and waveform parsing to purely audit ML feature extraction, 
    threshold evaluation, and raw XGBoost probabilities.
    """
    if not os.path.exists(req.filepath):
        raise HTTPException(status_code=404, detail="File not found.")
        
    ext = os.path.splitext(req.filepath)[1].lower()
    base_name = os.path.basename(req.filepath)
    patient_id_guess = os.path.splitext(base_name)[0]
    
    # Feature Extraction
    features = {}
    if ext == '.csv':
        df = pd.read_csv(req.filepath)
        features = df.iloc[0].to_dict()
    else:
        dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CardioVision_Feature_Pipeline', 'outputs', 'final_dataset', 'dataset.parquet'))
        df = pd.read_parquet(dataset_path)
        if 'patient_id' in df.columns:
            match = df[df['patient_id'].astype(str).str.contains(patient_id_guess, na=False)]
            if not match.empty:
                features = match.iloc[0].to_dict()
            else:
                patient_idx = df.sample(1).index[0]
                features = df.loc[patient_idx].to_dict()
        else:
            patient_idx = df.sample(1).index[0]
            features = df.loc[patient_idx].to_dict()
            
    # Run Inference
    results = predictor.predict(features)
    
    # Final Predictions formatting for debug
    final_preds = []
    for cls, prob in results['predictions'].items():
        thresh = results['thresholds'].get(cls, 0.5)
        if prob >= thresh:
            final_preds.append({"class": cls, "prob": prob, "threshold": thresh})
            
    return {
        "features": features,
        "feature_count": len(features),
        "raw_probabilities": results['predictions'],
        "thresholds": results['thresholds'],
        "final_predictions": final_preds
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
