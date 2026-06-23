import os
import argparse
import pandas as pd
import joblib

def load_thresholds(threshold_path):
    if not os.path.exists(threshold_path):
        raise FileNotFoundError("Optimal thresholds not found. Please train models first.")
    df = pd.read_csv(threshold_path)
    return dict(zip(df['Target Class'], df['Optimal Threshold']))

def main():
    parser = argparse.ArgumentParser(description="CardioVision Inference & Prediction")
    parser.add_argument("--sample", action="store_true", help="Run prediction on a random sample from the dataset")
    args = parser.parse_args()
    
    print("=== CardioVision Inference Engine ===")
    
    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'CardioVision_Feature_Pipeline', 'outputs'))
    models_dir = os.path.join(base_dir, 'models')
    dataset_path = os.path.join(base_dir, 'final_dataset', 'dataset.parquet')
    threshold_path = os.path.join(base_dir, 'thresholds', 'optimal_thresholds.csv')
    
    thresholds = load_thresholds(threshold_path)
    
    # Load Sample
    if args.sample:
        print(f"Loading random sample from {dataset_path}...")
        df = pd.read_parquet(dataset_path)
        patient_idx = df.sample(1).index[0]
        patient_data = df.loc[[patient_idx]].copy()
        print(f"Running inference for Patient Record #{patient_idx}...")
    else:
        print("Please provide --sample to test inference. (Support for --ecg_file coming soon).")
        return

    # Drop target columns to get features
    target_classes = list(thresholds.keys())
    drop_cols = target_classes + ['dataset_source', 'patient_id', 'ecg_id']
    X_features = patient_data.drop(columns=[c for c in drop_cols if c in patient_data.columns])
    
    predictions = {}
    
    for cls, thresh in thresholds.items():
        model_path = os.path.join(models_dir, f"{cls.lower()}_model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # Ensure we only pass the exact features the model was trained on
            model_features = list(model.feature_names_in_)
            X_model_input = patient_data[model_features]
            
            prob = model.predict_proba(X_model_input)[:, 1][0]
            
            if prob >= thresh:
                predictions[cls] = prob
                
    print("\n" + "="*50)
    print("POSITIVE PREDICTIONS DETECTED")
    print("="*50)
    
    if not predictions:
        print("No abnormalities detected. Normal ECG predicted.")
    else:
        for cls, prob in predictions.items():
            print(f"- {cls}: {prob:.1%} Probability (Threshold: {thresholds[cls]:.1%})")
            
    print("\n" + "="*50)
    print("KEY ECG FEATURES")
    print("="*50)
    
    key_features = ['Heart_Rate', 'RR_Mean', 'RR_Std', 'pNN50', 'P_Duration', 'PR_Interval', 'QRS_Dur', 'T_Duration']
    for feat in key_features:
        if feat in X_features.columns:
            print(f"{feat}: {X_features[feat].values[0]:.2f}")
            
    print("\n[To get a medical explanation of this prediction, you can copy the above output and paste it to the CardioVision AI Assistant LLM using the prompt in `CardioVision_Feature_Pipeline/xai/llm_system_prompt.txt`.]")

if __name__ == "__main__":
    main()
