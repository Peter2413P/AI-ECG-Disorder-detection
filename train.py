import pandas as pd
from CardioVision_Feature_Pipeline.models.xgboost_classifier import MultiLabelXGBoost
from CardioVision_Feature_Pipeline.models.calibration import IsotonicCalibrator
from CardioVision_Feature_Pipeline.xai.shap_explainer import SHAPExplainer
from CardioVision_Feature_Pipeline.validation.evaluator import model_evaluator

if __name__ == "__main__":
    # 1. Load the extracted features dataset
    print("Loading dataset.parquet...")
    df = pd.read_parquet("CardioVision_Feature_Pipeline/outputs/final_dataset/dataset.parquet")
    
    # Define features (X) and 12-class targets (y)
    # Ensure this matches your actual column definitions
    target_cols = [
        'NSR', 'Sinus_Tachycardia', 'Sinus_Arrhythmia', 'PAC', 'RBBB', 'LBBB', 
        'IVCD', 'WPW', 'Persistent_ST_Elevation', 'LAE', 'VF_Flutter', 'Pacemaker_Rhythm'
    ]
    X = df.drop(columns=target_cols + ['dataset_source', 'ecg_id'])
    y = df[target_cols]

    # Split dataset into Train, Calibration, and Test sets
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)

    # 2. Initialize and train the Multi-Label XGBoost Model
    print("Training Multi-Label XGBoost model...")
    model = MultiLabelXGBoost()
    model.fit(X_train, y_train)

    # 3. Perform Probability Calibration (Isotonic Regression)
    print("Calibrating probabilities...")
    calibrator = IsotonicCalibrator(model)
    calibrator.fit(X_calib, y_calib)

    # 4. Evaluate Model Performance
    print("Evaluating model...")
    y_pred_calibrated = calibrator.predict_proba(X_test)
    evaluation_results = model_evaluator(y_test, y_pred_calibrated)
    print(evaluation_results)

    # 5. Generate SHAP Explanations
    print("Setting up SHAP Explainer...")
    explainer = SHAPExplainer(model)
    explainer.fit(X_train)
    
    # Save models and explainers for API deployment
    print("Saving artifacts to outputs/models/...")
    model.save("CardioVision_Feature_Pipeline/outputs/models/xgboost_base.pkl")
    calibrator.save("CardioVision_Feature_Pipeline/outputs/models/isotonic_calibrator.pkl")
    explainer.save("CardioVision_Feature_Pipeline/outputs/models/shap_explainer.pkl")
    
    print("Training pipeline complete!")
