import pandas as pd
import numpy as np
from core.logger import get_logger

logger = get_logger("FeatureValidator")

class FeatureValidator:
    def __init__(self, output_path="../outputs/feature_validation_report.csv"):
        self.output_path = output_path
        
    def validate_features(self, df):
        """
        Takes the final feature DataFrame and computes validation metrics.
        Saves the report to output_path.
        """
        logger.info("Starting feature validation...")
        
        # Exclude metadata columns
        meta_cols = ['ecg_id', 'patient_id', 'dataset_source']
        label_cols = [c for c in df.columns if df[c].dropna().isin([0, 1]).all() and c not in meta_cols]
        feature_cols = [c for c in df.columns if c not in meta_cols and c not in label_cols]
        
        report_data = []
        
        for col in feature_cols:
            series = df[col]
            
            # Convert to numeric just in case
            series = pd.to_numeric(series, errors='coerce')
            
            mean_val = series.mean()
            std_val = series.std()
            min_val = series.min()
            max_val = series.max()
            missing_pct = series.isna().sum() / len(series) * 100.0
            
            # Outliers via IQR
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()
            
            report_data.append({
                "Feature": col,
                "Mean": mean_val,
                "StdDev": std_val,
                "Min": min_val,
                "Max": max_val,
                "Missing_Pct": missing_pct,
                "Outlier_Count": outlier_count
            })
            
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(self.output_path, index=False)
        logger.info(f"Feature validation report saved to {self.output_path}")
        
        # Flag highly missing features
        highly_missing = report_df[report_df['Missing_Pct'] > 15.0]['Feature'].tolist()
        if highly_missing:
            logger.warning(f"Features with >15% missing data: {highly_missing}")
            
        return report_df
