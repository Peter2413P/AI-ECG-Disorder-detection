import os
import sys
import pandas as pd

# Add the CardioVision_Feature_Pipeline directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'CardioVision_Feature_Pipeline')))

import argparse
from CardioVision_Feature_Pipeline.pipeline.run_feature_extraction import run_parallel_extraction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CardioVision Feature Extraction")
    parser.add_argument("--resume", action="store_true", help="Resume from previously completed chunks")
    args = parser.parse_args()

    # 1. Load your optimized dataset index
    dataset_path = os.path.join(os.path.dirname(__file__), "CardioVision_Feature_Pipeline", "outputs", "optimized_dataset", "optimized_labels_dataset.csv")
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # 2. Run the Parallel Feature Extraction Pipeline
    run_parallel_extraction(df, resume=args.resume)
