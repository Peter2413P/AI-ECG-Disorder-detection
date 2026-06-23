import os
import glob
import numpy as np
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from CardioVision_Feature_Pipeline.pipeline.orchestrator import PipelineOrchestrator
from CardioVision_Feature_Pipeline.core.logger import get_logger
from CardioVision_Feature_Pipeline.core.config import FINAL_DATASET_DIR

logger = get_logger("ParallelFeatureExtractor")

def process_single_row(row_tuple):
    """
    Worker function to process a single record. 
    We instantiate the Orchestrator per worker to ensure thread/process safety.
    """
    idx, row = row_tuple
    orchestrator = PipelineOrchestrator()
    
    try:
        labels = row.get('labels', row.get('original_codes', '[]'))
        if isinstance(labels, str):
            if labels.startswith('['):
                import json
                try:
                    labels = json.loads(labels)
                except Exception:
                    labels = [l.strip() for l in labels.strip('[]').replace('"', '').replace("'", '').split(',')]
            else:
                labels = [l.strip() for l in labels.split(',')]
                
        final_row = orchestrator.process_record(
            record_path=row['hea_path'],
            dataset_source=row['dataset_source'],
            labels=labels,
            patient_id=row.get('patient_id', None)
        )
        return idx, final_row, None
        
    except Exception as e:
        return idx, None, str(e)

def verify_chunk_integrity(filepath):
    """Verify chunk exists, is readable, and has rows."""
    if not os.path.exists(filepath):
        return False
    try:
        df = pd.read_parquet(filepath)
        if len(df) > 0:
            return True
        return False
    except Exception:
        return False

def run_parallel_extraction(metadata_df, n_chunks=10, resume=False):
    """
    Executes the feature extraction pipeline utilizing 80% of available CPU cores.
    Incorporates strict chunk integrity checks and checkpoint resume capabilities.
    """
    cpu_count = multiprocessing.cpu_count()
    target_cores = max(1, int(cpu_count * 0.8))
    
    logger.info(f"Starting PARALLEL chunked pipeline for {len(metadata_df)} records...")
    logger.info(f"Detected {cpu_count} CPU cores. Utilizing {target_cores} cores.")
    
    os.makedirs(FINAL_DATASET_DIR, exist_ok=True)
    chunks_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'feature_chunks'))
    os.makedirs(chunks_dir, exist_ok=True)
    
    df_chunks = np.array_split(metadata_df, n_chunks)
    all_failures = []
    
    # Pre-scan for resume functionality
    completed_chunks = []
    if resume:
        for i in range(n_chunks):
            chunk_file = os.path.join(chunks_dir, f"feature_chunk_{i:03d}.parquet")
            if verify_chunk_integrity(chunk_file):
                completed_chunks.append(i)
                
        remaining = n_chunks - len(completed_chunks)
        start_point = completed_chunks[-1] + 1 if completed_chunks else 0
        
        # Generate Resume Report
        report_content = (
            f"=== Extraction Resume Report ===\n"
            f"Total Chunks: {n_chunks}\n"
            f"Completed Chunks: {len(completed_chunks)}\n"
            f"Remaining Chunks: {remaining}\n"
            f"Resume Start Point: Chunk {start_point}\n"
        )
        with open(os.path.join(chunks_dir, "resume_report.txt"), "w") as f:
            f.write(report_content)
            
        logger.info(f"Resume requested. Verified {len(completed_chunks)} completed chunks. Remaining: {remaining}.")
    
    for i, chunk_df in enumerate(df_chunks):
        chunk_file = os.path.join(chunks_dir, f"feature_chunk_{i:03d}.parquet")
        
        if resume and i in completed_chunks:
            logger.info(f"Chunk {i}/{n_chunks} verified and skipped.")
            continue
            
        if os.path.exists(chunk_file) and not verify_chunk_integrity(chunk_file):
            logger.warning(f"Chunk {i}/{n_chunks} corrupted or empty. Rebuilding...")
            
        logger.info(f"Processing Chunk {i}/{n_chunks} ({len(chunk_df)} records)...")
        results = []
        failures = []
        
        tasks = list(chunk_df.iterrows())
        
        with ProcessPoolExecutor(max_workers=target_cores) as executor:
            futures = {executor.submit(process_single_row, task): task[0] for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Chunk {i}"):
                idx, final_row, error = future.result()
                
                if error:
                    failures.append({"index": idx, "error": error})
                elif final_row:
                    results.append(final_row)
                    
        if failures:
            logger.warning(f"Chunk {i}: {len(failures)} records failed.")
            all_failures.extend(failures)
            
        if results:
            chunk_result_df = pd.DataFrame(results)
            chunk_result_df.to_parquet(chunk_file, index=False)
            logger.info(f"Saved chunk {i} to {chunk_file}")
            
    # Combine chunks
    logger.info("All chunks processed. Combining into final dataset...")
    chunk_files = glob.glob(os.path.join(chunks_dir, "feature_chunk_*.parquet"))
    
    # Final sanity check to avoid duplication: read only unique chunks based on file path sorting
    chunk_files = sorted(chunk_files)
    
    if not chunk_files:
        logger.error("No valid chunks generated!")
        return None
        
    combined_df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
    
    parquet_path = os.path.join(FINAL_DATASET_DIR, "dataset.parquet")
    csv_path = os.path.join(FINAL_DATASET_DIR, "dataset.csv")
    
    combined_df.to_parquet(parquet_path, index=False)
    combined_df.to_csv(csv_path, index=False)
    logger.info(f"Pipeline complete. Final dataset aggregated ({len(combined_df)} records) saved to {parquet_path}")
    
    if all_failures:
        failures_df = pd.DataFrame(all_failures)
        os.makedirs(os.path.join(FINAL_DATASET_DIR, "..", "logs"), exist_ok=True)
        failures_df.to_csv(os.path.join(FINAL_DATASET_DIR, "..", "logs", "extraction_failures.csv"), index=False)
        logger.warning(f"Total {len(all_failures)} records failed across all chunks. See extraction_failures.csv.")
        
    return combined_df
