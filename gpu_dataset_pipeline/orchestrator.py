import sys
import time
import torch
import json
import pandas as pd
from .config import DEVICE, TARGET_DISORDERS
from .dataset_scanner import discover_datasets_multiprocessed
from .gpu_processor import get_dataloader, process_signal_batch
from .batched_verifier import verify_batch_rules
from .file_manager import commit_batch_to_disk

def render_progress(batch_idx, num_batches, total_files_processed, speed, dataset_counts):
    """Renders the high-performance CLI output specified by the user."""
    mem_used = "N/A"
    if DEVICE.type == 'cuda':
        mem = torch.cuda.memory_allocated() / (1024**3)
        mem_used = f"{mem:.1f} GB"
        
    sys.stdout.write("\033[K") # Clear line
    print(f"\rGPU: {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU Fallback'} | "
          f"Mem: {mem_used} | "
          f"Batch: {batch_idx}/{num_batches} | "
          f"Files: {total_files_processed} | "
          f"Speed: {speed:.0f} ECGs/sec", end="")
    sys.stdout.flush()

def run_pipeline(input_dirs):
    print(f"Initializing GPU Pipeline on device: {DEVICE}")
    print("--------------------------------------------------")
    
    start_time = time.time()
    
    # 1. CPU Producer: Discover and map metadata
    valid_records = discover_datasets_multiprocessed(input_dirs)
    if not valid_records:
        print("No valid ECGs found matching targets.")
        return
        
    # 2. Setup PyTorch DataLoader
    print("Initializing PyTorch Batched DataLoader...")
    dataloader = get_dataloader(valid_records)
    num_batches = len(dataloader)
    
    total_processed = 0
    total_extracted = 0
    disorder_counts = {d: 0 for d in TARGET_DISORDERS}
    
    # 3. GPU Consumer Loop
    print("\n--- Starting Deep Extraction ---")
    batch_start_time = time.time()
    
    for batch_idx, (tensors, indices) in enumerate(dataloader, 1):
        B = tensors.shape[0]
        
        # A. Batched GPU Signal Processing
        features = process_signal_batch(tensors)
        
        # B. Batched Rule Verification
        valid_masks = {}
        for target in TARGET_DISORDERS:
            mask = verify_batch_rules(features, target)
            valid_masks[target] = mask
            disorder_counts[target] += int(mask.sum())
            
        # C. Retrieve Original Metadata Records
        batch_records = [valid_records[i.item()] for i in indices]
        
        # D. Asynchronous File Copy (CPU threads handle I/O while GPU is free)
        extracted_count = commit_batch_to_disk(batch_records, features, valid_masks)
        total_extracted += extracted_count
        
        total_processed += B
        
        # Clean VRAM occasionally
        if DEVICE.type == 'cuda' and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            
        # Update UI
        elapsed = time.time() - batch_start_time
        speed = total_processed / elapsed if elapsed > 0 else 0
        render_progress(batch_idx, num_batches, total_processed, speed, disorder_counts)
        
    print("\n\n--- Extraction Complete ---")
    
    # Generate summary JSON
    stats = {
        "Total_Records_Scanned": len(valid_records),
        "Total_Extracted": total_extracted,
        "Total_Rejected": len(valid_records) * len(TARGET_DISORDERS) - total_extracted,
        "Disorder_Breakdown": disorder_counts
    }
    
    import os
    from .config import OUTPUT_DIR
    with open(os.path.join(OUTPUT_DIR, "statistics.json"), "w") as f:
        json.dump(stats, f, indent=4)
        
    # Generate summary CSV
    summary_df = pd.DataFrame(list(disorder_counts.items()), columns=["Disorder", "Extracted_Count"])
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    
    print(f"Saved reports to {OUTPUT_DIR}/")

if __name__ == "__main__":
    import sys
    # E.g. python -m gpu_dataset_pipeline.orchestrator "D:\College\intern\final\ptb-xl..."
    inputs = sys.argv[1:] if len(sys.argv) > 1 else [os.getcwd()]
    run_pipeline(inputs)
