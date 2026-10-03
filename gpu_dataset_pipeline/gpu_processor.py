import torch
import numpy as np
import wfdb
from torch.utils.data import Dataset, DataLoader
from .config import DEVICE, TARGET_FS, TARGET_LENGTH, BATCH_SIZE

class BatchedECGDataset(Dataset):
    """PyTorch Dataset to load raw signals into tensors."""
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        try:
            # wfdb.rdsamp expects path without extension
            signals, fields = wfdb.rdsamp(record["base_path"])
            # Transpose to shape (num_leads, signal_length)
            signals = signals.T 
            
            # Pad or truncate to target length
            if signals.shape[1] > TARGET_LENGTH:
                signals = signals[:, :TARGET_LENGTH]
            elif signals.shape[1] < TARGET_LENGTH:
                pad_width = TARGET_LENGTH - signals.shape[1]
                signals = np.pad(signals, ((0, 0), (0, pad_width)), mode='constant')
                
            tensor = torch.tensor(signals, dtype=torch.float32)
            return tensor, idx
        except Exception:
            # Return empty tensor if read fails
            return torch.zeros((12, TARGET_LENGTH), dtype=torch.float32), idx

def get_dataloader(records):
    """Creates a PyTorch DataLoader for batch processing."""
    dataset = BatchedECGDataset(records)
    # Use multiple workers for I/O if possible
    num_workers = 0 # Set to 0 on Windows to avoid pickling issues in interactive mode, can be scaled up
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))

def process_signal_batch(tensor_batch):
    """
    Applies vectorised operations across the entire batch (B, 12, L).
    Returns basic features extracted on GPU/CPU.
    """
    # Move to GPU if available and use half precision for speed
    if DEVICE.type == 'cuda':
        tensor_batch = tensor_batch.to(DEVICE, non_blocking=True).half()
        
    B, C, L = tensor_batch.shape
    
    # 1. Baseline Correction (Subtract mean per lead)
    means = tensor_batch.mean(dim=2, keepdim=True)
    centered = tensor_batch - means
    
    # 2. Simple High-pass / Denoise proxy using FFT (PyTorch native)
    # In real medical pipelines, you'd apply a strict Conv1d bandpass filter.
    # We compute energy and variance as features.
    
    variances = centered.var(dim=2) # Shape: (B, 12)
    max_amplitudes = centered.abs().max(dim=2)[0] # Shape: (B, 12)
    
    # Simulate a proxy for Heart Rate via dominant frequency
    # (Since we are skipping full NeuroKit2 delineator to use pure PyTorch tensors)
    fft_vals = torch.fft.rfft(centered.float(), dim=2)
    power = torch.abs(fft_vals) ** 2
    
    # Blank out frequencies below 1Hz and above 10Hz to isolate HR
    freqs = torch.fft.rfftfreq(L, 1.0/TARGET_FS).to(DEVICE)
    mask = (freqs > 1.0) & (freqs < 3.0) # HR between 60-180 bpm roughly
    
    power[:, :, ~mask] = 0
    # Find dominant frequency bin
    dom_freq_idx = torch.argmax(power, dim=2)
    dom_freq = freqs[dom_freq_idx] # Shape: (B, 12)
    
    # Average dominant frequency across leads, convert to BPM
    hr_bpm = (dom_freq.float().mean(dim=1) * 60.0)
    
    # Simulated QRS duration proxy: based on high-frequency energy ratio
    hf_mask = (freqs > 10.0) & (freqs < 40.0)
    lf_mask = (freqs > 1.0) & (freqs < 10.0)
    
    # Avoid division by zero
    hf_energy = torch.sum(torch.abs(fft_vals[:, :, hf_mask]), dim=2) + 1e-6
    lf_energy = torch.sum(torch.abs(fft_vals[:, :, lf_mask]), dim=2) + 1e-6
    energy_ratio = (hf_energy / lf_energy).mean(dim=1)
    
    # Rough mapping: higher ratio = narrower QRS. We map to 80-160ms
    qrs_proxy = 160.0 - (energy_ratio * 40.0)
    qrs_proxy = torch.clamp(qrs_proxy, min=60.0, max=200.0)
    
    # Return features back to CPU for rule verification
    return {
        "heart_rate": hr_bpm.cpu().numpy(),
        "qrs_duration": qrs_proxy.cpu().numpy(),
        "variance": variances.cpu().numpy(),
        "max_amp": max_amplitudes.cpu().numpy()
    }
