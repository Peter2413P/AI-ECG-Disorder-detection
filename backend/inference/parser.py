import os
import wfdb
import pandas as pd
import numpy as np

class ECGParser:
    def __init__(self):
        # Standard 12-lead names expected by the frontend
        self.standard_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def parse(self, record_path: str):
        """
        Parses a WFDB record or a CSV file.
        Returns: 
            signals: dict of 12 leads -> list of floats
            fs: sampling frequency
        """
        ext = os.path.splitext(record_path)[1].lower()
        base_path = os.path.splitext(record_path)[0]
        
        if ext == '.csv':
            return self._parse_csv(record_path)
        elif ext in ['.mat', '.dat', '.hea']:
            return self._parse_wfdb(base_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _parse_csv(self, file_path):
        df = pd.read_csv(file_path)
        signals = {}
        # If the CSV has standard leads as columns
        for lead in self.standard_leads:
            if lead in df.columns:
                signals[lead] = df[lead].tolist()
            else:
                signals[lead] = [0.0] * len(df)
                
        # Assume 500Hz for generic CSVs if not specified
        return {"signals": signals, "fs": 500}

    def _parse_wfdb(self, base_path):
        """
        Reads WFDB record (requires .hea and .mat/.dat to exist in the same dir)
        """
        try:
            record = wfdb.rdrecord(base_path)
            sig = record.p_signal
            sig_names = [name.strip().upper() for name in record.sig_name]
            fs = record.fs
            
            signals = {}
            for target_lead in self.standard_leads:
                target_upper = target_lead.upper()
                if target_upper in sig_names:
                    idx = sig_names.index(target_upper)
                    signals[target_lead] = sig[:, idx].tolist()
                else:
                    signals[target_lead] = [0.0] * sig.shape[0]
                    
            # Downsample or limit to 10 seconds for frontend performance (5000 samples @ 500Hz)
            # if sig.shape[0] > fs * 10:
            #    max_samples = int(fs * 10)
            #    for k in signals.keys():
            #        signals[k] = signals[k][:max_samples]
                    
            return {"signals": signals, "fs": fs}
        except Exception as e:
            raise RuntimeError(f"WFDB Parsing Failed: {str(e)}. Make sure both .hea and data files are uploaded together.")
