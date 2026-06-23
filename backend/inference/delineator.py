import neurokit2 as nk
import numpy as np

class ECGDelineatorWrapper:
    def __init__(self):
        pass

    def delineate(self, signal: list, fs: int):
        """
        Uses NeuroKit2 to delineate the ECG signal (usually Lead II) into P, QRS, and T components.
        Returns the indices for bounding boxes on the frontend.
        """
        # Convert list to numpy array, clean the signal
        sig_arr = np.array(signal)
        
        try:
            # 1. Clean the signal
            cleaned = nk.ecg_clean(sig_arr, sampling_rate=fs)
            
            # 2. Find R-peaks
            _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
            
            # 3. Delineate the waves
            # method="dwt" is generally robust for clinical data
            _, waves_peak = nk.ecg_delineate(cleaned, rpeaks['ECG_R_Peaks'], sampling_rate=fs, method="dwt", show=False)
            
            # 4. Extract region boundaries for the frontend
            # We will return lists of [start, end] tuples for P, QRS, and T regions
            
            regions = {
                "P_Waves": [],
                "QRS_Complexes": [],
                "T_Waves": []
            }
            
            # P waves
            p_onsets = waves_peak.get('ECG_P_Onsets', [])
            p_offsets = waves_peak.get('ECG_P_Offsets', [])
            for on, off in zip(p_onsets, p_offsets):
                if not np.isnan(on) and not np.isnan(off):
                    regions["P_Waves"].append([int(on), int(off)])
                    
            # QRS (Using R-peak as anchor if strict Q onset/S offset are noisy, but DWT usually provides them)
            r_onsets = waves_peak.get('ECG_R_Onsets', [])
            r_offsets = waves_peak.get('ECG_R_Offsets', [])
            for on, off in zip(r_onsets, r_offsets):
                if not np.isnan(on) and not np.isnan(off):
                    regions["QRS_Complexes"].append([int(on), int(off)])
                    
            # T waves
            t_onsets = waves_peak.get('ECG_T_Onsets', [])
            t_offsets = waves_peak.get('ECG_T_Offsets', [])
            for on, off in zip(t_onsets, t_offsets):
                if not np.isnan(on) and not np.isnan(off):
                    regions["T_Waves"].append([int(on), int(off)])
                    
            return regions
            
        except Exception as e:
            print(f"NeuroKit2 Delineation warning: {e}")
            # Return empty regions if it fails (e.g., extremely noisy signal or v-fib)
            return {"P_Waves": [], "QRS_Complexes": [], "T_Waves": []}
