import numpy as np

def extract_morphology_features(signal_1d, boundaries, fs):
    """
    Extract morphology (conduction, P, QRS, T, ST) features for a single lead.
    """
    features = {
        # Conduction
        "pr_interval": np.nan,
        "qrs_duration": np.nan,
        "qt_interval": np.nan,
        "qtc": np.nan,
        
        # P-Wave
        "p_amplitude": np.nan,
        "p_duration": np.nan,
        "p_area": np.nan,
        
        # QRS
        "r_amplitude": np.nan,
        "s_amplitude": np.nan,
        "qrs_area": np.nan,
        "qrs_energy": np.nan,
        "rs_ratio": np.nan,
        
        # ST Segment
        "st_deviation": np.nan,
        "st_slope": np.nan,
        
        # T-Wave
        "t_amplitude": np.nan,
        "t_duration": np.nan,
        "t_symmetry": np.nan
    }
    
    if not boundaries:
        return features
        
    def _safe_mean_diff(onsets, offsets):
        diffs = []
        for on, off in zip(onsets, offsets):
            if on is not None and off is not None and off > on:
                diffs.append((off - on) / fs * 1000.0) # in ms
        return np.mean(diffs) if diffs else np.nan
        
    # Conduction
    features["qrs_duration"] = _safe_mean_diff(boundaries.get('QRS_Onsets', []), boundaries.get('QRS_Offsets', []))
    features["p_duration"] = _safe_mean_diff(boundaries.get('P_Onsets', []), boundaries.get('P_Offsets', []))
    features["t_duration"] = _safe_mean_diff(boundaries.get('T_Onsets', []), boundaries.get('T_Offsets', []))
    features["pr_interval"] = _safe_mean_diff(boundaries.get('P_Onsets', []), boundaries.get('QRS_Onsets', []))
    features["qt_interval"] = _safe_mean_diff(boundaries.get('QRS_Onsets', []), boundaries.get('T_Offsets', []))
    
    # QTc (Bazett's formula requires RR in seconds, so we need global RR_mean, but we'll approximate here if needed, or leave to extractor to fill)
    
    # Simple Amplitudes & Areas
    p_amps, r_amps, s_amps, t_amps = [], [], [], []
    qrs_areas = []
    
    for i in range(len(boundaries.get('R_Peaks', []))):
        try:
            # QRS
            qrs_on = boundaries['QRS_Onsets'][i] if i < len(boundaries['QRS_Onsets']) else None
            qrs_off = boundaries['QRS_Offsets'][i] if i < len(boundaries['QRS_Offsets']) else None
            r_peak = boundaries['R_Peaks'][i]
            
            if r_peak is not None:
                r_amps.append(signal_1d[r_peak])
                
            if qrs_on is not None and qrs_off is not None and qrs_off > qrs_on:
                qrs_seg = signal_1d[qrs_on:qrs_off]
                s_amps.append(np.min(qrs_seg))
                if len(qrs_seg) > 0:
                    qrs_areas.append(np.trapezoid(np.abs(qrs_seg)))
                
            # P
            p_on = boundaries['P_Onsets'][i] if i < len(boundaries['P_Onsets']) else None
            p_off = boundaries['P_Offsets'][i] if i < len(boundaries['P_Offsets']) else None
            if p_on is not None and p_off is not None and p_off > p_on:
                p_amps.append(np.max(signal_1d[p_on:p_off]))
                
            # T
            t_on = boundaries['T_Onsets'][i] if i < len(boundaries['T_Onsets']) else None
            t_off = boundaries['T_Offsets'][i] if i < len(boundaries['T_Offsets']) else None
            if t_on is not None and t_off is not None and t_off > t_on:
                t_amps.append(np.max(np.abs(signal_1d[t_on:t_off])) * np.sign(np.sum(signal_1d[t_on:t_off])))
                
        except IndexError:
            pass

    if r_amps: features["r_amplitude"] = np.mean(r_amps)
    if s_amps: features["s_amplitude"] = np.mean(s_amps)
    if p_amps: features["p_amplitude"] = np.mean(p_amps)
    if t_amps: features["t_amplitude"] = np.mean(t_amps)
    if qrs_areas: features["qrs_area"] = np.mean(qrs_areas)
    
    if features["s_amplitude"] != 0 and not np.isnan(features["s_amplitude"]):
        features["rs_ratio"] = abs(features["r_amplitude"] / features["s_amplitude"])
        
    # ST Deviation (60ms after QRS offset)
    st_devs = []
    offset_samples = int(0.06 * fs)
    for q_off in boundaries.get('QRS_Offsets', []):
        if q_off is not None:
            st_point = q_off + offset_samples
            if st_point < len(signal_1d):
                st_devs.append(signal_1d[st_point])
    if st_devs:
        features["st_deviation"] = np.mean(st_devs)

    return features
