# dataset_pipeline/diagnosis_mapper.py

from .config import DIAGNOSIS_MAP, TARGET_DISORDERS

def map_diagnoses(raw_diagnoses):
    """
    Given a list of raw diagnosis strings/codes from metadata,
    maps them to the 12 target disorders.
    Returns a list of unique, matched target disorders.
    """
    matched_disorders = set()
    
    for raw_dx in raw_diagnoses:
        if not isinstance(raw_dx, str):
            raw_dx = str(raw_dx)
            
        dx_clean = raw_dx.strip().lower()
        
        # Direct match in map
        if dx_clean in DIAGNOSIS_MAP:
            matched_disorders.add(DIAGNOSIS_MAP[dx_clean])
            continue
            
        # Partial match if direct fails
        for key, target in DIAGNOSIS_MAP.items():
            if key in dx_clean:
                matched_disorders.add(target)
                break
                
    return list(matched_disorders)

def filter_targets(mapped_disorders):
    """
    Validates that the disorders belong to the 12 target project classes.
    """
    return [d for d in mapped_disorders if d in TARGET_DISORDERS]
