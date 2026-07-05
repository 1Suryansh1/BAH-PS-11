def get_modality_key(dataset_name: str, modality_name: str) -> str:
    """
    Maps the YAML config modality name (e.g., 'S1', 'S2_BEN', 'RGB') 
    to the exact dictionary key returned by the Dataset's __getitem__ 
    (e.g., 's1', 's2', 'rgb', 'sar').
    """
    dataset_name = dataset_name.lower()
    modality_name = modality_name.upper()
    
    if "bigearthnet" in dataset_name:
        if modality_name == "S1":
            return "s1"
        elif "S2" in modality_name:
            return "s2"
    elif "cbrsir" in dataset_name:
        if modality_name == "RGB":
            return "rgb"
        elif modality_name == "S1":
            return "sar"
    elif "dsrsid" in dataset_name:
        if modality_name == "PAN":
            return "pan"
        elif "MS" in modality_name:
            return "ms"
            
    # Fallback
    return modality_name.lower()
