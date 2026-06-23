def get_modality_key(dataset_name: str, modality: str) -> str:
    """
    Maps configuration modality names (e.g. S1, S2_BEN, RGB, SAR, PAN, MS4)
    to their corresponding dataloader batch keys (e.g. 's1', 's2', 'rgb', 'sar', 'pan', 'ms').
    """
    mod_lower = modality.lower()
    dataset_lower = dataset_name.lower()
    
    # BigEarthNet
    if 'bigearthnet' in dataset_lower:
        if 's1' in mod_lower:
            return 's1'
        elif 's2' in mod_lower:
            return 's2'
            
    # CBRSIR
    elif 'cbrsir' in dataset_lower:
        if 'rgb' in mod_lower:
            return 'rgb'
        elif 'sar' in mod_lower or 's1' in mod_lower:
            return 'sar'
            
    # DSRSID
    elif 'dsrsid' in dataset_lower:
        if 'pan' in mod_lower:
            return 'pan'
        elif 'ms' in mod_lower:
            return 'ms'
            
    # Generic fallbacks
    if 's1' in mod_lower:
        return 's1'
    elif 's2' in mod_lower:
        return 's2'
    elif 'rgb' in mod_lower:
        return 'rgb'
    elif 'sar' in mod_lower:
        return 'sar'
    elif 'pan' in mod_lower:
        return 'pan'
    elif 'ms' in mod_lower:
        return 'ms'
        
    return mod_lower
