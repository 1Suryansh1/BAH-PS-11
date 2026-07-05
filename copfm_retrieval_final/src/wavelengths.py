MODALITY_WAVELENGTHS = {
    # Sentinel-1 SAR (C-band ~5.6cm = 56,000,000 nm)
    "S1": {
        "wavelengths": [56_000_000.0, 56_000_000.0],   # VV, VH (nm)
        "bandwidths":  [1_000_000_000.0, 1_000_000_000.0],
        "channels": 2,
        "name": "Sentinel-1 SAR VV+VH"
    },
    # Sentinel-2 Multispectral (13 bands, B1-B12 + B8A)
    "S2": {
        "wavelengths": [442.7, 492.4, 559.8, 664.6, 704.1,
                        740.5, 782.8, 832.8, 864.7, 945.1,
                        1373.5, 1613.7, 2202.4],
        "bandwidths":  [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180],
        "channels": 13,
        "name": "Sentinel-2 TOA"
    },
    # Sentinel-2 BigEarthNet (12 bands, dropping B10)
    "S2_BEN": {
        "wavelengths": [442.7, 492.4, 559.8, 664.6, 704.1,
                        740.5, 782.8, 832.8, 864.7, 945.1,
                        1613.7, 2202.4],
        "bandwidths":  [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 90, 180],
        "channels": 12,
        "name": "Sentinel-2 BigEarthNet"
    },
    # Sentinel-2 BigEarthNet (10 bands, dropping B1, B9, B10)
    "S2_BEN_10": {
        "wavelengths": [492.4, 559.8, 664.6, 704.1,
                        740.5, 782.8, 832.8, 864.7, 
                        1613.7, 2202.4],
        "bandwidths":  [65, 35, 30, 15, 15, 20, 115, 20, 90, 180],
        "channels": 10,
        "name": "Sentinel-2 BigEarthNet 10-Band"
    },
    # RGB optical (approximate, adjust if dataset has exact band specs)
    "RGB": {
        "wavelengths": [620.0, 532.0, 450.0],           # R, G, B
        "bandwidths":  [100.0, 90.0, 90.0],
        "channels": 3,
        "name": "RGB Optical"
    },
    # Panchromatic (Cartosat / DSRSID PAN band)
    "PAN": {
        "wavelengths": [675.0],                          # broad 520-860nm, center
        "bandwidths":  [340.0],
        "channels": 1,
        "name": "Panchromatic"
    },
    # Multispectral 4-band (Gaofen-1 / DSRSID MS)
    "MS4": {
        "wavelengths": [450.0, 520.0, 630.0, 760.0],   # B, G, R, NIR
        "bandwidths":  [90.0, 70.0, 60.0, 130.0],
        "channels": 4,
        "name": "Multispectral 4-band"
    }
}

def get_wavelengths(modality: str):
    """
    Returns wavelengths and bandwidths for the given modality.
    """
    if modality not in MODALITY_WAVELENGTHS:
        raise ValueError(f"Modality {modality} not found in MODALITY_WAVELENGTHS")
    
    info = MODALITY_WAVELENGTHS[modality]
    return info["wavelengths"], info["bandwidths"]
