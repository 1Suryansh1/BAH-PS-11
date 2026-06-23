import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class DSRSIDDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=128, mask_ratio=0.50):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.mask_ratio = mask_ratio
        
        self.samples = [{"sample_id": f"dsrsid_{i}"} for i in range(100)]
        self.resize = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)

    def __len__(self):
        return len(self.samples)

    def normalize_pan(self, pan_raw):
        # Z-score per channel
        # Placeholder stats
        mean = 100.0
        std = 50.0
        return (pan_raw - mean) / std

    def normalize_ms(self, ms_raw):
        # Divide by max range, then z-score
        # Assuming uint16 for now
        ms = ms_raw / 10000.0
        mean = torch.tensor([0.1]*4).view(4, 1, 1)
        std = torch.tensor([0.05]*4).view(4, 1, 1)
        return (ms - mean) / std

    def __getitem__(self, idx):
        sample_id = self.samples[idx]["sample_id"]
        
        # Dummy data
        pan_raw = torch.rand(1, 256, 256) * 255.0
        ms_raw = torch.rand(4, 64, 64) * 10000.0
        
        # Resize to matching resolution
        pan_resized = self.resize(pan_raw)
        ms_resized = self.resize(ms_raw)
        
        pan_norm = self.normalize_pan(pan_resized)
        ms_norm = self.normalize_ms(ms_resized)
        
        # Single label (8 classes)
        label = torch.tensor(torch.randint(0, 8, ()).item(), dtype=torch.long)
        
        # Generate realistic coordinates and seasonal DOY
        lon, lat = 78.96, 20.59
        doy_pan = float(torch.randint(1, 366, ()).item())
        doy_ms = float(torch.randint(1, 366, ()).item())
        
        meta_pan = torch.tensor([lon, lat, doy_pan, 1.44], dtype=torch.float32)
        meta_ms = torch.tensor([lon, lat, doy_ms, 1.44], dtype=torch.float32)
        
        return {
            'pan': pan_norm,
            'ms': ms_norm,
            'label': label,
            'sample_id': sample_id,
            'meta_pan': meta_pan,
            'meta_ms': meta_ms
        }

