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
        # Scale to [0, 1] if raw is 0-255
        pan = pan_raw / 255.0 if pan_raw.max() > 1.0 else pan_raw
        mean = 0.5
        std = 0.25
        return (pan - mean) / std

    def normalize_ms(self, ms_raw):
        # Scale to [0, 1] if raw is 0-255
        ms = ms_raw / 255.0 if ms_raw.max() > 1.0 else ms_raw
        # B, G, R, NIR bands
        mean = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(4, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225, 0.25]).view(4, 1, 1)
        return (ms - mean) / std

    def __getitem__(self, idx):
        sample_id = self.samples[idx]["sample_id"]
        
        # Dummy data in real-world ranges (RGB/NIR [0, 255], PAN [0, 255])
        pan_raw = torch.rand(1, 256, 256) * 255.0
        ms_raw = torch.rand(4, 64, 64) * 255.0
        
        # Resize to matching resolution
        pan_resized = self.resize(pan_raw)
        ms_resized = self.resize(ms_raw)
        
        pan_norm = self.normalize_pan(pan_resized)
        ms_norm = self.normalize_ms(ms_resized)
        
        # Metadata: [lon, lat, time_doy, patch_area]
        time_doy = torch.randint(1, 366, ()).item()
        pan_meta = torch.tensor([120.1, 36.3, float(time_doy), 0.262], dtype=torch.float32)
        ms_meta = torch.tensor([120.1, 36.3, float(time_doy), 0.262], dtype=torch.float32)
        
        # Single label (8 classes)
        label = torch.tensor(torch.randint(0, 8, ()).item(), dtype=torch.long)
        
        return {
            'pan': pan_norm,
            'ms': ms_norm,
            'pan_meta': pan_meta,
            'ms_meta': ms_meta,
            'label': label,
            'sample_id': sample_id
        }
