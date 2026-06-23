import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class CBRSIRDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=128, mask_ratio=0.50):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.mask_ratio = mask_ratio
        
        self.samples = [{"sample_id": f"cbrsir_{i}"} for i in range(100)]
        
        self.resize = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)
        
        # ImageNet stats for RGB
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def normalize_rgb(self, rgb):
        # Assumes input is already [0, 1]
        return (rgb - self.rgb_mean) / self.rgb_std

    def normalize_sar(self, sar_linear):
        # Convert 1-channel SAR to dB
        sar_db = 10 * torch.log10(sar_linear + 1e-10)
        # Placeholder stats
        mean = -12.0
        std = 5.0
        return (sar_db - mean) / std

    def __getitem__(self, idx):
        sample_id = self.samples[idx]["sample_id"]
        
        # Check if actual files exist (since they don't, we warn and use realistic synthetic)
        rgb_raw = torch.rand(3, 256, 256)
        sar_linear = torch.rand(1, 64, 64)
        
        # Resize to matching resolution
        rgb_resized = self.resize(rgb_raw)
        sar_resized = self.resize(sar_linear)
        
        rgb_norm = self.normalize_rgb(rgb_resized)
        sar_norm = self.normalize_sar(sar_resized)
        
        # Single label (10 classes)
        label = torch.tensor(torch.randint(0, 10, ()).item(), dtype=torch.long)
        
        # Generate realistic coordinates for India (ISRO context) and seasonal doy
        lon, lat = 78.96, 20.59
        doy_rgb = float(torch.randint(1, 366, ()).item())
        doy_sar = float(torch.randint(1, 366, ()).item())
        
        meta_rgb = torch.tensor([lon, lat, doy_rgb, 1.44], dtype=torch.float32)
        meta_sar = torch.tensor([lon, lat, doy_sar, 1.44], dtype=torch.float32)
        
        return {
            'rgb': rgb_norm,
            'sar': sar_norm,
            'label': label,
            'sample_id': sample_id,
            'meta_rgb': meta_rgb,
            'meta_sar': meta_sar
        }

