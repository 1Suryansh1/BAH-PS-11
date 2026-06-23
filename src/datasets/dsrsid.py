import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class DSRSIDDataset(Dataset):
    """
    DSRSID Dataset loader.
    """
    def __init__(self, root_dir, split='train', img_size=128, mask_ratio=0.50, 
                 ms_is_16bit=True, pan_mean=None, pan_std=None, ms_mean=None, ms_std=None):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.mask_ratio = mask_ratio
        
        # Set to True if MS data is 16-bit L2A surface reflectance (values roughly scaled by 10000)
        # Set to False if MS data is 8-bit imagery
        self.ms_is_16bit = ms_is_16bit
        
        # Actual dataset statistics should be computed and provided here
        self.pan_mean = pan_mean
        self.pan_std = pan_std
        self.ms_mean = ms_mean
        self.ms_std = ms_std
        
        self.samples = [{"sample_id": f"dsrsid_{i}"} for i in range(100)]
        self.resize = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)

    def __len__(self):
        return len(self.samples)

    def normalize_pan(self, pan_raw):
        # Scale to [0, 1] if raw is 0-255 (Dynamic scaling logic)
        pan = pan_raw / 255.0 if pan_raw.max() > 1.0 else pan_raw
        
        if self.pan_mean is not None and self.pan_std is not None:
            return (pan - self.pan_mean) / self.pan_std
        return pan

    def normalize_ms(self, ms_raw):
        # Apply 16-bit scaling if configured, else dynamic [0, 1] scaling
        if self.ms_is_16bit:
            ms = ms_raw / 10000.0
        else:
            ms = ms_raw / 255.0 if ms_raw.max() > 1.0 else ms_raw
            
        if self.ms_mean is not None and self.ms_std is not None:
            return (ms - self.ms_mean) / self.ms_std
        return ms

    def __getitem__(self, idx):
        sample_id = self.samples[idx]["sample_id"]
        
        # Dummy data generation (replacing with real image reading in practice)
        pan_raw = torch.rand(1, 256, 256) * 255.0
        if self.ms_is_16bit:
            ms_raw = torch.rand(4, 64, 64) * 10000.0
        else:
            ms_raw = torch.rand(4, 64, 64) * 255.0
        
        # Resize to matching resolution
        pan_resized = self.resize(pan_raw)
        ms_resized = self.resize(ms_raw)
        
        pan_norm = self.normalize_pan(pan_resized)
        ms_norm = self.normalize_ms(ms_resized)
        
        # Metadata: [lon, lat, time_doy, patch_area]
        time_doy = float(torch.randint(1, 366, ()).item())
        lon, lat = 120.1, 36.3
        
        # Gaofen-1 physical patch area for 256x256 at 2m/pixel: 0.262 km^2
        patch_area = 0.262
        
        pan_meta = torch.tensor([lon, lat, time_doy, patch_area], dtype=torch.float32)
        ms_meta = torch.tensor([lon, lat, time_doy, patch_area], dtype=torch.float32)
        
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

def compute_dsrsid_statistics(dataset, batch_size=16, num_workers=4):
    """
    Utility function to compute the actual mean and standard deviation over the DSRSID dataset.
    This replaces generic ImageNet statistics with dataset-specific distributions.
    
    Usage:
        dataset = DSRSIDDataset(root_dir="path/to/data", ms_is_16bit=True)
        stats = compute_dsrsid_statistics(dataset)
        
        # Then initialize the dataset again with proper stats
        dataset = DSRSIDDataset(..., **stats)
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    pan_sum, pan_sq_sum = 0.0, 0.0
    ms_sum, ms_sq_sum = 0.0, 0.0
    num_pixels_pan = 0
    num_pixels_ms = 0
    
    for batch in dataloader:
        pan = batch['pan'] # Expected (B, 1, H, W)
        ms = batch['ms']   # Expected (B, 4, H, W)
        
        pan_sum += pan.sum(dim=[0, 2, 3])
        pan_sq_sum += (pan ** 2).sum(dim=[0, 2, 3])
        num_pixels_pan += pan.size(0) * pan.size(2) * pan.size(3)
        
        ms_sum += ms.sum(dim=[0, 2, 3])
        ms_sq_sum += (ms ** 2).sum(dim=[0, 2, 3])
        num_pixels_ms += ms.size(0) * ms.size(2) * ms.size(3)
        
    pan_mean = pan_sum / num_pixels_pan
    pan_std = torch.sqrt((pan_sq_sum / num_pixels_pan) - (pan_mean ** 2))
    
    ms_mean = ms_sum / num_pixels_ms
    ms_std = torch.sqrt((ms_sq_sum / num_pixels_ms) - (ms_mean ** 2))
    
    return {
        'pan_mean': pan_mean.view(1, 1, 1),
        'pan_std': pan_std.view(1, 1, 1),
        'ms_mean': ms_mean.view(4, 1, 1),
        'ms_std': ms_std.view(4, 1, 1)
    }
