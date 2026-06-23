import os
import torch
from torch.utils.data import Dataset
import numpy as np

class BEN14KDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=112, mask_ratio=0.50):
        """
        Args:
            root_dir: path to BigEarthNet-MM folder
            split: 'train', 'val', 'test'
            img_size: 112
            mask_ratio: 0.50
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.mask_ratio = mask_ratio
        
        # Placeholder for actual data paths. In a real scenario, we parse metadata/LMDB here.
        self.samples = [{"pair_id": f"sample_{i}"} for i in range(100)]
        
        # Sentinel-2 BigEarthNet Official stats (divided by 10000.0 to match reflectance scale [0, 1])
        self.s2_mean = torch.tensor([
            1353.7, 1117.2, 1041.8, 946.5, 1199.1, 2003.0, 
            2374.0, 2301.2, 2599.7, 732.1, 1820.6, 1118.2
        ]).view(-1, 1, 1) / 10000.0
        
        self.s2_std = torch.tensor([
            897.3, 736.0, 684.8, 620.0, 791.9, 1341.3, 
            1595.4, 1545.5, 1750.1, 475.1, 1216.5, 736.7
        ]).view(-1, 1, 1) / 10000.0

    def __len__(self):
        return len(self.samples)

    def normalize_s1(self, s1_db):
        # BigEarthNet-MM S1 is already in the dB scale. DO NOT calculate log10.
        mean = torch.tensor([-12.548, -20.192]).view(2, 1, 1)
        std = torch.tensor([5.257, 5.912]).view(2, 1, 1)
        return (s1_db - mean) / std

    def normalize_s2(self, s2_raw):
        # Divide by 10000 (uint16 scaled to reflectance)
        s2 = s2_raw / 10000.0
        s2 = torch.clamp(s2, 0.0, 1.0)
        # Z-score normalize
        return (s2 - self.s2_mean) / self.s2_std

    def __getitem__(self, idx):
        # Dummy data generation - replace with real LMDB/GeoTIFF reading logic
        sample_id = self.samples[idx]["pair_id"]
        
        # Generate S1 directly in dB scale (VV centered at -12.5dB, VH centered at -20.2dB)
        s1_db_vv = torch.randn(1, self.img_size, self.img_size) * 5.257 - 12.548
        s1_db_vh = torch.randn(1, self.img_size, self.img_size) * 5.912 - 20.192
        s1_db = torch.cat([s1_db_vv, s1_db_vh], dim=0)
        
        # Generate S2 in raw DN range [0, 10000]
        s2_raw = torch.randint(0, 10000, (12, self.img_size, self.img_size), dtype=torch.float32)
        
        s1_norm = self.normalize_s1(s1_db)
        s2_norm = self.normalize_s2(s2_raw)
        
        # Metadata: [lon, lat, time_doy, patch_area]
        time_doy = torch.randint(1, 366, ()).item()
        s1_meta = torch.tensor([20.4, 44.8, float(time_doy), 0.0256], dtype=torch.float32)
        s2_meta = torch.tensor([20.4, 44.8, float(time_doy), 0.0256], dtype=torch.float32)
        
        # Multilabel annotation (19 classes)
        label = torch.zeros(19, dtype=torch.float32)
        label[torch.randint(0, 19, (3,))] = 1.0
        
        return {
            's1': s1_norm,
            's2': s2_norm,
            's1_meta': s1_meta,
            's2_meta': s2_meta,
            'label': label,
            'pair_id': sample_id
        }

def compute_relevance(label_a, label_b):
    # Jaccard similarity of label sets
    intersection = (label_a * label_b).sum()
    union = ((label_a + label_b) > 0).float().sum()
    return intersection / (union + 1e-8)
