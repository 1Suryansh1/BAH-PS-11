import os
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T

class BEN14KDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=112, mask_ratio=0.50):
        """
        Args:
            root_dir: path to BigEarthNet-MM folder
            split: 'train', 'val', 'test'
            img_size: 112
            mask_ratio: 0.50
        """
        self.split = split
        self.img_size = img_size
        self.mask_ratio = mask_ratio
        
        # Sentinel-2 BigEarthNet Official stats (divided by 10000.0 to match reflectance scale [0, 1])
        self.s2_mean = torch.tensor([
            1353.7, 1117.2, 1041.8, 946.5, 1199.1, 2003.0, 
            2374.0, 2301.2, 2599.7, 732.1, 1820.6, 1118.2
        ]).view(-1, 1, 1) / 10000.0
        
        self.s2_std = torch.tensor([
            897.3, 736.0, 684.8, 620.0, 791.9, 1341.3, 
            1595.4, 1545.5, 1750.1, 475.1, 1216.5, 736.7
        ]).view(-1, 1, 1) / 10000.0

        self.resize = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)

        # Locate metadata.parquet and folders
        self.root_dir = root_dir
        parquet_path = os.path.join(root_dir, "metadata.parquet")
        if not os.path.exists(parquet_path):
            alternatives = [
                "BEN_14k",
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "BEN_14k"),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../../BEN_14k")),
            ]
            for alt in alternatives:
                p = os.path.join(alt, "metadata.parquet")
                if os.path.exists(p):
                    self.root_dir = alt
                    parquet_path = p
                    break
        
        import pandas as pd
        if os.path.exists(parquet_path):
            print(f"Loading metadata from {parquet_path}...")
            df = pd.read_parquet(parquet_path)
            # Filter by split
            df = df[df['split'] == split]
            
            # Check if folders exist
            s2_dir = os.path.join(self.root_dir, "BigEarthNet-S2", split)
            if os.path.exists(s2_dir):
                existing_patches = set([os.path.splitext(f)[0] for f in os.listdir(s2_dir) if f.endswith('.tif')])
                df = df[df['patch_id'].isin(existing_patches)]
                
            self.df = df
            self.use_real = len(self.df) > 0
            print(f"BEN14KDataset (split={split}) initialized with {len(self.df)} real samples.")
        else:
            self.use_real = False
            print(f"Warning: metadata.parquet not found. Falling back to synthetic data.")
            self.samples = [{"pair_id": f"sample_{i}", "s1_name": f"s1_{i}", "patch_id": f"s2_{i}", "labels": []} for i in range(100)]

    def __len__(self):
        return len(self.df) if self.use_real else len(self.samples)

    def normalize_s1(self, s1_db):
        # BigEarthNet-MM S1 is already in the dB scale. DO NOT calculate log10.
        mean = torch.tensor([-12.548, -20.192]).view(2, 1, 1)
        std = torch.tensor([5.257, 5.912]).view(2, 1, 1)
        return (s1_db - mean) / std

    def normalize_s2(self, s2_raw):
        # Divide by 10000 (uint16 scaled to reflectance)
        s2 = s2_raw / 10000.0
        s2 = torch.clamp(s2, 0.0, 1.0)
        return (s2 - self.s2_mean) / self.s2_std

    def __getitem__(self, idx):
        import re
        import datetime
        
        def get_doy(filename):
            match = re.search(r'\d{8}', filename)
            if match:
                date_str = match.group(0)
                try:
                    dt = datetime.datetime.strptime(date_str, "%Y%m%d")
                    return float(dt.timetuple().tm_yday)
                except ValueError:
                    pass
            return 1.0
            
        if self.use_real:
            row = self.df.iloc[idx]
            pair_id = row['patch_id']
            s1_name = row['s1_name']
            
            s1_path = os.path.join(self.root_dir, "BigEarthNet-S1", self.split, s1_name + ".tif")
            s2_path = os.path.join(self.root_dir, "BigEarthNet-S2", self.split, pair_id + ".tif")
            
            import rasterio
            import rasterio.warp
            
            # Read S1
            if os.path.exists(s1_path):
                with rasterio.open(s1_path) as src:
                    s1_db = src.read()
                    cx = (src.bounds.left + src.bounds.right) / 2
                    cy = (src.bounds.bottom + src.bounds.top) / 2
                    try:
                        xs, ys = rasterio.warp.transform(src.crs, 'EPSG:4326', [cx], [cy])
                        lon, lat = xs[0], ys[0]
                    except Exception:
                        lon, lat = 19.07, 45.13
            else:
                s1_db_vv = np.random.randn(1, 120, 120).astype(np.float32) * 5.257 - 12.548
                s1_db_vh = np.random.randn(1, 120, 120).astype(np.float32) * 5.912 - 20.192
                s1_db = np.concatenate([s1_db_vv, s1_db_vh], axis=0)
                lon, lat = 19.07, 45.13
                
            # Read S2
            if os.path.exists(s2_path):
                with rasterio.open(s2_path) as src:
                    s2_raw = src.read()
            else:
                s2_raw = np.random.randint(0, 10000, (12, 120, 120)).astype(np.uint16)
                
            # Process S1
            s1_db = torch.from_numpy(s1_db).float()
            s1_norm = self.normalize_s1(s1_db)
            s1_norm = self.resize(s1_norm)
            
            # Process S2
            s2_raw = torch.from_numpy(s2_raw.astype(np.float32))
            s2_norm = self.normalize_s2(s2_raw)
            s2_norm = self.resize(s2_norm)
            
            # Multi-label index mapping
            CLASS_NAMES = [
                'Agro-forestry areas',
                'Arable land',
                'Beaches, dunes, sands',
                'Broad-leaved forest',
                'Coastal wetlands',
                'Complex cultivation patterns',
                'Coniferous forest',
                'Industrial or commercial units',
                'Inland waters',
                'Inland wetlands',
                'Land principally occupied by agriculture, with significant areas of natural vegetation',
                'Marine waters',
                'Mixed forest',
                'Moors, heathland and sclerophyllous vegetation',
                'Natural grassland and sparsely vegetated areas',
                'Pastures',
                'Permanent crops',
                'Transitional woodland, shrub',
                'Urban fabric'
            ]
            label = torch.zeros(19, dtype=torch.float32)
            row_labels = row['labels']
            if isinstance(row_labels, (list, np.ndarray)):
                indices = [CLASS_NAMES.index(l) for l in row_labels if l in CLASS_NAMES]
                label[indices] = 1.0
                
            doy_s1 = get_doy(s1_name)
            doy_s2 = get_doy(pair_id)
            
            meta_s1 = torch.tensor([lon, lat, doy_s1, 1.44], dtype=torch.float32)
            meta_s2 = torch.tensor([lon, lat, doy_s2, 1.44], dtype=torch.float32)
            
            return {
                's1': s1_norm,
                's2': s2_norm,
                'label': label,
                'pair_id': pair_id,
                'meta_s1': meta_s1,
                'meta_s2': meta_s2
            }
            
        else:
            # Fallback synthetic branch
            sample = self.samples[idx]
            pair_id = sample['pair_id']
            
            # Generate S1 directly in dB scale (VV centered at -12.5dB, VH centered at -20.2dB)
            s1_db_vv = torch.randn(1, self.img_size, self.img_size) * 5.257 - 12.548
            s1_db_vh = torch.randn(1, self.img_size, self.img_size) * 5.912 - 20.192
            s1_db = torch.cat([s1_db_vv, s1_db_vh], dim=0)
            
            s2_raw = torch.randint(0, 10000, (12, self.img_size, self.img_size), dtype=torch.float32)
            
            s1_norm = self.normalize_s1(s1_db)
            s2_norm = self.normalize_s2(s2_raw)
            
            label = torch.zeros(19, dtype=torch.float32)
            label[torch.randint(0, 19, (3,))] = 1.0
            
            lon, lat = 19.07, 45.13
            doy_s1 = float(torch.randint(1, 366, ()).item())
            doy_s2 = float(torch.randint(1, 366, ()).item())
            
            meta_s1 = torch.tensor([lon, lat, doy_s1, 1.44], dtype=torch.float32)
            meta_s2 = torch.tensor([lon, lat, doy_s2, 1.44], dtype=torch.float32)
            
            return {
                's1': s1_norm,
                's2': s2_norm,
                'label': label,
                'pair_id': pair_id,
                'meta_s1': meta_s1,
                'meta_s2': meta_s2
            }

def compute_relevance(label_a, label_b):
    # Jaccard similarity of label sets
    intersection = (label_a * label_b).sum()
    union = ((label_a + label_b) > 0).float().sum()
    return intersection / (union + 1e-8)
