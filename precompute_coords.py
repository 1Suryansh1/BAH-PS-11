import os
import json
import pandas as pd
import rasterio
import rasterio.warp
from tqdm import tqdm

print("Precomputing coordinates for all test split patches...")
root_dir = '../BEN_14k'
parquet_path = os.path.join(root_dir, "metadata.parquet")

if not os.path.exists(parquet_path):
    print("Error: metadata.parquet not found!")
    exit(1)

df = pd.read_parquet(parquet_path)
df = df[df['split'] == 'test'].reset_index(drop=True)

# Keep only existing patches in BigEarthNet-S2/test to match dataset length
s2_dir = os.path.join(root_dir, "BigEarthNet-S2", "test")
if os.path.exists(s2_dir):
    existing_patches = set([os.path.splitext(f)[0] for f in os.listdir(s2_dir) if f.endswith('.tif')])
    df = df[df['patch_id'].isin(existing_patches)].reset_index(drop=True)

coords = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    s1_name = row['s1_name']
    s1_path = os.path.join(root_dir, "BigEarthNet-S1", "test", s1_name + ".tif")
    
    lon, lat = 19.07, 45.13 # default
    if os.path.exists(s1_path):
        try:
            with rasterio.open(s1_path) as src:
                cx = (src.bounds.left + src.bounds.right) / 2
                cy = (src.bounds.bottom + src.bounds.top) / 2
                xs, ys = rasterio.warp.transform(src.crs, 'EPSG:4326', [cx], [cy])
                lon, lat = xs[0], ys[0]
        except Exception:
            pass
            
    coords.append({"index": idx, "lon": lon, "lat": lat})

# Save to JSON
with open("test_coordinates.json", "w") as f:
    json.dump(coords, f)

print(f"Coordinates saved successfully for {len(coords)} test items.")
