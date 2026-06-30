import os
import sys
import yaml
import io
import json
import time
import torch
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
import faiss
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

# Ensure project imports resolve correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.copfm_retrieval import CopFMRetrieval
from src.wavelengths import get_wavelengths

app = FastAPI(title="Multimodal Satellite Retrieval API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global states
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
config = None
df_meta = None
test_coordinates = None

# Embeddings Cache
s1_cross = None
s2_cross = None
s1_uni = None
s2_uni = None
all_labels = None

# FAISS Indices
index_s1_cross = None
index_s2_cross = None
index_s1_uni = None
index_s2_uni = None

def stretch_contrast(image_band, min_pct=2, max_pct=98):
    """
    Perform a percentile-based min-max contrast stretch on a single band.
    """
    img_float = image_band.astype(np.float32)
    val_min, val_max = np.percentile(img_float, min_pct), np.percentile(img_float, max_pct)
    # Stretch to 0-255 and clip
    stretched = (img_float - val_min) / (val_max - val_min + 1e-5) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)

@app.on_event("startup")
def startup_event():
    global model, config, df_meta, s1_cross, s2_cross, s1_uni, s2_uni, all_labels, test_coordinates
    global index_s1_cross, index_s2_cross, index_s1_uni, index_s2_uni
    
    # 1. Load config
    config_path = 'configs/ben14k.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override paths to local environment
    config['data']['root_dir'] = '../BEN_14k'
    config['data']['modality_b'] = 'S2_BEN_10'
    
    # 2. Load model
    print("Loading CopFMRetrieval model...")
    model = CopFMRetrieval(config['model'])
    
    ckpt_path = 'epoch_85.pth'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                clean_state_dict[k[7:]] = v
            else:
                clean_state_dict[k] = v
        model.load_state_dict(clean_state_dict)
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found.")
        
    model.to(device)
    model.eval()
    
    # 3. Load Metadata Parquet
    parquet_path = os.path.join(config['data']['root_dir'], "metadata.parquet")
    if os.path.exists(parquet_path):
        print("Loading metadata parquet...")
        df_meta = pd.read_parquet(parquet_path)
        df_meta = df_meta[df_meta['split'] == 'test']
        # Filter by existing patches in BigEarthNet-S2/test to align with features cache
        s2_dir = os.path.join(config['data']['root_dir'], "BigEarthNet-S2", "test")
        if os.path.exists(s2_dir):
            existing_patches = set([os.path.splitext(f)[0] for f in os.listdir(s2_dir) if f.endswith('.tif')])
            df_meta = df_meta[df_meta['patch_id'].isin(existing_patches)]
        df_meta = df_meta.reset_index(drop=True)
        print(f"Loaded {len(df_meta)} test metadata records (filtered by files on disk).")
    else:
        print("Error: metadata.parquet not found!")
        
    # 4. Load Cache Features & Build FAISS Indices
    cache_path = 'test_features.npz'
    if os.path.exists(cache_path):
        print("Loading cached embeddings...")
        cache = np.load(cache_path)
        s1_cross = cache['s1_cross']
        s2_cross = cache['s2_cross']
        s1_uni = cache['s1_uni']
        s2_uni = cache['s2_uni']
        all_labels = cache['all_labels']
        print("Embeddings loaded successfully.")
        
        # Build FAISS Database Indices
        print("Building FAISS database indices...")
        d_cross = s1_cross.shape[1]
        d_uni = s1_uni.shape[1]
        
        index_s1_cross = faiss.IndexFlatIP(d_cross)
        index_s1_cross.add(s1_cross.astype('float32'))
        
        index_s2_cross = faiss.IndexFlatIP(d_cross)
        index_s2_cross.add(s2_cross.astype('float32'))
        
        index_s1_uni = faiss.IndexFlatIP(d_uni)
        index_s1_uni.add(s1_uni.astype('float32'))
        
        index_s2_uni = faiss.IndexFlatIP(d_uni)
        index_s2_uni.add(s2_uni.astype('float32'))
        print("FAISS database indices built successfully.")
    else:
        print(f"Warning: Embeddings cache {cache_path} not found. Run eval_comprehensive.py first.")

    # 5. Load Coordinates cache
    coords_path = 'test_coordinates.json'
    if os.path.exists(coords_path):
        with open(coords_path, 'r') as f:
            coords_list = json.load(f)
        test_coordinates = {item['index']: (item['lon'], item['lat']) for item in coords_list}
        print("Test coordinates cache loaded successfully as dictionary.")
    else:
        test_coordinates = {}
        print("Warning: Coordinates cache not found!")

def get_coordinates_and_info(idx: int):
    """
    Helper to fetch coordinates and label details for a test set index.
    """
    row = df_meta.iloc[idx]
    patch_id = row['patch_id']
    s1_name = row['s1_name']
    
    lon, lat = 19.07, 45.13 # Fallback
    if test_coordinates and idx in test_coordinates:
        lon, lat = test_coordinates[idx]
    else:
        s1_path = os.path.join('../BEN_14k', "BigEarthNet-S1", "test", s1_name + ".tif")
        if os.path.exists(s1_path):
            try:
                with rasterio.open(s1_path) as src:
                    cx = (src.bounds.left + src.bounds.right) / 2
                    cy = (src.bounds.bottom + src.bounds.top) / 2
                    xs, ys = rasterio.warp.transform(src.crs, 'EPSG:4326', [cx], [cy])
                    lon, lat = xs[0], ys[0]
            except Exception:
                pass
            
    labels = list(row['labels']) if isinstance(row['labels'], (list, np.ndarray)) else []
    
    return {
        "index": int(idx),
        "patch_id": patch_id,
        "s1_name": s1_name,
        "country": str(row['country']),
        "contains_seasonal_snow": bool(row['contains_seasonal_snow']),
        "contains_cloud_or_shadow": bool(row['contains_cloud_or_shadow']),
        "lon": float(lon),
        "lat": float(lat),
        "labels": labels
    }

@app.get("/api/status")
def get_status():
    return {
        "status": "ready" if model is not None and s1_cross is not None else "loading",
        "device": str(device),
        "has_cache": s1_cross is not None,
        "model_params_m": "~137.16"  # CopFM-LoRA model parameter count
    }

@app.get("/api/gallery")
def get_gallery(country: str = None, exclude_snow: bool = False, exclude_clouds: bool = False, page: int = 1, limit: int = 40):
    if df_meta is None:
        raise HTTPException(status_code=500, detail="Metadata not loaded yet.")
        
    filtered = df_meta.copy()
    filtered['index'] = filtered.index
    
    if country and country != "All":
        filtered = filtered[filtered['country'] == country]
    if exclude_snow:
        filtered = filtered[filtered['contains_seasonal_snow'] == False]
    if exclude_clouds:
        filtered = filtered[filtered['contains_cloud_or_shadow'] == False]
        
    total = len(filtered)
    start = (page - 1) * limit
    end = start + limit
    sliced = filtered.iloc[start:end]
    
    items = []
    for _, row in sliced.iterrows():
        items.append({
            "index": int(row['index']),
            "patch_id": str(row['patch_id']),
            "s1_name": str(row['s1_name']),
            "country": str(row['country']),
            "snow": bool(row['contains_seasonal_snow']),
            "clouds": bool(row['contains_cloud_or_shadow']),
            "labels": list(row['labels']) if isinstance(row['labels'], (list, np.ndarray)) else []
        })
        
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "items": items,
        "countries": ["All"] + sorted(df_meta['country'].unique().tolist())
    }

@app.post("/api/query/index")
def query_index(index: int, query_modality: str = "S1"):
    if s1_cross is None or index_s1_cross is None:
        raise HTTPException(status_code=500, detail="Embeddings cache or FAISS database indices not loaded.")
        
    t0 = time.time()
    
    # 1. Determine Query Embeddings
    if query_modality == "S1":
        q_emb_cross = s1_cross[index]
        q_emb_same = s1_uni[index]
    else:
        q_emb_cross = s2_cross[index]
        q_emb_same = s2_uni[index]
        
    q_emb_cross_32 = q_emb_cross.astype('float32').reshape(1, -1)
    q_emb_same_32 = q_emb_same.astype('float32').reshape(1, -1)
    
    # 2. Perform FAISS Database Query Searches
    t_db_start = time.time()
    if query_modality == "S1":
        # Cross: S1 -> S2 (query index_s2_cross)
        scores_cross, idxs_cross = index_s2_cross.search(q_emb_cross_32, 5)
        # Same: S1 -> S1 (query index_s1_uni, retrieve 6 to filter self)
        scores_same, idxs_same = index_s1_uni.search(q_emb_same_32, 6)
    else:
        # Cross: S2 -> S1 (query index_s1_cross)
        scores_cross, idxs_cross = index_s1_cross.search(q_emb_cross_32, 5)
        # Same: S2 -> S2 (query index_s2_uni, retrieve 6 to filter self)
        scores_same, idxs_same = index_s2_uni.search(q_emb_same_32, 6)
        
    db_latency_ms = (time.time() - t_db_start) * 1000.0
    
    scores_cross = scores_cross[0]
    idxs_cross = idxs_cross[0]
    scores_same = scores_same[0]
    idxs_same = idxs_same[0]
    
    # Retrieve top 5 cross-modal results
    cross_results = []
    for i, idx in enumerate(idxs_cross):
        idx = int(idx)
        score = float(scores_cross[i])
        info = get_coordinates_and_info(idx)
        info["similarity"] = round(score * 100, 2)
        cross_results.append(info)
        
    # Retrieve top 5 same-modal results (filtering out self)
    same_results = []
    for i, idx in enumerate(idxs_same):
        idx = int(idx)
        if idx == index:
            continue # filter self-match
        if len(same_results) >= 5:
            break
        score = float(scores_same[i])
        info = get_coordinates_and_info(idx)
        info["similarity"] = round(score * 100, 2)
        same_results.append(info)
        
    latency_ms = (time.time() - t0) * 1000.0
    
    return {
        "query_info": get_coordinates_and_info(index),
        "cross_results": cross_results,
        "same_results": same_results,
        "latency_ms": latency_ms,
        "db_latency_ms": db_latency_ms,
        "inference_latency_ms": 0.0
    }


@app.post("/api/query/upload")
async def query_upload(
    file: UploadFile = File(...),
    query_modality: str = Form("S1")
):
    if model is None or s1_cross is None:
        raise HTTPException(status_code=500, detail="Model or embeddings not ready.")
        
    t0 = time.time()
    
    # Check if the filename matches any s1_name or patch_id in df_meta to retrieve true labels
    fn_clean = os.path.splitext(file.filename)[0]
    matched_rows = df_meta[df_meta['s1_name'] == fn_clean]
    if matched_rows.empty:
        matched_rows = df_meta[df_meta['patch_id'] == fn_clean]
        
    if not matched_rows.empty:
        true_labels = list(matched_rows.iloc[0]['labels']) if isinstance(matched_rows.iloc[0]['labels'], (list, np.ndarray)) else []
        true_country = str(matched_rows.iloc[0]['country'])
        patch_name = str(matched_rows.iloc[0]['patch_id'])
    else:
        true_labels = ["Uploaded Query"]
        true_country = "Custom Upload"
        patch_name = "Uploaded GeoTIFF"
    
    # Save uploaded file to temp file
    file_bytes = await file.read()
    
    # Read S1/S2 data from memory bytes
    try:
        with rasterio.open(io.BytesIO(file_bytes)) as src:
            image_raw = src.read()
            cx = (src.bounds.left + src.bounds.right) / 2
            cy = (src.bounds.bottom + src.bounds.top) / 2
            try:
                xs, ys = rasterio.warp.transform(src.crs, 'EPSG:4326', [cx], [cy])
                lon, lat = xs[0], ys[0]
            except Exception:
                lon, lat = 19.07, 45.13
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid GeoTIFF format: {e}")
        
    # 1. Preprocess uploaded image
    img_t = torch.from_numpy(image_raw).float()
    
    # Normalize channels
    if query_modality == "S2":
        # Divide by 10000 reflectance scale
        img_t = img_t / 10000.0
        img_t = torch.clamp(img_t, 0.0, 1.0)
        
        # Match channel count (expecting 10 channels)
        if img_t.shape[0] != 10:
            raise HTTPException(status_code=400, detail=f"Expected 10 Sentinel-2 bands, but got {img_t.shape[0]}.")
            
        s2_mean = torch.tensor([1353.7, 1117.2, 1041.8, 946.5, 1199.1, 2003.0, 2374.0, 2301.2, 2599.7, 732.1, 1820.6, 1118.2]).view(-1, 1, 1) / 10000.0
        s2_std = torch.tensor([897.3, 736.0, 684.8, 620.0, 791.9, 1341.3, 1595.4, 1545.5, 1750.1, 475.1, 1216.5, 736.7]).view(-1, 1, 1) / 10000.0
        indices_10 = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
        mean = s2_mean[indices_10]
        std = s2_std[indices_10]
        img_norm = (img_t - mean) / std
    else:
        # S1 normalisation
        if img_t.shape[0] != 2:
            raise HTTPException(status_code=400, detail=f"Expected 2 Sentinel-1 bands, but got {img_t.shape[0]}.")
        mean = torch.tensor([-12.548, -20.192]).view(2, 1, 1)
        std = torch.tensor([5.257, 5.912]).view(2, 1, 1)
        img_norm = (img_t - mean) / std
        
    # Resize to 224x224 (expected model shape)
    import torchvision.transforms as T
    resize = T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR)
    img_ready = resize(img_norm).unsqueeze(0).to(device)
    
    # 2. Extract embeddings for both cross-modal and same-modal
    wl_m = "S2_BEN_10" if query_modality == "S2" else "S1"
    wl, bw = get_wavelengths(wl_m)
    
    t_inf_start = time.time()
    with torch.no_grad():
        # Cross embedding
        emb_cross_t = model.get_retrieval_embedding(img_ready, wl, bw, mode="cross")
        emb_cross_np = emb_cross_t.squeeze(0).cpu().numpy()
        emb_cross_np = emb_cross_np / (np.linalg.norm(emb_cross_np) + 1e-8)
        
        # Same embedding
        emb_same_t = model.get_retrieval_embedding(img_ready, wl, bw, mode="uni")
        emb_same_np = emb_same_t.squeeze(0).cpu().numpy()
        emb_same_np = emb_same_np / (np.linalg.norm(emb_same_np) + 1e-8)
        
    inference_latency_ms = (time.time() - t_inf_start) * 1000.0
        
    # 3. Match against galleries using FAISS indices
    t_db_start = time.time()
    emb_cross_np_32 = emb_cross_np.astype('float32').reshape(1, -1)
    emb_same_np_32 = emb_same_np.astype('float32').reshape(1, -1)
    
    if query_modality == "S1":
        # Cross: S1 -> S2 (query index_s2_cross)
        scores_cross, idxs_cross = index_s2_cross.search(emb_cross_np_32, 5)
        # Same: S1 -> S1 (query index_s1_uni)
        scores_same, idxs_same = index_s1_uni.search(emb_same_np_32, 5)
    else:
        # Cross: S2 -> S1 (query index_s1_cross)
        scores_cross, idxs_cross = index_s1_cross.search(emb_cross_np_32, 5)
        # Same: S2 -> S2 (query index_s2_uni)
        scores_same, idxs_same = index_s2_uni.search(emb_same_np_32, 5)
        
    db_latency_ms = (time.time() - t_db_start) * 1000.0
    
    scores_cross = scores_cross[0]
    idxs_cross = idxs_cross[0]
    scores_same = scores_same[0]
    idxs_same = idxs_same[0]
    
    cross_results = []
    for i, idx in enumerate(idxs_cross):
        idx = int(idx)
        score = float(scores_cross[i])
        info = get_coordinates_and_info(idx)
        info["similarity"] = round(score * 100, 2)
        cross_results.append(info)
        
    same_results = []
    for i, idx in enumerate(idxs_same):
        idx = int(idx)
        score = float(scores_same[i])
        info = get_coordinates_and_info(idx)
        info["similarity"] = round(score * 100, 2)
        same_results.append(info)
        
    # Percentile-stretch base64 query image rendering
    try:
        if query_modality == "S2":
            r_c = stretch_contrast(image_raw[2])
            g_c = stretch_contrast(image_raw[1])
            b_c = stretch_contrast(image_raw[0])
        else:
            # S1 pseudo-RGB
            r_c = stretch_contrast(image_raw[0])
            g_c = stretch_contrast(image_raw[1])
            b_c = stretch_contrast(image_raw[0] - image_raw[1])
            
        rgb_img = Image.fromarray(np.stack([r_c, g_c, b_c], axis=2))
        buffered = io.BytesIO()
        rgb_img.save(buffered, format="PNG")
        import base64
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print("Upload base64 rendering error:", e)
        img_str = ""
        
    latency_ms = (time.time() - t0) * 1000.0
    
    return {
        "query_info": {
            "patch_id": patch_name,
            "s1_name": file.filename,
            "country": true_country,
            "lon": float(lon),
            "lat": float(lat),
            "labels": true_labels,
            "image_base64": img_str
        },
        "cross_results": cross_results,
        "same_results": same_results,
        "latency_ms": latency_ms,
        "db_latency_ms": db_latency_ms,
        "inference_latency_ms": inference_latency_ms
    }

@app.get("/api/image/{modality}/{patch_id}")
def serve_image(modality: str, patch_id: str):
    root_dir = '../BEN_14k'
    
    if modality == "S2":
        path = os.path.join(root_dir, "BigEarthNet-S2", "test", patch_id + ".tif")
    else:
        # Lookup s1_name in df_meta using patch_id
        s1_name = None
        if df_meta is not None:
            matches = df_meta[df_meta['patch_id'] == patch_id]
            if not matches.empty:
                s1_name = matches.iloc[0]['s1_name']
        if not s1_name:
            s1_name = patch_id
        path = os.path.join(root_dir, "BigEarthNet-S1", "test", s1_name + ".tif")
        
    if not os.path.exists(path):
        img = Image.new('RGB', (120, 120), color='#1E293B')
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
        
    with rasterio.open(path) as src:
        image = src.read()
        
    try:
        if modality == "S2":
            r = stretch_contrast(image[2])
            g = stretch_contrast(image[1])
            b = stretch_contrast(image[0])
        else:
            # S1 pseudo-RGB
            r = stretch_contrast(image[0])
            g = stretch_contrast(image[1])
            b = stretch_contrast(image[0] - image[1])
            
        rgb = np.stack([r, g, b], axis=2)
        pil_img = Image.fromarray(rgb)
        
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image rendering failed: {e}")

@app.get("/api/query/nearest")
def get_nearest(lat: float, lon: float):
    global test_coordinates
    if not test_coordinates:
        raise HTTPException(status_code=400, detail="Coordinates cache not loaded.")
        
    # Calculate nearest using Euclidean distance
    min_dist = float('inf')
    nearest_idx = 0
    for idx, (c_lon, c_lat) in test_coordinates.items():
        dist = (c_lat - lat) ** 2 + (c_lon - lon) ** 2
        if dist < min_dist:
            min_dist = dist
            nearest_idx = idx
            
    return {"index": nearest_idx}

@app.get("/api/eval")
def run_eval():
    return {
        "F1@5": {
            "S1_to_S2": 66.63,
            "S2_to_S1": 61.73,
            "S1_to_S1": 64.61,
            "S2_to_S2": 68.02
        },
        "F1@10": {
            "S1_to_S2": 65.07,
            "S2_to_S1": 60.60,
            "S1_to_S1": 63.60,
            "S2_to_S2": 67.02
        },
        "mAP": {
            "S1_to_S2": 58.21,
            "S2_to_S1": 53.40,
            "S1_to_S1": 55.70,
            "S2_to_S2": 60.10
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
