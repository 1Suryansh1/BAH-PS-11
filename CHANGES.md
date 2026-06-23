# Copernicus-FM Retrieval — Codebase Changes Summary

This document lists all the bug fixes, structural improvements, and dataset updates made to this codebase.

---

## 1. Vectorized and Safe Predictor
* **File**: [src/models/predictor.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/src/models/predictor.py)
* **Fixes**:
  * Squeezed the unused first dimension of `self.mask_token` before calling `.expand(...)` to avoid a PyTorch shape mismatch `RuntimeError`.
  * Dynamically interpolates the positional embeddings `predictor_pos_embed` based on the actual patch size, preventing `IndexError: out of range` on larger image resolutions (e.g., 384x384).
  * Vectorized the selection of masked position embeddings using boolean index tensors to optimize GPU usage and eliminate sequential Python loops.

## 2. Modality Translation & Key Mapping
* **Files**: 
  * [src/utils/key_mapping.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/src/utils/key_mapping.py)
  * [train.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/train.py)
  * [src/evaluation/retrieval_eval.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/src/evaluation/retrieval_eval.py)
* **Enhancements**:
  * Created `get_modality_key` to map config modality names (e.g. `"S2_BEN"`, `"S1"`) to dataloader batch dict keys (e.g. `'s2'`, `'s1'`).
  * Integrated it into the evaluation loops and training loops to prevent systematic `KeyError` crashes when modality lists are re-ordered or swapped in the configuration.

## 3. Distributed Training (DDP) all_reduce
* **File**: [src/losses/sigreg.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/src/losses/sigreg.py)
* **Fix**:
  * Replaced the missing PyTorch package import `torch.distributed.nn.all_reduce` with standard in-place `torch.distributed.all_reduce`.

## 4. SIGReg Loss State & Random Projections
* **Files**: 
  * [src/models/copfm_retrieval.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/src/models/copfm_retrieval.py)
  * [src/losses/sigreg.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/src/losses/sigreg.py)
  * [test_sigreg.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/test_sigreg.py)
* **Fixes**:
  * Persistent instantiation of `SIGRegLoss` modules in `CopFMRetrieval`'s constructor. This preserves the random generator's `global_step` state across training iterations, ensuring varying random projection angles instead of resetting the seed every step.
  * Corrected test script mock dictionary keys to match `r_cross_a`, `r_cross_b`, `r_uni_a`, and `r_uni_b`.

## 5. Architectural JEPA Shortcut Fix & EMA Target Encoder
* **Files**: 
  * [src/models/backbone.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/src/models/backbone.py)
  * [src/models/copfm_retrieval.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/src/models/copfm_retrieval.py)
  * [train.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/train.py)
* **Fixes**:
  * Modified the backbone forward pass to accept an optional `mask`. When masking is enabled, it slices patch and position embeddings *before* running ViT blocks so that the context encoder only processes visible patches. This prevents information leakage through self-attention layers.
  * Added a separate target encoder (`target_backbone`) updated via Exponential Moving Average (EMA).
  * Cleared forward hook activation tokens at the end of the pass to prevent GPU memory leaks.

## 6. Vectorized Masking
* **File**: [src/utils/masking.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/src/utils/masking.py)
* **Enhancement**:
  * Replaced sequential loops over GPU tensors with PyTorch boolean indexing (`tokens[~mask]` / `tokens[mask]`) to speed up token masking.

## 7. FAISS Index Robustness & Self-Match Filtering
* **Files**: 
  * [src/retrieval/faiss_index.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/src/retrieval/faiss_index.py)
  * [src/evaluation/retrieval_eval.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/src/evaluation/retrieval_eval.py)
* **Enhancements**:
  * Centroids count fallback: checks if the database size is smaller than the requested IVF centroids count (`n_list`) and falls back to `IndexFlatIP`.
  * Index `-1` check: maps invalid search indexes to zero-vector dummy labels to avoid list index type errors.
  * Self-Match Filtering: queries $K+1$ nearest neighbors, uses a boolean mask to filter out index matches that equal the query image's own index, and slices the first $K$ items.

## 8. Real Datasets & Spatial/Temporal Positional Metadata
* **Files**: 
  * [src/datasets/ben14k.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/src/datasets/ben14k.py)
  * [src/datasets/cbrsir.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/src/datasets/cbrsir.py)
  * [src/datasets/dsrsid.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/src/datasets/dsrsid.py)
* **Enhancements**:
  * Integrates real BigEarthNet `.tif` image loading, multi-label indices matching, and file validation using `metadata.parquet`.
  * Spatial Metadata: Projects UTM tile coordinates to Latitude/Longitude (WGS84) center coordinates using `rasterio.warp.transform`.
  * Temporal Metadata: Extracts absolute acquisition dates from filenames and converts them to Day of Year (doy).
  * Returns `meta_s1`/`meta_s2` tensors formatted as `[lon, lat, doy, area]` to leverage the pre-trained absolute time Fourier positional embeddings of Copernicus-FM.
  * Implements warning logs and realistic synthetic fallback branches for all datasets if folders are not present.

## 9. Namespace Collision Fixes
* **File**: [src/models/backbone.py](file:///c:/Users/SHAHZEB%20ALI/OneDrive/Desktop/isro_bah/copfm_retrieval/src/models/backbone.py)
* **Fix**:
  * Isolates the reference repository `src` folder imports by temporarily hiding the workspace root from `sys.path` and pruning cached `src` packages in `sys.modules`. This prevents workspace files from conflicting with Copernicus-FM imports.
