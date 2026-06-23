## File: src/losses/nce.py
---
### What Suryansh did [ in indepth bullet points with grounded reality ]
- Created the initial structure and functions for `info_nce_loss`, `cosine_alignment_loss`, and `total_retrieval_loss`.
- Included descriptive docstrings and inline comments (e.g., "Symmetric InfoNCE (NT-Xent) loss for cross-modal alignment.", "# Similarity matrix (B, B) — symmetric").
- **Crucial Mathematical Flaw:** Omitted L2-normalization of `embeddings_a` and `embeddings_b` before calculating their similarities. In `info_nce_loss`, the dot product `(embeddings_a @ embeddings_b.T)` is computed directly. Without normalization, this is just a scaled dot-product, not cosine similarity, which will cause instability and incorrect InfoNCE behavior. Similarly, in `cosine_alignment_loss`, the `cos_sim` uses unnormalized tensors.

### What Aarushi di did [ indepth bullet points with grounded reality ]
- Identified and fixed the mathematical bug in the loss computations.
- Added `F.normalize(embeddings_a, dim=-1)` and `F.normalize(embeddings_b, dim=-1)` at the start of `info_nce_loss`.
- Added `a = F.normalize(embeddings_a, dim=-1)` and `b = F.normalize(embeddings_b, dim=-1)` at the start of `cosine_alignment_loss`.
- Ensured that InfoNCE and cosine alignment properly operate on the unit hypersphere (true cosine similarity).
- **Drawback:** Stripped out all of the helpful docstrings and inline comments present in the original version, making the code undocumented and harder to read.

### What Shahzeb did [ indepth bullet points with grounded reality ]
- Maintained the exact same codebase as Suryansh's version. The logic and content are identical.
- Kept the docstrings and comments intact.
- **Drawback:** Failed to notice or fix the missing normalization bug. The mathematical flaw remains completely unfixed in his version.

### Conclusion: What to keep, what to remove and from whoms....
- **What to keep:** The correct mathematical logic (`F.normalize` operations) from **Aarushi di** to ensure true cosine similarity. The docstrings and explanatory inline comments from **Suryansh/Shahzeb** to maintain readable documentation.
- **What to remove:** The unnormalized, erroneous dot product calculations from **Suryansh/Shahzeb**. The undocumented, comment-stripped structure from **Aarushi di**.
- **Final Action:** Merge Aarushi di's `F.normalize` mathematical fixes into Suryansh/Shahzeb's fully-commented codebase.



## File: src/models/copfm_retrieval.py
---
### What Suryansh did [ in indepth bullet points with grounded reality ]
- **Correct Architectural Core**: Correctly omitted the EMA `target_backbone` and `update_target_ema` functions. He strictly relied on a single shared online backbone to process both context and targets, which is mathematically correct for CR-JEPA since it depends entirely on `SIGRegLoss` (applied to retrieval projections) to prevent representation collapse rather than a momentum encoder.
- **Information Leakage Flaw**: Failed to prevent information leakage in the forward pass. By passing the full, unmasked image batch through `self.backbone` to generate `tokens_a` and *then* slicing `vis_a` and `tgt_a` via `get_visible_tokens`, the self-attention mechanism in the ViT backbone allows visible patches to attend to target patches. The context "sees" the target before the predictor even runs, breaking the core predictive learning objective.

### What Aarushi di did [ indepth bullet points with grounded reality ]
- **Incorrect Architectural Core**: Incorrectly introduced `target_backbone = copy.deepcopy(self.backbone)` and the `update_target_ema` method. This violates CR-JEPA's design, which explicitly does not use an EMA target encoder.
- **Information Leakage Flaw**: Suffers from the exact same information leakage flaw as Suryansh. The online backbone processes the full `batch_a` without a mask, meaning the visible tokens extracted from it have already attended to the masked tokens.
- **Added SIGRegLoss**: Correctly recognized the need for `SIGRegLoss` (which was missing in Suryansh's `__init__`), though applied it in a basic manner.

### What Shahzeb did [ indepth bullet points with grounded reality ]
- **Incorrect Architectural Core**: Like Aarushi di, incorrectly introduced the EMA `target_backbone` and EMA update logic, which fundamentally contradicts CR-JEPA.
- **Correct Information Leakage Prevention**: Correctly and explicitly passed the mask into the context backbone (`vis_a = self.backbone(..., mask=mask_a)`). This enforces that the ViT only processes visible patches, mathematically isolating them from the target patches during self-attention and properly preventing leakage.
- **Correct Retrieval Head Full Pass**: Realized that since the context pass was masked, the retrieval heads still needed a full token sequence. He correctly added a second unmasked pass (`full_context_a = self.backbone(...)`) to get the full embeddings for the retrieval projections.
- **Persistent SIGRegLoss**: Instantiated persistent `SIGRegLoss` modules for each retrieval head, which is a sound engineering choice to preserve global step tracking to prevent static random projections.

### Conclusion: What to keep, what to remove and from whoms....
- **Remove**: All EMA logic (`target_backbone`, `update_target_ema`, and `self.ema_decay`) from Aarushi di and Shahzeb's implementations. They are mathematically wrong for CR-JEPA.
- **Keep from Suryansh**: The single shared online backbone approach for generating both visible context and target tokens. 
- **Keep from Shahzeb**: The leakage prevention mechanism. The online backbone MUST be called with the `mask` argument (`vis_a = self.backbone(..., mask=mask_a)`) to generate the strictly isolated context tokens.
- **Keep from Shahzeb**: The two-pass strategy. Generate targets from the shared backbone on the full image (typically using `torch.no_grad()`), generate context using the mask, and then run a full unmasked pass (`full_context_a = self.backbone(...)`) to generate the complete sequences needed for `phi_cross` and `phi_uni`.
- **Keep from Shahzeb**: The persistent `SIGRegLoss` modules defined in `__init__` and their application to the retrieval projections, as this is the true mechanism CR-JEPA uses to prevent representation collapse without an EMA encoder.



## File: src/utils/masking.py
---
### What Suryansh did [ in indepth bullet points with grounded reality ]
- **Sequential Looping**: Implemented token extraction using standard Python `for` loops to iterate over the batch dimension (`for i in range(B):`).
- **List Appending**: Accumulated the masked and visible tokens into Python lists (`visible_tokens.append(vis)`) and then relied on `torch.stack` to merge them back into a tensor.
- **Performance Impact (Grounded Reality)**: This is an anti-pattern in PyTorch. Iterating through batches and using Python lists drastically slows down execution by breaking vectorization, causing CPU-GPU sync bottlenecks, and failing to utilize the parallel processing power of the GPU.

### What Aarushi di did [ indepth bullet points with grounded reality ]
- **Vectorized Boolean Indexing**: Completely eliminated the Python `for` loops and list operations in favor of native PyTorch boolean indexing (`tokens[~mask]`).
- **Efficient Reshaping**: Used `.view(B, -1, D)` to seamlessly reshape the indexed tensor back to the desired 3D structure `(Batch, Tokens, Dim)`.
- **Performance Impact (Grounded Reality)**: This approach is significantly faster and highly optimized. By keeping the operation entirely within PyTorch's backend (C++/CUDA), it maximizes GPU throughput and provides a massive speedup during JEPA-style training.

### What Shahzeb did [ indepth bullet points with grounded reality ]
- **Vectorized Boolean Indexing**: Implemented the exact same optimal, loop-free boolean indexing logic as Aarushi di (`tokens[~mask].view(B, -1, D)` and `tokens[mask].view(B, -1, D)`).
- **Identical Implementation**: The file logic is completely identical to Aarushi di's version, demonstrating a shared understanding of PyTorch optimization.
- **Performance Impact (Grounded Reality)**: Matches the optimized throughput of Aarushi di's code, utilizing PyTorch's fast C-level operations to bypass the Python-level overhead seen in Suryansh's code.

### Conclusion: What to keep, what to remove and from whoms....
**What to remove**: Completely discard Suryansh's implementation. The `for` loop and `torch.stack` accumulation method will severely bottleneck training speeds. 
**What to keep**: Keep the optimized PyTorch boolean indexing implementation (`tokens[~mask].view(B, -1, D)` and `tokens[mask].view(B, -1, D)`).
**From whoms**: You can use either Aarushi di's or Shahzeb's versions, as they both correctly recognized the performance bottleneck and wrote identical, highly optimized vectorized code.



## File: src/datasets/dsrsid.py
---
### What Suryansh did [ in indepth bullet points with grounded reality ]
* Uses `pan_meta` and `ms_meta` as keys for the metadata output dictionary.
* Correctly estimates the patch area metadata as `0.262` km². This precisely aligns with the grounded reality of the Gaofen-1 satellite in the DSRSID dataset: a 256x256 panchromatic image at 2m/pixel spatial resolution represents a 512m x 512m ground footprint, which equals 262,144 m² or ~0.262 km².
* Assumes the multispectral (MS) and panchromatic (PAN) imagery raw inputs are 8-bit integers (values in `[0, 255]`), properly scaling them to `[0, 1]` before normalization.
* Uses generic ImageNet-like means and standard deviations for multispectral normalization (e.g., `[0.485, 0.456, 0.406, 0.5]`), which may not accurately represent the true spectral distribution of the Gaofen-1 MS sensor.

### What Aarushi di did [ indepth bullet points with grounded reality ]
* Renames the metadata keys to `meta_pan` and `meta_ms`, creating an inconsistency with Suryansh's pipeline.
* Hardcodes a generic metadata patch area of `1.44` km², which is factually incorrect for the standard 256x256 (2m/px) patches in the DSRSID dataset.
* Assumes multispectral imagery is provided as 16-bit integers (`uint16`) with surface reflectance values typically divided by `10000.0`.
* Applies placeholder global statistics for image normalizations (e.g., mean `100.0`, std `50.0` for PAN; mean `0.1`, std `0.05` for MS), which lack statistical grounding.
* Hardcodes geographical coordinates to a location in India (`lon=78.96`, `lat=20.59`), which is a broad assumption that may not apply to all Gaofen-1 captured DSRSID samples.

### What Shahzeb did [ indepth bullet points with grounded reality ]
* Copied Aarushi di's implementation exactly. The file is a byte-for-byte duplicate (2275 bytes, 68 lines) of Aarushi di's version.
* Inherits the same incorrect patch area of `1.44` km² for the Gaofen-1 patches.
* Inherits the `uint16` / `10000.0` reflectance scaling assumption for MS images.
* Inherits the generic placeholder statistics for PAN and MS image normalization.
* Inherits the hardcoded geographic coordinates for India.

### Conclusion: What to keep, what to remove and from whoms....
* **Keep from Suryansh**: The mathematically correct patch area estimation (`0.262` km²) based on Gaofen-1's true physical properties, as well as the dynamic [0, 1] scaling logic for pixel value ranges.
* **Keep from Aarushi di**: The assumption of `uint16` / `10000.0` for raw MS images can be retained *only* if the specific DSRSID distribution you are working with provides 16-bit L2A data instead of 8-bit imagery.
* **Remove**: 
  * Shahzeb's file should be completely removed/ignored as it is a redundant duplicate.
  * Remove the incorrect `1.44` km² patch area from Aarushi di's implementation.
  * Remove the generic ImageNet normalization statistics and compute the actual mean/std over the DSRSID dataset.
* **Final Verdict**: Base the final `dsrsid.py` on **Suryansh's** physically accurate footprint and structure, standardize the dictionary keys, and incorporate Aarushi di's `uint16` scaling if your specific data format demands it.



## File: src/retrieval/faiss_index.py
---
### What Suryansh did [ in indepth bullet points with grounded reality ]
- **Basic Index Implementation**: Implemented a fundamental `FAISSRetrievalIndex` supporting `IVFFlat`, `IVFPQ`, and `Flat` (as else fallback) algorithms.
- **Missing Data Preprocessing**: Did not use `np.ascontiguousarray` or `faiss.normalize_L2`, which can lead to errors with Faiss (especially inner product metric which requires normalized vectors for cosine similarity).
- **Bug in `search` method**: Implemented labels retrieval using `[self.labels[idx] for idx in indices]`. Since Faiss returns `indices` as a 2D array, iterating this way without an inner loop or proper numpy broadcasting causes issues. It also fails to account for `idx == -1` (when Faiss can't find enough neighbors).
- **Missing Fallbacks**: No checks for dataset size being smaller than the number of IVF centroids (`n_list`), which will crash Faiss during training.

### What Aarushi di did [ indepth bullet points with grounded reality ]
- **Data Normalization & Formatting**: Added `np.ascontiguousarray` and `faiss.normalize_L2` in both `build()` and `search()` methods, preventing C++ Faiss memory access issues and ensuring correct cosine similarity behavior.
- **Robustness Fallback & Assertions**: Added `use_flat = len(embeddings) < self.n_list` to automatically fallback to a `Flat` index if the dataset is too small. Also added an assertion `D % 8 == 0` for `IVFPQ` since it expects dimensionality divisible by the sub-quantizer count.
- **Handled 2D Indices & `-1` Indices**: Corrected the label retrieval to iterate over rows: `[[self.labels[idx] if idx >= 0 else None for idx in row] for row in indices]`. This safely handles Faiss returning `-1` for empty slots without index-out-of-bounds errors.

### What Shahzeb did [ indepth bullet points with grounded reality ]
- **Self-Match Filtering**: Introduced `query_indices` parameter in `search()` to filter out self-matches (e.g., when a database element is used as a query). Uses `fetch_k = k + 1` to ensure `k` neighbors are returned even if one is dropped.
- **Explicit Fallback Warning**: Handled the small dataset fallback (`N < self.n_list`) explicitly with a printed warning: `"Warning: Database size {N} is smaller than IVF centroids... Falling back"`, which is great for debugging.
- **Advanced `-1` Index Handling**: When `idx == -1`, checks if `self.labels` is a numpy array. If so, it appends a zeroed array of the exact same shape as a label (`np.zeros_like(self.labels[0])`). Otherwise, it appends `None`.
- **Missed Normalization**: Did not include the crucial `faiss.normalize_L2` and `np.ascontiguousarray` calls in the `build()` and `search()` methods.

### Conclusion: What to keep, what to remove and from whoms....
- **Keep from Aarushi di**: The strict data preprocessing (`np.ascontiguousarray` and `faiss.normalize_L2`) in both `build` and `search`. Also, keep the `assert D % 8 == 0` for `IVFPQ`.
- **Keep from Shahzeb**: The self-match filtering logic (`query_indices` handling with `fetch_k = k + 1`). Keep the fallback warning message when `N < self.n_list`. Keep the numpy-compatible `-1` handling (`np.zeros_like`).
- **Remove (from all/Suryansh)**: The basic `search` label retrieval logic from Suryansh. Remove Shahzeb's missing normalization vulnerability. Remove Aarushi's silent fallback in favor of Shahzeb's explicit warning fallback.
- **Final Action**: Merge Aarushi di's robust normalization and type-casting with Shahzeb's query-index filtering and detailed warnings to create the perfect `FAISSRetrievalIndex`.



## File: src/models/backbone.py
---
### What Suryansh did [ in indepth bullet points with grounded reality ]
- His `forward` method in `CopFMBackbone` accepts arguments like `image`, `wavelengths`, `bandwidths`, `return_patch_tokens`, and `meta_info`, but entirely lacks support for a `mask` parameter.
- The method processes the full, unmasked image by passing it directly to the base model: `_ = self.base_model(x=image, ...)`.
- It relies on a PyTorch forward hook (`self.base_model.norm.register_forward_hook`) to extract token representations after they have already passed through the entire network.
- **Critical Flaw**: Because the full token sequence passes through the ViT's global self-attention layers simultaneously, information freely leaks between the theoretically "masked" and "visible" regions. 
- This architecture fundamentally violates CR-JEPA's equation `z_V = g(f(x; V))`, which strictly mandates that the shared transformer trunk `g` must only process the selected subset of tokens (`V`).

### What Aarushi di did [ indepth bullet points with grounded reality ]
- Upgraded the `forward` method to natively accept a `mask` argument (`mask: torch.Tensor = None`).
- When a mask is provided, the method skips the standard model call and manually steps through the embedding process, starting with `self.base_model.patch_embed_spectral`.
- It dynamically computes and adds all required positional embeddings (including coordinate, scale, and time embeddings) to the patch tokens.
- **Crucial Step**: Introduced the logic `x_vis = x[~mask].view(B, -1, embed_dim)` to explicitly filter out masked tokens *before* appending the CLS token.
- Only these selected `x_vis` tokens are iterated through the transformer layers (`for block in self.base_model.blocks: x_vis = block(x_vis)`).
- This implementation perfectly fulfills the CR-JEPA requirement `z_V = g(f(x; V))` by guaranteeing that the self-attention blocks never see the masked tokens, thereby eliminating any chance of information leakage.

### What Shahzeb did [ indepth bullet points with grounded reality ]
- Shahzeb's implementation is entirely identical to Aarushi di's implementation.
- He incorporated the same `mask` argument handling in the `forward` method.
- Implemented the identical custom forward pass logic, extracting patch embeddings and applying the meta-information positional encodings dynamically.
- Utilized the exact same masking operation `x_vis = x[~mask]` to drop masked tokens prior to entering the ViT blocks.
- Fully aligns with the CR-JEPA design, preserving the mathematical integrity of the latent predictive learning process by ensuring strict independence between visible and masked tokens.

### Conclusion: What to keep, what to remove and from whoms....
- **What to keep**: The custom masked forward pass logic that actively subsets and drops tokens before the ViT self-attention blocks using `x[~mask]`. This is mathematically correct and essential for fulfilling CR-JEPA's `z_V = g(f(x; V))` formulation without information leakage.
- **What to remove**: Suryansh's hook-based, full-image processing forward pass. Passing the entire image through the transformer layers causes attention-based information leakage and breaks the core predictive logic of the architecture.
- **From whoms**: Keep the backbone implementation from **Aarushi di** (or **Shahzeb**, since they wrote the exact same code). Completely remove Suryansh's version.



## File: src/datasets/cbrsir.py
---
### What Suryansh did
*   **RGB Generation and Normalization**: Generated synthetic RGB data in the realistic 8-bit integer range `[0, 255]` (`torch.rand() * 255.0`). The `normalize_rgb` method contains robust logic that checks if the maximum value exceeds 1.0 before dividing by 255.0, making the pipeline adaptable to both raw 8-bit images and pre-scaled float inputs.
*   **SAR Generation and Normalization**: Generated synthetic SAR data directly in the logarithmic dB scale using a normal distribution with statistically accurate Sentinel-1 GRD backscatter statistics (mean = -12.548, std = 5.257). 
*   **SAR Clamping**: Implemented a mathematically sound dB clamp `[-40.0, 0.0]` in `normalize_sar`. In real-world SAR processing, values below -40 dB are typically system noise floors (or thermal noise in shadow regions), and values above 0 dB are very rare, extremely strong scatterers (e.g., corner reflectors in urban areas). Clamping these bounds prevents extreme outliers from breaking neural network gradients.
*   **Metadata Coordinates**: Used bounding box coordinates `[113.9, 22.5]` which corresponds geographically to southern China / Hong Kong, rather than ISRO's primary area of interest (India).
*   **Patch Area / Stats**: Handled patch area meta values `0.0655` which are arbitrary but present.

### What Aarushi di did
*   **RGB Generation and Normalization**: Generated synthetic RGB data directly in the `[0, 1]` range. The `normalize_rgb` function assumes the input is strictly pre-scaled to `[0, 1]` and lacks defensive checks, which could lead to massive gradient explosions if fed raw 8-bit `[0, 255]` images from actual dataset files.
*   **SAR Generation and Normalization**: Generated SAR data in a linear scale using a uniform distribution `[0, 1]` (`torch.rand()`), and then dynamically converted it to dB via `10 * torch.log10(sar_linear + 1e-10)`. While dynamically converting linear to dB is a common reality when ingesting raw Geotiffs, uniformly distributed linear noise does not yield realistic SAR backscatter profiles when converted to dB.
*   **Placeholder Statistics**: Used approximate placeholder values for SAR statistics (mean = -12.0, std = 5.0) instead of accurately derived moments. Missing the critical `[-40, 0]` clamp, which leaves the network vulnerable to extreme `log(0)` approximations (bounded only by the `1e-10` epsilon resulting in deep negatives like -100 dB).
*   **Metadata Coordinates**: Intelligently tailored the metadata values to the ISRO context by generating coordinates corresponding to India (`lon, lat = 78.96, 20.59`), increasing geographic relevance for the CopFM-Retrieval task.

### What Shahzeb did
*   **Identical Implementation to Aarushi di**: The file provided by Shahzeb is a direct, exact copy of Aarushi di's version (matching byte-for-byte at 2488 bytes, 70 lines).
*   **RGB Handling**: Identical `[0, 1]` implicit assumption without 8-bit `[0, 255]` defensive normalization checks.
*   **SAR Handling**: Identical uniform linear generation converted to dB, missing outlier clamping, and using the exact same placeholder statistics (-12.0, 5.0).
*   **Metadata**: Identical usage of Indian geographic coordinates `[78.96, 20.59]` for metadata tensors.

### Conclusion: What to keep, what to remove and from whoms....
*   **Keep from Suryansh**: 
    *   The robust `normalize_rgb` defensive scaling logic (`rgb_raw / 255.0 if rgb_raw.max() > 1.0 else rgb_raw`), which safely handles reality where images can be either 8-bit `[0, 255]` or float `[0, 1]`.
    *   The SAR processing logic: The statistically accurate distribution (mean=-12.548, std=5.257) and specifically the `torch.clamp(sar_db, min=-40.0, max=0.0)` which is critical for physical SAR validity to remove shadow noise floors and extreme bright outliers.
*   **Keep from Aarushi di (and Shahzeb)**: 
    *   The dynamically tailored ISRO geographical coordinates in the metadata (`lon, lat = 78.96, 20.59`) to keep the dataset grounded in the context of the Indian subcontinent.
*   **Remove**: 
    *   Aarushi di's and Shahzeb's naive `[0, 1]` uniform SAR generation and uncalibrated linear-to-dB log transformations, which create unrealistic statistical distributions without clamping.
    *   Aarushi di's and Shahzeb's rigid RGB normalization that breaks if fed raw 8-bit PNG/TIFF data.
    *   Suryansh's irrelevant metadata coordinates (China/Hong Kong bounds).
*   **Final Action**: Merge Suryansh's robust normalization and physically accurate SAR clamping pipelines with Aarushi di's geographically relevant metadata implementation. Shahzeb's file can be discarded as a duplicate of Aarushi di's.



## File: train.py
---
### What Suryansh did
* **Modality Key Mapping**: Used a hardcoded if/elif block inside `train.py` to map dataset modality combinations (e.g., forcing `'s1'`/`'s2'`, `'rgb'`/`'sar'`). Mapped metadata by appending the `_meta` suffix (i.e., `mod_a_key + '_meta'`).
* **Target Path**: Accurately respected the non-EMA target path of CR-JEPA. He purposefully did not call `model.update_target_ema()` in the training loop, ensuring the target representations are not derived from exponential moving average updates.
* **Validation Loop**: Reused the pre-training `forward_train()` method with random block masking in the validation loop, meaning it merely computed reconstruction/feature loss rather than assessing actual cross-modal retrieval performance.
* **Scheduler**: Stepped the Cosine scheduler once per epoch based on epoch-level calculation.

### What Aarushi di did
* **Modality Key Mapping**: Refactored the key extraction to dynamically use `get_modality_key(dataset_name, config['data']['modality_a'])` from `src.utils.key_mapping`. Adapted metadata mapping to use a prefix (`'meta_' + mod_a_key`).
* **Target Path**: Disrespected the non-EMA target path by adding a `model.update_target_ema()` call inside the training loop (after `optimizer.step()`), effectively treating it like a standard BYOL/I-JEPA model instead of maintaining CR-JEPA's non-EMA architecture.
* **Validation Loop**: Correctly rewrote the validation loop to extract `e_a` and `e_b` embeddings using `model.get_retrieval_embedding(..., mode='cross')` and computed proper validation metrics via `info_nce_loss`.
* **Scheduler**: Upgraded the scheduler to calculate `total_steps` as `epochs * len(train_loader)` and stepped it iteratively per batch. Included additional tensor conversion utilities (`_to_device`, `_loss_to_float`).

### What Shahzeb did
* **Modality Key Mapping**: Adopted the same modular `get_modality_key` and `'meta_' + mod_a_key` mapping design as Aarushi di.
* **Target Path**: Like Aarushi di, he failed to respect the non-EMA target path by erroneously introducing `model.update_target_ema()` into the epoch loop.
* **Validation Loop**: Reverted to Suryansh's approach for the validation loop, invoking `forward_train` to generate training losses on validation data rather than measuring retrieval quality.
* **Scheduler**: Kept the scheduler stepping outside the batch loop (per-epoch) like Suryansh.

### Conclusion: What to keep, what to remove and from whoms....
* **Keep from Suryansh**: The omission of `model.update_target_ema()` in the training loop. This is the correct behavior to respect the non-EMA target path intrinsic to CR-JEPA.
* **Keep from Aarushi di**: The modular `get_modality_key` utility and `meta_` prefix for modality mapping. The entire validation block evaluating cross-modal alignment via `get_retrieval_embedding` and `info_nce_loss`. Her per-step learning rate scheduler logic.
* **Remove from Suryansh**: The rigid if/elif modality string mapping, and the flawed validation implementation.
* **Remove from Aarushi di & Shahzeb**: The calls to `model.update_target_ema()`, which violate the architecture's non-EMA constraint.
* **Remove from Shahzeb**: The broken validation loop inherited from Suryansh.



## File: src/evaluation/retrieval_eval.py
---
### What Suryansh did
- **Hardcoded Modality Branches:** Suryansh implemented hardcoded `if-elif` statements (`'s1'`, `'rgb'`, `'pan'`) to extract the correct modality tensor. This violates the CR-JEPA mathematical core's goal of dynamic scaling because it strictly couples the evaluation to specific known strings instead of treating modalities dynamically.
- **Index-Based Self-Match Filtering:** To exclude trivial self-matches in same-modal retrieval (as mandated by Section 3.8 of CR-JEPA), Suryansh filters out matches using `mask = ret_idx != i`.
  - *Pros:* This executes in $O(1)$ time per query, perfectly leveraging FAISS for low latency, aligning with ISRO PS-11 requirements.
  - *Cons:* It strictly assumes that the query dataset and gallery dataset are identical and perfectly ordered row-for-row. If the datasets are shuffled or query is a subset, this logic breaks entirely and filters the wrong embeddings.
- **Missing Identifiers:** His extraction function only returns `embeddings` and `labels`, completely discarding the semantic IDs (`pair_id`, `sample_id`), making robust tracking impossible.

### What Aarushi di did
- **Dynamic Dataset Inference & Key Mapping:** She eliminated the hardcoded modality if-statements by dynamically inferring the dataset name and utilizing a `get_modality_key` utility. This aligns perfectly with CopFM's scalable, sensor-agnostic philosophy.
- **Semantic ID Extraction:** She dynamically extracts `pair_id` or `sample_id` and returns an `all_ids` list alongside embeddings and labels. This accurately grounds the retrieval results to the dataset metadata.
- **ID-Based Self-Match Filtering:** She implemented robust filtering by checking `q_id in gallery_ids` and finding its index via `gallery_ids.index(q_id)`. 
  - *Pros:* This mathematically guarantees that self-matches are excluded correctly regardless of query/gallery ordering, aligning with CR-JEPA's "same indexed sample" constraint.
  - *Cons:* Using `.index()` inside a python for-loop over `query_embs` creates an $O(N \times M)$ bottleneck. For large datasets like DSRSID (80,000 images), this will result in massive evaluation latency, severely violating the ISRO PS-11 strict low-latency requirements.

### What Shahzeb did
- **Identical Implementation:** Shahzeb's code is byte-for-byte identical to Aarushi di's version. He implemented the exact same dynamic key mapping, ID extraction, and $O(N \times M)$ ID-based filtering mechanism.
- *Grounded Reality:* Because the implementations are identical, Shahzeb inherits all the same strengths (dynamic modality keys, robust ID extraction) and the exact same critical performance flaw (the list `.index()` search in the FAISS retrieval loop).

### Conclusion: What to keep, what to remove and from whoms....
- **Keep from Aarushi di / Shahzeb:**
  - The dynamic dataset inference (`dataset_name = dataloader.dataset.dataset_name`).
  - The use of `get_modality_key` instead of hardcoded strings to fetch the correct image tensor.
  - The extraction of `pair_id` / `sample_id` (`all_ids`) to ensure robust metadata tracking.
- **Remove from Aarushi di / Shahzeb:**
  - The $O(N \times M)$ `.index()` list lookup for self-match filtering (`self_idx = gallery_ids.index(q_id)`). This will catastrophically throttle retrieval evaluation speed on massive Earth observation datasets.
- **Keep from Suryansh:**
  - The fast, low-latency intent of his index masking concept, but it must be mathematically upgraded.
- **Final Recommendation for the Merged Code:**
  - Merge Aarushi di's dynamic extraction and ID tracking.
  - Fix the self-match filtering bottleneck by pre-computing a fast lookup dictionary (Hash Map) for `gallery_ids` before the evaluation loop (e.g., `gallery_id_to_idx = {gid: idx for idx, gid in enumerate(gallery_ids)}`). 
  - During the retrieval loop, perform an $O(1)$ dictionary lookup `self_idx = gallery_id_to_idx.get(q_id)` to filter the FAISS indices. This perfectly marries Aarushi di's mathematical robustness with Suryansh's low-latency execution, completely satisfying both the SOTA CR-JEPA paper constraints and the ISRO PS-11 hackathon speed requirements.



## File: src/datasets/ben14k.py
---
### What Suryansh did [ in indepth bullet points with grounded reality ]
*   **Dummy Data Implementation:** Suryansh entirely skipped implementing the actual data loading pipeline. His `__getitem__` generates random synthetic data tensors rather than reading from disk or a `.parquet` file.
*   **Correct S2 Statistics:** Despite the dummy data loading, Suryansh correctly implemented the empirically derived normalization statistics (Mean and Standard Deviation) for the Sentinel-2 12-band data. These statistics match the official BigEarthNet dataset profile when scaled to `[0, 1]` reflectance.
*   **Correct Understanding of S1 Scale:** Suryansh accurately documented and implemented S1 normalization under the premise that BigEarthNet-MM Sentinel-1 patches are natively distributed in the decibel (dB) scale. He applied a standard Z-score normalization `(s1_db - mean) / std` without attempting to erroneously log-transform the data.
*   **Incomplete Metadata/Labels:** Extracted geographical metadata (lat/lon/DOY) and multi-labels are completely random and hardcoded, providing no real mapping to the CLC2018 19-class nomenclature.

### What Aarushi di did [ indepth bullet points with grounded reality ]
*   **Robust Data Loading Pipeline:** Aarushi di implemented a fully functional data loader that uses `pandas` to read the `metadata.parquet` file, filters by split, and correctly falls back to synthetic generation if the parquet or patches are missing.
*   **Advanced Feature Extraction:** She successfully utilized `rasterio` and regex to dynamically extract realistic geospatial (Latitude, Longitude) and temporal (Day of Year) metadata from the actual imagery patches and filenames. 
*   **Proper Label Mapping:** She implemented a proper string-to-multihot label mapping utilizing the standard BigEarthNet 19-class nomenclature (`CLASS_NAMES`), parsing the string list from the parquet file into a `19D` tensor.
*   **Critical Error with S1 Normalization:** Aarushi di incorrectly assumed S1 data was in a linear scale. She applied `10 * torch.log10(s1_linear + 1e-10)` to the data. Since BigEarthNet S1 is already provided in dB, applying a logarithm to a logarithmic value severely corrupts the SAR data (potentially resulting in NaNs due to negative dB values).
*   **Critical Error with S2 Statistics:** She used entirely fake placeholder statistics `[0.1] * 12` for both the S2 mean and standard deviation, destroying the radiometric calibration during S2 normalization.
*   **Questionable Band Padding Hack:** She implemented a hacky 10-to-12 band duplication logic for S2 (e.g., `B01 <- B02`, `B09 <- B8A`). BigEarthNet inherently provides 12 bands. This logic masks underlying data corruption rather than solving it.

### What Shahzeb did [ indepth bullet points with grounded reality ]
*   **Zero Original Contribution:** Shahzeb's file is 100% identical to Aarushi di's file. The byte count, line count, and code structure are completely verbatim.
*   **Plagiarized Errors:** By copy-pasting Aarushi di's code, Shahzeb imported all of her critical errors, including the corrupted S1 log-transform on dB values, the fake `[0.1]` S2 statistics, and the 10-band replication hack.

### Conclusion: What to keep, what to remove and from whoms....
**What to Keep:**
1.  **From Aarushi di:** Keep the entirety of her `metadata.parquet` parsing logic, the robust `__getitem__` file reading with `rasterio`, the `lon/lat/DOY` metadata extraction, the 19-class label encoding, and the synthetic fallback mechanism.
2.  **From Suryansh:** Keep his accurate S2 Mean and Standard Deviation tensors, and his accurate S1 Z-score normalization logic that correctly recognizes S1 is already in the dB scale.

**What to Remove:**
1.  **From Aarushi di:** Remove the `10 * torch.log10` S1 conversion, remove the fake `[0.1]*12` S2 statistics, and remove the arbitrary 10-to-12 S2 band duplication hack.
2.  **From Suryansh:** Remove his hardcoded dummy sample generation and dummy random labels.
3.  **From Shahzeb:** Discard entirely, as it is a pure copy-paste of Aarushi di's work with no unique contributions.

**Final Verdict:** Merge Suryansh's radiometric understanding and official statistics into Aarushi di's robust dataloading and geospatial extraction pipeline.



## File: src/models/predictor.py
---
### What Suryansh did [ in indepth bullet points with grounded reality ]
- **Correct Architectural Alignment**: Correctly implemented the predictor logic by expanding the `mask_token` and adding the corresponding `predictor_pos_embed` based on the masked positions.
- **Strict Adherence to the Paper**: Perfectly aligned with the CR-JEPA paper, which specifies the use of learnable mask queries and target-position embeddings for fixed 224x224 static images (196 tokens max). Did not introduce any hallucinated interpolation methods.
- **Suboptimal but Correct Logic**: Used a `for` loop over the batch size to extract `pos_emb` and construct `queries`. While this Python `for` loop is not vectorized and is likely slower during training, it is factually and mathematically correct.

### What Aarushi di did [ indepth bullet points with grounded reality ]
- **Good Vectorization**: Improved upon Suryansh's `for` loops by using tensor expansion and boolean indexing (`mask_tokens[mask].reshape(...)`) to create the queries in a fully vectorized manner, which is much better for execution speed.
- **Hallucinated Concept (1D Linear Interpolation)**: Introduced a dynamic `_get_pos_embed` function that uses 1D linear interpolation (`F.interpolate(..., mode='linear')`) if the number of patches exceeds `max_patches`. 
- **Outside Scope**: The CR-JEPA paper uses static 224x224 image inputs with a patch size of 16 (yielding a maximum of 196 tokens). The paper never mentions handling variable resolutions or using 1D linear interpolation for positional embeddings. Thus, she loses points for introducing hallucinated architectural complexities.

### What Shahzeb did [ indepth bullet points with grounded reality ]
- **Good Vectorization**: Like Aarushidi, successfully vectorized the extraction of masked position embeddings and creation of mask queries, eliminating the slow Python `for` loops.
- **Hallucinated Concept (2D Bicubic Interpolation)**: Introduced dynamic grid generation and 2D bicubic interpolation (`F.interpolate(..., mode='bicubic')`) by calculating `h_max` and `h_curr` to resize the positional embeddings if `N_all` is larger than the initialized parameter.
- **Outside Scope**: Completely hallucinated this feature, as the paper strictly operates on static 224x224 images and never discusses dynamic grid generation or variable resolutions for the predictor's positional embeddings.

### Conclusion: What to keep, what to remove and from whoms....
- **What to Keep**:
  - The strict, static positional embedding constraint and architectural alignment from **Suryansh** (no dynamic interpolations).
  - The vectorized masking logic and boolean indexing from **Aarushidi** or **Shahzeb** to ensure the code runs fast without Python `for` loops.
- **What to Remove**: 
  - The slow Python `for` loop from **Suryansh's** implementation.
  - The hallucinated 1D linear interpolation logic from **Aarushidi's** implementation.
  - The hallucinated 2D bicubic interpolation and dynamic grid generation logic from **Shahzeb's** implementation.
- **Final Verdict**: Combine Suryansh's correct, strict interpretation of the paper (no interpolation) with the vectorized query construction used by Aarushidi/Shahzeb.



## File: src/losses/sigreg.py
---
### What Suryansh did [ in indepth bullet points with grounded reality ]
- **Autograd-Aware Reduction**: Correctly implemented the DDP distributed `all_reduce` utility using `torch.distributed.nn.all_reduce` (functional all_reduce). This is a functional, non in-place reduction designed explicitly for autograd. This was the mathematically correct choice because `cos_mean` and `sin_mean` require gradients to compute the loss, and the functional API properly tracks these operations to compute exact distributed gradients during the backward pass.
- **Random Projection Synchronization**: Successfully implemented DDP synchronization for the random projection matrix `A`. By doing `global_step_sync = all_reduce(self.global_step.clone(), op="MAX")`, he ensured that every GPU seeds its `torch.Generator()` with exactly the same integer value, guaranteeing identically sliced matrices across the distributed setup.
- **CRITICAL MATH BUG (Loss Castration via Pre-Normalization)**: He manually standardized the batch inside the loss wrapper: `normalized_emb = (embeddings - mean) / std`. This completely castrates SIGReg! Epps-Pulley tests against $\mathcal{N}(0, I)$. By pre-normalizing the batch, the test statistic is forced to evaluate a distribution that *already* has mean 0 and variance 1, meaning the loss cannot penalize the network for variance collapse (dimensional collapse). The CR-JEPA paper explicitly states: *"We apply SIGReg to the raw pre-normalized retrieval projections"* so that the loss *forces* the network to emit mean 0, variance 1.
- **Severe Performance Bottleneck (Memory/Reallocation)**: The `sigreg_loss()` wrapper creates a completely new instance of `SIGRegLoss()` on every single forward pass. This means the `EppsPulley` object is recreated, re-allocating the `t`, `phi`, and `weights` buffers on the CPU and repeatedly dispatching them to the GPU four times per step (for `r_cross_a`, `r_cross_b`, etc.), causing massive memory fragmentation, CUDA sync overheads, and disastrously slow training.

### What Aarushi di did [ indepth bullet points with grounded reality ]
- **Buffer Reallocation Fix**: Addressed Suryansh's performance bottleneck by passing a pre-instantiated `sigreg_fn` into `total_sigreg_loss`. This reuses the same buffers for all four loss computations, dramatically improving training speed and eliminating redundant CPU-GPU tensor transfers.
- **Batch Size Guardrail**: Added a very robust check (`if embeddings.shape[0] < 2`) that returns a zero tensor with `requires_grad=True`. This prevents DDP from crashing with division-by-zero or `NaN` outputs when dealing with single-element batches at the end of an epoch.
- **Critical DDP Autograd Bug**: "Cleaned up" Suryansh's `all_reduce` by replacing it with PyTorch's base `dist.all_reduce(x)`. Because this operates **in-place**, it silently breaks the PyTorch computational graph for `cos_mean` and `sin_mean`. The backward pass will not flow backwards across GPUs, resulting in fundamentally incorrect local gradients.
- **Random Projection Desync Bug (Type Crash)**: Hardcoded `dist.ReduceOp.SUM` and forced a division by `world_size` in her custom `all_reduce`, effectively ignoring the `op="MAX"` argument passed during seed synchronization. When `global_step` (a `LongTensor`) is divided by `world_size`, it becomes a `FloatTensor`. This causes `seed = global_step_sync.item()` to yield a Python `float`, which throws a `TypeError: manual_seed expected a long, but got float` and completely crashes DDP training!
- **Copied the Pre-Normalization Bug**: Like Suryansh, she also applied the same `(embeddings - mean) / std` pre-normalization hack, castrating the collapse-prevention mechanism.

### What Shahzeb did [ indepth bullet points with grounded reality ]
- **Pre-computed Loss Fast-Path**: Optimized `total_sigreg_loss` by checking `if 'l_sigreg' in model_output`. If the loss was already batched and computed upstream in the model, this neatly avoids all recalculations.
- **Dynamic DDP Operation Mapping**: Improved the `all_reduce` utility to robustly map string operations (like `"MAX"`) to `dist.ReduceOp.MAX` dynamically, while retaining special handling for `"AVG"`. Because of this, his seed synchronization logic successfully avoids Aarushi's float-division crash. `"MAX"` on a `LongTensor` remains a `LongTensor`, so `item()` correctly returns an `int` for `manual_seed()`.
- **Broke Autograd**: Like Aarushi, he used the in-place `dist.all_reduce()` which ruins the computational graph for distributed gradient calculation.
- **Missed the Optimization**: Reverted back to creating a new `SIGRegLoss` instance on every call, retaining Suryansh's huge performance bottleneck.
- **Copied the Pre-Normalization Bug**: Also copied the fatal `(embeddings - mean) / std` bug.

### Conclusion: What to keep, what to remove and from whoms....
- **What to Keep**:
  - The functional, autograd-safe `torch.distributed.nn.all_reduce` and proper seed synchronization from **Suryansh**.
  - The dynamic DDP Operation Mapping for `"MAX"` and `"AVG"` from **Shahzeb**.
  - The memory-efficient pre-instantiation of `SIGRegLoss()` and the batch size guardrail from **Aarushidi**.
  - The fast-path logic (`if 'l_sigreg' in model_output`) from **Shahzeb**.
- **What to Remove**:
  - The in-place `dist.all_reduce` logic from **Aarushidi** and **Shahzeb**.
  - The performance bottleneck of recreating the class from **Suryansh** and **Shahzeb**.
  - **CRITICAL**: The `(embeddings - mean) / std` pre-normalization must be **deleted completely** from the `SIGRegLoss.forward()` method. The embeddings must be passed completely raw to `self.test()` so that SIGReg can mathematically enforce variance to prevent collapse, as dictated by the paper.
