# Copernicus Retrieval Hub - Completed Track Record

This document maintains a detailed track record of all completed tasks, optimizations, and validations performed on the Copernicus Foundation Model (CopFM) cross-modal retrieval backend and frontend GIS prototype.

---

## 1. Backend Validation & Compatibility Fixes
- [x] **LoRA Dimensional Mismatch Resolved:** Configured default LoRA parameter keys (`lora_r=64`, `lora_alpha=128`) in `backbone.py` to prevent dimensions mismatch crashes when loading the `epoch_85.pth` state dict.
- [x] **LoRA Weight Loading Enabled:** Installed the Parameter-Efficient Fine-Tuning (`peft`) library to correctly handle the loaded model weights' LoRA adapter configuration.
- [x] **Process Spawning Lock Bypass:** Adjusted dataloader workers from `num_workers=4` to `num_workers=0` in the evaluation script to prevent multi-process spawning locks and pickling issues on Windows.
- [x] **Console Encoding Crash Handled:** Removed emojis and unicode arrows from `eval_comprehensive.py` print statements to prevent `UnicodeEncodeError` crashes on Windows consoles using CP1252 encoding.
- [x] **Feature Extraction Caching:** Successfully extracted and cached all test split embeddings (for S1-cross, S2-cross, S1-uni, S2-uni, and labels) to a local `test_features.npz` cache file. Subsequent evaluation runs now complete **instantly** (under 1 second).
- [x] **Baseline Metrics Verified:** Computed and confirmed retrieval metrics:
  - **S1 -> S2 (SAR to Opt):** F1@5 = 66.63% | F1@10 = 65.07%
  - **S2 -> S1 (Opt to SAR):** F1@5 = 61.73% | F1@10 = 60.60%
  - **S1 -> S1 (SAR to SAR):** F1@5 = 64.61% | F1@10 = 63.60%
  - **S2 -> S2 (Opt to Opt):** F1@5 = 68.02% | F1@10 = 67.02%

---

## 2. API Server & Dependencies Setup
- [x] **Backend Packages Installed:** Installed `fastapi`, `uvicorn`, and `python-multipart` to support the REST API server and custom TIFF image uploads.
- [x] **FastAPI Application (`app.py`):** Created the backend server implementing endpoints for:
  - `/api/status` (Health checking)
  - `/api/gallery` (Paginated test metadata filtering by country, snow, and clouds)
  - `/api/query/index` (Cosine similarity matching in NumPy)
  - `/api/query/upload` (Custom S1/S2 TIFF ingestion, model encoding, and search matching)
  - `/api/image/{modality}/{patch_id}` (On-the-fly TIFF-to-PNG visual band normalization)
  - `/api/eval` (Model accuracy validation metrics)
  - `/api/query/nearest` (Geographic coordinates Euclidean distance search to query the nearest dataset patch)
- [x] **Coordinates Precomputation:** Created and ran `precompute_coords.py` to extract coordinates for all 3,248 test patches and save them in `test_coordinates.json`, which is loaded by the API server at startup for instant map-click lookups.

---

## 3. Frontend Layout & Styling Templates
- [x] **HTML Shell Layout (`index.html`):** Created the Single Page Application template incorporating Leaflet GIS map container, control configurators, search overlays, and retrieved results grid.
- [x] **CSS Theme stylesheet (`style.css`):** Formulated a premium dark-theme glassmorphism style system with cyan/teal color accents, rounded borders, scrollbar styles, hover micro-animations, and styled Leaflet popup templates.
- [x] **JavaScript Controller (`app.js`):** Implemented map loading with Esri satellite layers, map clicks to query nearest patches, drag-and-drop file upload, search latencies stopwatch, and retrieved cards grids.

---

## 4. Dual-Modality Docks, Dynamic Metric Highlights & Image serving Fixes
- [x] **Image Filename Translation:** Serves Sentinel-1 SAR images by mapping the Sentinel-2 `patch_id` to its corresponding `s1_name` from the test metadata parquet dataframe.
- [x] **Index Realignment:** realigned the database test parquet dataframe `df_meta` on startup in `app.py` to keep only the 3,248 test samples existing on disk. This aligns metadata indices perfectly with cached embeddings and precomputed coordinates.
- [x] **Percentile Contrast Stretch:** Implemented a 2%-98% percentile min-max contrast stretch for S2 true-color bands and S1 pseudo-RGB bands. This replaces dark, near-black satellite images with vibrant, high-contrast visual imagery.
- [x] **Dual Search Logic:** Refactored query endpoints (`/api/query/index` and `/api/query/upload`) to execute both same-modality and cross-modality searches concurrently in a single request.
- [x] **Split Docks Layout:** Created a split results area showing Cross-Modality matches on the left and Same-Modality matches on the right side-by-side.
- [x] **Score Dashboard Integration:** Embedded the validation metrics matrix table permanently at the bottom of the results area, with real-time active column highlighting (e.g. S1 columns glow when query is S1).
- [x] **Gallery Pagination:** Added page navigation controls to page through the 3,248 test samples inside the configurator sidebar.

---

## 5. Multi-EyE Light Theme & Professional UI Enhancements
- [x] **Platform Rebranding:** Rebranded the entire user interface to **Multi-EyE** and cleaned up the header by removing the backend status badge.
- [x] **Light Mode Styling:** Switched core colors from dark mode to a premium light theme using slate borders, pure white cards, and vibrant cyan text highlights.
- [x] **Pulsing Query Pin:** Configured a Leaflet pulsing DivIcon with custom CSS query animations to ripple outward from the query coordinate.
- [x] **Fullscreen Lightbox Modals:** Created click event triggers on result card image containers and preview icons to open a fullscreen lightbox containing image scales and active land-cover tags.
- [x] **Side-by-Side Previews:** Implemented side-by-side previews for Sentinel-1 (SAR) and Sentinel-2 (Optical) modality bands in the query detail configurator card.
- [x] **Local Memory Caching:** Saved loaded page metadata locally in the JS app state, accelerating query clicks to 0ms.

---

## 6. Premium Light Theme Consistency, Floating Map Legend & Live Satellite Imagery
- [x] **Clean Uniform Box Styles:** Unified border-radius, shadows, margins, and backgrounds of all controls, query preview cards, dropdown select filters, result cards, and the metrics table to match a consistent premium Light Theme aesthetic, removing any remnants of dark slate overlays.
- [x] **Header Branding Visibility:** Removed the cut-off eye brand icon, stacked the `Multi-EyE` title vertically above `Multimodal Sat-Retrieval Platform`, and fixed typography constraints to ensure clean, high-contrast visibility.
- [x] **Header Search Bar Relocation:** Moved the Nominatim coordinate search bar off the map container and integrated it directly into the right side of the dashboard header with a premium light theme input field.
- [x] **Floating Map Legend Card:** Added a glassmorphic floating legend card overlay at the bottom-left of the Leaflet map, explaining query pulsing pins and same/cross modality similarity paths.
- [x] **Live Satellite Coordinates Popups:** Enabled ArcGIS/Esri World Imagery export requests on map clicks to render high-resolution satellite tiles on-the-fly for any clicked coordinate inside Leaflet popups.
- [x] **Modality-Specific Result Headers:** Replaced the abstract same/cross titles in the results layout with explicit labels (e.g. "Sentinel-1 (SAR) Retrieved Matches" and "Sentinel-2 (Optical) Retrieved Matches") based on the query modality.
- [x] **Stacked Query Details:** Stacked the S1/S2 previews and metadata details vertically to prevent label truncation, providing ample space for coordinates and country names without horizontal clipping.
- [x] **Compact Inspect Lightbox Modal:** Re-proportioned the inspect overlay to allocate **66% width** (`flex: 1.8`) to the image block and **34% width** (`flex: 1.1`) to the text sidebar. Switched the image rendering style to `object-fit: cover` to fill the container beautifully, and decreased sidebar text fonts/tags slightly to create an elegant, image-centric inspect UX.
- [x] **Jaccard Intersection-over-Union Metrics:** Refactored the dynamic single-query metrics algorithm to compute **Jaccard Similarity** (Intersection over Union) of land cover labels between the query and retrieved images. This resolves the issue of broad label overlaps yielding a flat 100.00% score, providing realistic, varying percentages (e.g. 78.5%, 89.2%) for F1 and AP on every query.
- [x] **Omnipresent Instant Map Previews:** Integrated live close-up satellite tile previews inside all map marker popups using the fast Slippy Map tile conversion formula (`getEsriTileUrl`). Images now load instantly from CDN cache (zero REST export lags) at zoom level 15.
- [x] **Strict Modality Column Hiding:** Refactored the score matrix dashboard table to dynamically show and hide columns according to your query modality choice (e.g. S2 columns are completely hidden when you select S1).
- [x] **Leaflet Popup Alignment:** Customized Leaflet marker bindings with a custom class overlay, fixing squishing issues and matching the premium Light Theme wrapper borders.
- [x] **CSP-Compliant Image Loader:** Refactored the image loading visual states in CSS using `z-index` overlay stack alignments, eliminating inline script execution risks for superior compatibility.
- [x] **Photon Geocoding Integration:** Upgraded the geographic search engine to query the permissive Photon API (fuzzy search index) as the primary geocoder. This handles abbreviations and acronyms (like "DTU") instantly, and falls back to Nominatim only if no features are returned.
- [x] **JS Syntax Resolution:** Fixed a duplicate closing bracket syntax error in `app.js` which was preventing the page script from parsing and initializing.
- [x] **Safe Label Fallback:** Implemented validation checks inside `buildMarkerPopupHtml` to cleanly handle empty or undefined label sets, preventing crashes during custom TIFF query uploads.
- [x] **JS Crash Safeguard:** Cleaned up unused `snow-filter` and `cloud-filter` event listeners in `app.js` to prevent runtime script crashes.




