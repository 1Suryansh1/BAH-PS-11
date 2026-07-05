# Copernicus Foundation Model Cross-Modal Retrieval Dashboard (CopFM-LoRA)

An interactive, state-of-the-art web application and deep learning pipeline designed to perform same-modal and cross-modal remote sensing image retrieval on the **BigEarthNet-MM** dataset. By utilizing Parameter-Efficient Fine-Tuning (PEFT) with **LoRA (Low-Rank Adaptation)** on the **Copernicus Sentinel Foundation Model**, CopFM-LoRA aligns Sentinel-1 Synthetic Aperture Radar (SAR) backscatter profiles with Sentinel-2 Multi-Spectral Optical imagery embeddings in a joint representation space.

---

## LIVE RUN !!! \

https://www.kaggle.com/code/suryansh10100/copfm-jepa 

OUR LIVE BEN-14K BENCHMARK RESULTS  ! 

Its CopFM-JEPA !


## 🚀 Key Features

* **Cross-Modal Retrieval**: Search Sentinel-1 (SAR) queries to retrieve matching Sentinel-2 (Optical) scenes, and vice versa.
* **Interactive GIS Map**: Powered by Leaflet.js. Click on any coordinates or search globally (via Nominatim) to snap to the nearest database patch and trigger instant similar-scene retrieval.
* **SOTA Benchmarks Standings**: A premium, light-mode white card modal comparing the performance of **14 state-of-the-art foundation models** (e.g. CR-JEPA, X-JEPA, SatMAE, SkySense, CROMA). **CopFM-LoRA** is highlighted in gold and ranks **#2🥈 overall** in cross-modal retrieval performance.
* **Detailed Latency Split**: Tracks execution speeds on each search:
  * **API Roundtrip Latency** (Network overhead)
  * **Model Inference Latency** (Live deep learning feature extraction)
  * **FAISS Search Latency** (Sub-millisecond nearest-neighbor lookups)
* **Custom GeoTIFF Upload**: Upload or drag-and-drop your own `.tif` files. If the filename matches a dataset scene, the true land cover labels and country are automatically matched.
* **Dynamic Metric Evaluation**: Computes Jaccard label overlap, dynamic F1@5, F1@10, and mean Average Precision (mAP) metrics on the fly.

---

## 🛠️ Technology Stack

### Backend
* **Python 3.8+**
* **FastAPI**: Lightweight high-performance REST API.
* **PyTorch**: Deep learning model loading and feature extraction.
* **FAISS**: Facebook AI Similarity Search for high-dimensional vector similarity lookups.
* **Rasterio**: Reading and warping GeoTIFF coordinates.
* **Pandas / NumPy**: Metadata indexing and data wrangling.

### Frontend
* **Vanilla HTML5 & CSS3**: Glassmorphic dark styling, responsive dashboard layout, and light-themed tables.
* **Vanilla JavaScript**: Asynchronous API pipelines, UI state transitions, and search status phases.
* **Leaflet.js**: Geospatial map rendering, marker groups, and spatial connection polylines.

---

## 📂 Project Structure

```text
├── app.py                     # FastAPI web server and API routes
├── app.js                     # Frontend logic (map events, upload, API calls)
├── index.html                 # Main dashboard UI
├── style.css                  # UI styling sheet
├── src/
│   ├── models/                # Copernicus FM architecture and LoRA adapters
│   └── datasets/              # BigEarthNet dataset loader class
├── configs/                   # Model and retrieval configs
├── test_coordinates.json      # Pre-mapped longitude/latitude coordinates
├── test_features.npz          # Precomputed test dataset embeddings
├── epoch_85.pth               # Model checkpoint weights (LoRA adapters)
└── .gitignore                 # Files excluded from version control
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/copfm-lora-retrieval.git
cd copfm-lora-retrieval
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Server
```bash
python app.py
```
The application will launch locally at **`http://localhost:8000`**. Open this address in your web browser to explore the dashboard.
