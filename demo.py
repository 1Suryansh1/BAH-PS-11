import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from src.models.copfm_retrieval import CopFMRetrieval
from src.retrieval.faiss_index import FAISSRetrievalIndex

def demo_retrieval():
    print("Starting End-to-End Retrieval Demo...")
    
    # 1. Initialize Model
    config = {
        'backbone_checkpoint': 'dummy.pth',
        'freeze_mode': 'full',
        'retrieval_dim': 256
    }
    model = CopFMRetrieval(config)
    model.eval()
    
    # 2. Simulate Data Extraction
    print("Extracting embeddings for database (N=100)...")
    db_images = torch.randn(100, 3, 224, 224)
    # CopFM wavelengths/bandwidths config
    wl = [0.4, 0.5, 0.6]
    bw = [0.1, 0.1, 0.1]
    
    with torch.no_grad():
        db_embeddings = model.get_retrieval_embedding(db_images, wl, bw, mode='cross').numpy()
    
    db_labels = np.array([f"DB_Img_{i}" for i in range(100)])
    
    # 3. Build FAISS Index
    print("Building FAISS Index...")
    index = FAISSRetrievalIndex(embedding_dim=256, index_type='IVFFlat', n_list=10)
    index.build(db_embeddings, db_labels)
    
    # 4. Query
    query_image = torch.randn(1, 3, 224, 224)
    print("Executing search query...")
    with torch.no_grad():
        start = time.time()
        query_emb = model.get_retrieval_embedding(query_image, wl, bw, mode='cross').numpy()
        distances, indices, ret_labels = index.search(query_emb, k=5)
        end = time.time()
    
    total_ms = (end - start) * 1000
    print(f"Retrieved Top-5 in {total_ms:.2f}ms")
    print(f"Top-5 Labels: {ret_labels}")
    
    # 5. Visualizations
    os.makedirs('./demo_output', exist_ok=True)
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    
    # helper for dummy images
    def to_img(tensor):
        img = tensor.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img
    
    axes[0].imshow(to_img(query_image[0]))
    axes[0].set_title("Query")
    axes[0].axis('off')
    
    for i in range(5):
        idx = indices[0][i]
        axes[i+1].imshow(to_img(db_images[idx]))
        axes[i+1].set_title(f"Rank {i+1}\nDist: {distances[0][i]:.2f}")
        axes[i+1].axis('off')
        
    plt.tight_layout()
    plt.savefig('./demo_output/retrieval_results.png')
    print("\nVisualizations saved to ./demo_output/retrieval_results.png")

if __name__ == "__main__":
    demo_retrieval()
