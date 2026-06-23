import torch
import numpy as np
from tqdm import tqdm
import faiss
from .metrics import compute_f1_at_k, compute_map, compute_precision_at_k
from ..wavelengths import get_wavelengths

def extract_all_embeddings(model, dataloader, modality: str, mode: str = 'cross', device='cpu'):
    model.eval()
    all_embs, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {modality} embeddings"):
            wl, bw = get_wavelengths(modality)
            mod_key = modality.lower()
            if 's1' in mod_key or 's2' in mod_key:
                img = batch[mod_key].to(device)
            elif 'rgb' in mod_key or 'sar' in mod_key:
                img = batch[mod_key].to(device)
            elif 'pan' in mod_key or 'ms' in mod_key:
                img = batch[mod_key].to(device)
            else:
                img = batch['image'].to(device) # fallback
                
            meta_key = mod_key + '_meta'
            meta_info = batch[meta_key].to(device) if meta_key in batch else None
            
            emb = model.get_retrieval_embedding(img, wl, bw, mode=mode, meta_info=meta_info)
            all_embs.append(emb.cpu().numpy())
            all_labels.append(batch['label'].cpu().numpy())
            
    embeddings = np.concatenate(all_embs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, labels

def evaluate_retrieval_direction(query_embs, query_labels,
                                  gallery_embs, gallery_labels,
                                  ks=[5, 10], metric='f1', exclude_self=False):
    D = query_embs.shape[1]
    
    query_embs = np.ascontiguousarray(query_embs, dtype=np.float32)
    gallery_embs = np.ascontiguousarray(gallery_embs, dtype=np.float32)
    
    index = faiss.IndexFlatIP(D)
    index.add(gallery_embs)
    
    if exclude_self:
        # Search for one extra item in case the query itself is matched
        distances, indices = index.search(query_embs, max(ks) + 1)
        filtered_indices = []
        for i, ret_idx in enumerate(indices):
            # Exclude query itself (gallery index equal to query index i)
            mask = ret_idx != i
            filtered = ret_idx[mask]
            filtered_indices.append(filtered[:max(ks)])
        indices = np.stack(filtered_indices)
    else:
        distances, indices = index.search(query_embs, max(ks))
    
    total_relevant_list = []
    if metric == 'f1':
        for q_label in query_labels:
            # handle both single and multi label generic cases
            q_arr = np.array(q_label)
            g_arr = np.array(gallery_labels)
            if q_arr.ndim > 0 and g_arr.ndim > 1 and q_arr.shape[-1] > 1:
                total_relevant_list.append((np.dot(g_arr, q_arr) > 0).sum())
            else:
                if g_arr.ndim > 1 and g_arr.shape[1] == 1:
                    g_arr = g_arr.ravel()
                if q_arr.ndim > 0 and q_arr.shape[0] == 1:
                    q_arr = q_arr.item()
                total_relevant_list.append((g_arr == q_arr).sum())
                
    metrics = {}
    for k in ks:
        scores = []
        for i, (q_label, retrieved_idx) in enumerate(zip(query_labels, indices)):
            ret_labels = gallery_labels[retrieved_idx[:k]]
            if metric == 'f1':
                total_relevant = total_relevant_list[i]
                score = compute_f1_at_k(ret_labels, q_label, total_relevant, k)
            elif metric == 'map':
                score = compute_map(ret_labels, q_label)
            else:
                score = 0
            scores.append(score)
        metrics[f'{metric}@{k}'] = np.mean(scores) * 100
    
    return metrics
