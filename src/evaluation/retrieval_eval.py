import torch
import numpy as np
from tqdm import tqdm
import faiss
from .metrics import compute_f1_at_k, compute_map, compute_precision_at_k
from ..wavelengths import get_wavelengths

from ..utils.key_mapping import get_modality_key

def extract_all_embeddings(model, dataloader, modality: str, mode: str = 'cross', device='cpu'):
    model.eval()
    all_embs, all_labels, all_ids = [], [], []
    
    # Infer dataset name
    dataset_name = ""
    if hasattr(dataloader.dataset, 'dataset_name'):
        dataset_name = dataloader.dataset.dataset_name
    else:
        cls_name = dataloader.dataset.__class__.__name__
        if "BEN14K" in cls_name:
            dataset_name = "BigEarthNet"
        elif "CBRSIR" in cls_name:
            dataset_name = "CBRSIR"
        elif "DSRSID" in cls_name:
            dataset_name = "DSRSID"
            
    mod_key = get_modality_key(dataset_name, modality)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {modality} embeddings"):
            wl, bw = get_wavelengths(modality)
            if mod_key in batch:
                img = batch[mod_key].to(device)
            else:
                img = batch['image'].to(device) # fallback
                
            emb = model.get_retrieval_embedding(img, wl, bw, mode=mode)
            all_embs.append(emb.cpu().numpy())
            all_labels.append(batch['label'].cpu().numpy())
            
            id_key = 'pair_id' if 'pair_id' in batch else 'sample_id'
            if id_key in batch:
                all_ids.extend(batch[id_key])
            else:
                all_ids.extend([f"id_{len(all_ids)+j}" for j in range(len(img))])
            
    embeddings = np.concatenate(all_embs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, labels, all_ids

def evaluate_retrieval_direction(query_embs, query_labels,
                                  gallery_embs, gallery_labels,
                                  ks=[5, 10], metric='f1',
                                  query_ids=None, gallery_ids=None,
                                  filter_self=False):
    D = query_embs.shape[1]
    
    query_embs = np.ascontiguousarray(query_embs, dtype=np.float32)
    gallery_embs = np.ascontiguousarray(gallery_embs, dtype=np.float32)
    
    max_k = max(ks)
    # If filtering self-matches, retrieve k+1 neighbors
    fetch_k = max_k + 1 if filter_self else max_k
    
    index = faiss.IndexFlatIP(D)
    index.add(gallery_embs)
    
    distances, indices = index.search(query_embs, fetch_k)
    
    # Filter self-matches if enabled
    filtered_indices = []
    for i in range(len(query_embs)):
        idx_i = indices[i]
        if filter_self and query_ids is not None and gallery_ids is not None:
            q_id = query_ids[i]
            if q_id in gallery_ids:
                self_idx = gallery_ids.index(q_id)
                mask = (idx_i != self_idx)
                idx_i = idx_i[mask][:max_k]
            else:
                idx_i = idx_i[:max_k]
        else:
            idx_i = idx_i[:max_k]
        filtered_indices.append(idx_i)
        
    indices = np.array(filtered_indices)
    
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
