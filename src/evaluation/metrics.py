import numpy as np

def compute_f1_at_k(retrieved_labels, query_label, total_relevant, k=5):
    """
    F1@K for multi-label retrieval (BEN-14K protocol).
    """
    retrieved_k = retrieved_labels[:k]
    
    r_arr = np.array(retrieved_k)
    q_arr = np.array(query_label)
    
    if q_arr.ndim > 0 and r_arr.ndim > 1 and q_arr.shape[-1] > 1:
        relevant_mask = (np.dot(r_arr, q_arr) > 0).astype(int)
    else:
        # Single label
        if r_arr.ndim > 1 and r_arr.shape[1] == 1:
            r_arr = r_arr.ravel()
        if q_arr.ndim > 0 and q_arr.shape[0] == 1:
            q_arr = q_arr.item()
        relevant_mask = (r_arr == q_arr).astype(int)
        
    num_relevant_in_k = relevant_mask.sum()
    
    if total_relevant == 0:
        return 0.0
        
    precision_k = num_relevant_in_k / k
    recall_k = num_relevant_in_k / total_relevant
    
    if precision_k + recall_k == 0:
        return 0.0
        
    return 2 * (precision_k * recall_k) / (precision_k + recall_k)

def compute_map(retrieved_labels, query_label):
    """
    Mean Average Precision for single-label and multi-label datasets.
    """
    q_arr = np.array(query_label)
    r_arr = np.array(retrieved_labels)
    
    if q_arr.ndim > 0 and r_arr.ndim > 1 and q_arr.shape[-1] > 1:
        # Multi-label or one-hot encoded single label
        is_relevant = (np.dot(r_arr, q_arr) > 0).astype(int)
    else:
        # Single label
        if r_arr.ndim > 1 and r_arr.shape[1] == 1:
            r_arr = r_arr.ravel()
        if q_arr.ndim > 0 and q_arr.shape[0] == 1:
            q_arr = q_arr.item()
        is_relevant = (r_arr == q_arr).astype(int)
        
    if is_relevant.sum() == 0:
        return 0.0
        
    precisions = []
    num_rel = 0
    for i, rel in enumerate(is_relevant):
        if rel == 1:
            num_rel += 1
            precisions.append(num_rel / (i + 1.0))
            
    return np.mean(precisions)

def compute_precision_at_k(retrieved_labels, query_label, k=5):
    """
    P@K for single-label datasets.
    """
    retrieved_k = retrieved_labels[:k]
    q_arr = np.array(query_label)
    r_arr = np.array(retrieved_k)
    
    if q_arr.ndim > 0 and r_arr.ndim > 1 and q_arr.shape[-1] > 1:
        is_relevant = (np.dot(r_arr, q_arr) > 0).astype(int)
    else:
        if r_arr.ndim > 1 and r_arr.shape[1] == 1:
            r_arr = r_arr.ravel()
        if q_arr.ndim > 0 and q_arr.shape[0] == 1:
            q_arr = q_arr.item()
        is_relevant = (r_arr == q_arr).astype(int)
        
    return is_relevant.sum() / k

def run_full_evaluation(embeddings: dict, labels: dict, dataset_name: str):
    """
    Placeholder for complete evaluation running script.
    """
    return {"status": "Metrics ready for computation."}
