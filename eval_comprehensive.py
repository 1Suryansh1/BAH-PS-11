import os
import time
import torch
import yaml
import faiss
import numpy as np
from tqdm import tqdm
from src.datasets.ben14k import BEN14KDataset
from src.models.copfm_retrieval import CopFMRetrieval
from src.wavelengths import get_wavelengths

def evaluate_comprehensive(query_embs, query_labels, gallery_embs, gallery_labels, ks=[5, 10], is_same_modal=False):
    D = query_embs.shape[1]
    index = faiss.IndexFlatIP(D)
    index.add(gallery_embs.astype(np.float32))
    
    search_k = max(ks) + 1 if is_same_modal else max(ks)
    start_time = time.time()
    distances, indices = index.search(query_embs.astype(np.float32), search_k)
    avg_time_ms = ((time.time() - start_time) / len(query_embs)) * 1000.0
    
    results = {}
    for k in ks:
        all_f1s = []
        for i in range(len(query_embs)):
            q_label = query_labels[i]
            q_sum = q_label.sum()
            valid_retrievals = 0
            query_f1s = []
            
            for j in range(search_k):
                r_idx = indices[i, j]
                if is_same_modal and r_idx == i:
                    continue
                if valid_retrievals >= k:
                    break
                
                r_label = gallery_labels[r_idx]
                r_sum = r_label.sum()
                intersection = np.dot(q_label, r_label)
                if r_sum == 0 or q_sum == 0:
                    f1_qr = 0.0
                else:
                    p_qr = intersection / r_sum
                    r_qr = intersection / q_sum
                    f1_qr = 0.0 if (p_qr + r_qr) == 0 else 2 * p_qr * r_qr / (p_qr + r_qr + 1e-8)
                query_f1s.append(f1_qr)
                valid_retrievals += 1
            all_f1s.append(np.mean(query_f1s))
        results[f'f1@{k}'] = np.mean(all_f1s) * 100.0
    return results, avg_time_ms

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on device: {device}")
    
    # Load config
    config_path = 'configs/ben14k.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Override paths to local environment
    config['data']['root_dir'] = '../BEN_14k'
    config['data']['modality_b'] = 'S2_BEN_10'  # 10 channels to match the checkpoint & local tiff files
    print("MODALITY_B IN CONFIG:", config['data']['modality_b'])
    
    ckpt_path = 'epoch_85.pth'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        
    # Initialize model
    print("Initializing model...")
    model = CopFMRetrieval(config['model'])
    
    # If Multi-GPU was used, load state dict with prefix handling
    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Strip 'module.' prefix if present in checkpoint (saved from DataParallel) but not in our model
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Load dataset
    print("Initializing test dataloader...")
    test_dataset = BEN14KDataset(config['data']['root_dir'], split='test', img_size=config['data']['img_size'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=0
    )
    
    wl_a, bw_a = get_wavelengths(config['data']['modality_a'])
    wl_b, bw_b = get_wavelengths(config['data']['modality_b'])
    
    features_cache_path = 'test_features.npz'
    if os.path.exists(features_cache_path):
        print(f"Loading cached features from {features_cache_path}...")
        cache = np.load(features_cache_path)
        s1_cross = cache['s1_cross']
        s2_cross = cache['s2_cross']
        s1_uni = cache['s1_uni']
        s2_uni = cache['s2_uni']
        all_labels = cache['all_labels']
        print("Cached features loaded successfully!")
    else:
        s1_cross, s2_cross, s1_uni, s2_uni, all_labels = [], [], [], [], []
        
        print("Extracting features for the TEST set...")
        with torch.no_grad():
            for batch in tqdm(test_loader):
                img_a, img_b = batch['s1'].to(device), batch['s2'].to(device)
                B, _, H, W = img_a.shape
                mask = torch.zeros((B, (H//16)*(W//16)), dtype=torch.bool, device=device)
                
                out = model(img_a, img_b, wl_a, wl_b, bw_a, bw_b, mask, mask)
                
                e_c_a = out['e_cross_a'].cpu().numpy()
                e_c_b = out['e_cross_b'].cpu().numpy()
                s1_cross.append(e_c_a / (np.linalg.norm(e_c_a, axis=1, keepdims=True) + 1e-8))
                s2_cross.append(e_c_b / (np.linalg.norm(e_c_b, axis=1, keepdims=True) + 1e-8))
                
                e_u_a = out['e_uni_a'].cpu().numpy()
                e_u_b = out['e_uni_b'].cpu().numpy()
                s1_uni.append(e_u_a / (np.linalg.norm(e_u_a, axis=1, keepdims=True) + 1e-8))
                s2_uni.append(e_u_b / (np.linalg.norm(e_u_b, axis=1, keepdims=True) + 1e-8))
                
                all_labels.append(batch['label'].numpy())
                
        s1_cross = np.concatenate(s1_cross, axis=0)
        s2_cross = np.concatenate(s2_cross, axis=0)
        s1_uni = np.concatenate(s1_uni, axis=0)
        s2_uni = np.concatenate(s2_uni, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        print(f"Saving extracted features to {features_cache_path}...")
        np.savez(
            features_cache_path,
            s1_cross=s1_cross,
            s2_cross=s2_cross,
            s1_uni=s1_uni,
            s2_uni=s2_uni,
            all_labels=all_labels
        )
    
    print("\n" + "="*50 + "\n=== CROSS-MODAL EVALUATION ===\n" + "="*50)
    res_s1_s2, t_s1_s2 = evaluate_comprehensive(s1_cross, all_labels, s2_cross, all_labels, is_same_modal=False)
    print(f"S1 -> S2 (SAR to Opt) | F1@5: {res_s1_s2['f1@5']:.2f}% | F1@10: {res_s1_s2['f1@10']:.2f}% | Time/Query: {t_s1_s2:.4f} ms")
    
    res_s2_s1, t_s2_s1 = evaluate_comprehensive(s2_cross, all_labels, s1_cross, all_labels, is_same_modal=False)
    print(f"S2 -> S1 (Opt to SAR) | F1@5: {res_s2_s1['f1@5']:.2f}% | F1@10: {res_s2_s1['f1@10']:.2f}% | Time/Query: {t_s2_s1:.4f} ms")
    
    print("\n" + "="*50 + "\n=== SAME-MODAL EVALUATION (Self-Matches Filtered) ===\n" + "="*50)
    res_s1_s1, t_s1_s1 = evaluate_comprehensive(s1_uni, all_labels, s1_uni, all_labels, is_same_modal=True)
    print(f"S1 -> S1 (SAR to SAR) | F1@5: {res_s1_s1['f1@5']:.2f}% | F1@10: {res_s1_s1['f1@10']:.2f}% | Time/Query: {t_s1_s1:.4f} ms")
    
    res_s2_s2, t_s2_s2 = evaluate_comprehensive(s2_uni, all_labels, s2_uni, all_labels, is_same_modal=True)
    print(f"S2 -> S2 (Opt to Opt) | F1@5: {res_s2_s2['f1@5']:.2f}% | F1@10: {res_s2_s2['f1@10']:.2f}% | Time/Query: {t_s2_s2:.4f} ms")

if __name__ == "__main__":
    main()
