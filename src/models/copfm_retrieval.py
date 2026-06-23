import torch
import torch.nn as nn
from .backbone import CopFMBackbone
from .predictor import CrossModalPredictor, SameModalPredictor
from .retrieval_heads import DualRetrievalHead
from ..utils.masking import get_visible_tokens, get_masked_tokens

class CopFMRetrieval(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        
        self.backbone = CopFMBackbone(
            checkpoint_path=config['backbone_checkpoint'],
            freeze_mode=config['freeze_mode']
        )
        
        # Same-modal predictors
        depth = config.get('predictor_depth', 6)
        self.predictor_aa = SameModalPredictor(depth=depth)
        self.predictor_bb = SameModalPredictor(depth=depth)
        
        # Cross-modal predictor (SHARED)
        self.predictor_cross = CrossModalPredictor(depth=depth)
        
        # Retrieval heads
        retrieval_dim = config.get('retrieval_dim', 256)
        self.retrieval_heads = DualRetrievalHead(output_dim=retrieval_dim)

    def encode(self, image, wavelengths, bandwidths, meta_info=None):
        """
        Encode a single modality image to patch tokens.
        """
        return self.backbone(image, wavelengths, bandwidths, return_patch_tokens=True, meta_info=meta_info)

    def get_retrieval_embedding(self, image, wavelengths, bandwidths, mode='cross', meta_info=None):
        """
        Get retrieval embedding for inference.
        """
        tokens = self.encode(image, wavelengths, bandwidths, meta_info=meta_info)
        if mode == 'cross':
            e, _ = self.retrieval_heads.phi_cross(tokens)
        else:
            e, _ = self.retrieval_heads.phi_uni(tokens)
        return e

    def forward_train(self, batch_a, batch_b, wl_a, wl_b, bw_a, bw_b, mask_a, mask_b, meta_info_a=None, meta_info_b=None):
        """
        Full training forward pass.
        """
        # Step 1: encode both modalities
        tokens_a = self.backbone(batch_a, wl_a, bw_a, return_patch_tokens=True, meta_info=meta_info_a)
        tokens_b = self.backbone(batch_b, wl_b, bw_b, return_patch_tokens=True, meta_info=meta_info_b)
        
        # Step 2: get visible/masked splits
        vis_a = get_visible_tokens(tokens_a, mask_a)   # (B, N_vis, D)
        vis_b = get_visible_tokens(tokens_b, mask_b)
        tgt_a = get_masked_tokens(tokens_a, mask_a)    # (B, N_mask, D)  — TARGET
        tgt_b = get_masked_tokens(tokens_b, mask_b)
        
        # Step 3: predictive paths
        # Same-modal A (context A visible -> predict A masked)
        pred_aa = self.predictor_aa(vis_a, mask_a)
        # Same-modal B
        pred_bb = self.predictor_bb(vis_b, mask_b)
        # Cross-modal A->B (context A visible -> predict B masked)
        pred_ab = self.predictor_cross(vis_a, mask_b)
        # Cross-modal B->A
        pred_ba = self.predictor_cross(vis_b, mask_a)
        
        # Step 4: retrieval embeddings from FULL token sequences
        e_ca, r_ca = self.retrieval_heads.phi_cross(tokens_a)
        e_cb, r_cb = self.retrieval_heads.phi_cross(tokens_b)
        e_ua, r_ua = self.retrieval_heads.phi_uni(tokens_a)
        e_ub, r_ub = self.retrieval_heads.phi_uni(tokens_b)
        
        return {
            'tokens_a': tokens_a, 'tokens_b': tokens_b,
            'tgt_a': tgt_a, 'tgt_b': tgt_b,
            'pred_aa': pred_aa, 'pred_bb': pred_bb,
            'pred_ab': pred_ab, 'pred_ba': pred_ba,
            'e_cross_a': e_ca, 'r_cross_a': r_ca,
            'e_cross_b': e_cb, 'r_cross_b': r_cb,
            'e_uni_a': e_ua, 'r_uni_a': r_ua,
            'e_uni_b': e_ub, 'r_uni_b': r_ub
        }
