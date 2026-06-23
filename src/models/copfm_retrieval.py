import torch
import torch.nn as nn
import copy
from .backbone import CopFMBackbone
from .predictor import CrossModalPredictor, SameModalPredictor
from .retrieval_heads import DualRetrievalHead
from ..utils.masking import get_visible_tokens, get_masked_tokens
from ..losses.sigreg import SIGRegLoss

class CopFMRetrieval(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        
        self.backbone = CopFMBackbone(
            checkpoint_path=config['backbone_checkpoint'],
            freeze_mode=config['freeze_mode']
        )
        
        # Frozen EMA-updated target backbone to prevent information leakage
        self.target_backbone = copy.deepcopy(self.backbone)
        for p in self.target_backbone.parameters():
            p.requires_grad = False
            
        self.ema_decay = config.get('ema_decay', 0.996)
        
        # Same-modal predictors
        depth = config.get('predictor_depth', 6)
        self.predictor_aa = SameModalPredictor(depth=depth)
        self.predictor_bb = SameModalPredictor(depth=depth)
        
        # Cross-modal predictor (SHARED)
        self.predictor_cross = CrossModalPredictor(depth=depth)
        
        # Retrieval heads
        retrieval_dim = config.get('retrieval_dim', 256)
        self.retrieval_heads = DualRetrievalHead(output_dim=retrieval_dim)
        
        # Persistent SIGRegLoss modules to preserve global_step tracking
        self.sigreg_loss_cross_a = SIGRegLoss()
        self.sigreg_loss_cross_b = SIGRegLoss()
        self.sigreg_loss_uni_a   = SIGRegLoss()
        self.sigreg_loss_uni_b   = SIGRegLoss()

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

    @torch.no_grad()
    def update_target_ema(self):
        """
        Update the target encoder weights via Exponential Moving Average (EMA).
        """
        for param_q, param_k in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            param_k.data.mul_(self.ema_decay).add_((1 - self.ema_decay) * param_q.detach().data)

    def forward_train(self, batch_a, batch_b, wl_a, wl_b, bw_a, bw_b, mask_a, mask_b, meta_a=None, meta_b=None):
        """
        Full training forward pass.
        """
        # Step 1: encode visible patches only for context backbone (leakage prevention)
        vis_a = self.backbone(batch_a, wl_a, bw_a, return_patch_tokens=True, mask=mask_a, meta_info=meta_a)
        vis_b = self.backbone(batch_b, wl_b, bw_b, return_patch_tokens=True, mask=mask_b, meta_info=meta_b)
        
        # Step 2: encode full target images using frozen target backbone (EMA)
        with torch.no_grad():
            full_tokens_a = self.target_backbone(batch_a, wl_a, bw_a, return_patch_tokens=True, meta_info=meta_a)
            full_tokens_b = self.target_backbone(batch_b, wl_b, bw_b, return_patch_tokens=True, meta_info=meta_b)
            
            tgt_a = get_masked_tokens(full_tokens_a, mask_a)    # (B, N_mask, D)  — TARGET
            tgt_b = get_masked_tokens(full_tokens_b, mask_b)
        
        # Step 3: predictive paths
        pred_aa = self.predictor_aa(vis_a, mask_a)
        pred_bb = self.predictor_bb(vis_b, mask_b)
        pred_ab = self.predictor_cross(vis_a, mask_b)
        pred_ba = self.predictor_cross(vis_b, mask_a)
        
        # Step 4: retrieval embeddings from FULL token sequences of context backbone
        full_context_a = self.backbone(batch_a, wl_a, bw_a, return_patch_tokens=True, meta_info=meta_a)
        full_context_b = self.backbone(batch_b, wl_b, bw_b, return_patch_tokens=True, meta_info=meta_b)
        
        e_ca, r_ca = self.retrieval_heads.phi_cross(full_context_a)
        e_cb, r_cb = self.retrieval_heads.phi_cross(full_context_b)
        e_ua, r_ua = self.retrieval_heads.phi_uni(full_context_a)
        e_ub, r_ub = self.retrieval_heads.phi_uni(full_context_b)
        
        # Compute SIGReg losses using persistent modules to avoid static random projections
        loss_sigreg_cross_a = self.sigreg_loss_cross_a(r_ca)
        loss_sigreg_cross_b = self.sigreg_loss_cross_b(r_cb)
        loss_sigreg_uni_a   = self.sigreg_loss_uni_a(r_ua)
        loss_sigreg_uni_b   = self.sigreg_loss_uni_b(r_ub)
        l_sigreg = (loss_sigreg_cross_a + loss_sigreg_cross_b + loss_sigreg_uni_a + loss_sigreg_uni_b) / 4.0
        
        return {
            'tokens_a': full_context_a, 'tokens_b': full_context_b,
            'tgt_a': tgt_a, 'tgt_b': tgt_b,
            'pred_aa': pred_aa, 'pred_bb': pred_bb,
            'pred_ab': pred_ab, 'pred_ba': pred_ba,
            'e_cross_a': e_ca, 'r_cross_a': r_ca,
            'e_cross_b': e_cb, 'r_cross_b': r_cb,
            'e_uni_a': e_ua, 'r_uni_a': r_ua,
            'e_uni_b': e_ub, 'r_uni_b': r_ub,
            'l_sigreg': l_sigreg
        }
