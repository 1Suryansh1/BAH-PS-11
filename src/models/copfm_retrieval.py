import copy
import torch
import torch.nn as nn
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

        self.target_backbone = copy.deepcopy(self.backbone)
        for p in self.target_backbone.parameters():
            p.requires_grad = False

        depth = config.get('predictor_depth', 6)
        self.predictor_aa = SameModalPredictor(depth=depth)
        self.predictor_bb = SameModalPredictor(depth=depth)
        self.predictor_cross = CrossModalPredictor(depth=depth)

        retrieval_dim = config.get('retrieval_dim', 256)
        self.retrieval_heads = DualRetrievalHead(output_dim=retrieval_dim)

        self.sigreg = SIGRegLoss(num_slices=128)

    @torch.no_grad()
    def update_target_ema(self, momentum: float = 0.996):
        for p_online, p_target in zip(
            self.backbone.parameters(), self.target_backbone.parameters()
        ):
            p_target.data = momentum * p_target.data + (1.0 - momentum) * p_online.data

    def encode(self, image, wavelengths, bandwidths, meta=None):
        return self.backbone(image, wavelengths, bandwidths, meta=meta, return_patch_tokens=True)

    def get_retrieval_embedding(self, image, wavelengths, bandwidths, mode='cross', meta=None):
        tokens = self.encode(image, wavelengths, bandwidths, meta=meta)
        if mode == 'cross':
            e, _ = self.retrieval_heads.phi_cross(tokens)
        else:
            e, _ = self.retrieval_heads.phi_uni(tokens)
        return e

    def forward_train(self, batch_a, batch_b, wl_a, wl_b, bw_a, bw_b,
                      mask_a, mask_b, meta_a=None, meta_b=None):
        device = next(self.parameters()).device
        batch_a = batch_a.to(device)
        batch_b = batch_b.to(device)
        mask_a = mask_a.to(device)
        mask_b = mask_b.to(device)

        tokens_a = self.backbone(batch_a, wl_a, bw_a, meta=meta_a, return_patch_tokens=True)
        tokens_b = self.backbone(batch_b, wl_b, bw_b, meta=meta_b, return_patch_tokens=True)

        with torch.no_grad():
            tgt_a_full = self.target_backbone(batch_a, wl_a, bw_a, meta=meta_a, return_patch_tokens=True)
            tgt_b_full = self.target_backbone(batch_b, wl_b, bw_b, meta=meta_b, return_patch_tokens=True)

        vis_a = get_visible_tokens(tokens_a, mask_a)
        vis_b = get_visible_tokens(tokens_b, mask_b)
        tgt_a = get_masked_tokens(tgt_a_full.detach(), mask_a)
        tgt_b = get_masked_tokens(tgt_b_full.detach(), mask_b)

        pred_aa = self.predictor_aa(vis_a, mask_a)
        pred_bb = self.predictor_bb(vis_b, mask_b)
        pred_ab = self.predictor_cross(vis_a, mask_b)
        pred_ba = self.predictor_cross(vis_b, mask_a)

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
            'e_uni_b': e_ub, 'r_uni_b': r_ub,
        }
