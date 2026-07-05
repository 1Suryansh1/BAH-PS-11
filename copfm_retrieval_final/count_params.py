import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
from src.models.copfm_retrieval import CopFMRetrieval

config = {
    'backbone_checkpoint': 'non_existent_file.pth', # will warn and initialize randomly
    'freeze_mode': 'lora',
    'predictor_depth': 6,
    'retrieval_dim': 256
}

# instantiate
model = CopFMRetrieval(config)

def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

print("\n--- EXACT TRAINABLE PARAMETER COUNTS ---")

# 1. Stem (Dynamic Wavelength Embeddings)
# In PEFT, we need to check if patch_embed_spectral has requires_grad
stem_trainable = count_params(model.backbone.base_model.base_model.model.patch_embed_spectral) if hasattr(model.backbone.base_model, 'base_model') else 0
print(f"1. Stems (Dynamic Wavelength Embeddings): {stem_trainable:,}")

# 2. LoRA Adapters
# We can just count all trainable params in the backbone since only LoRA is trainable
lora_trainable = count_params(model.backbone) - stem_trainable
print(f"2. LoRA Adapters (Rank 64): {lora_trainable:,}")

# 3. JEPA Latent Predictors
aa = count_params(model.predictor_aa)
bb = count_params(model.predictor_bb)
cross = count_params(model.predictor_cross)
print(f"3. JEPA Latent Predictors:")
print(f"   - Same-modal (AA + BB): {aa + bb:,} ({aa:,} each)")
print(f"   - Cross-modal (Shared): {cross:,}")
print(f"   - Total Predictors: {aa + bb + cross:,}")

# 4. Decoupled Retrieval Heads
heads = count_params(model.retrieval_heads)
print(f"4. Decoupled Retrieval Heads: {heads:,}")

# Total
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"--- TOTAL TRAINABLE PARAMETERS: {total:,} ---")
