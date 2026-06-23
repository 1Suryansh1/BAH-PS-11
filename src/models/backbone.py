import sys
import os
import torch
import torch.nn as nn

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Safely import Copernicus-FM without shadowing the main `src` package
copfm_repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reference_repos/Copernicus-FM/Copernicus-FM'))

_original_sys_path = sys.path.copy()
_original_modules = sys.modules.copy()

# Remove the workspace root from sys.path to avoid import collision of 'src'
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path = [p for p in sys.path if os.path.abspath(p) != workspace_root]

# Insert both copfm_repo_path and its nested 'src' folder at the very beginning of sys.path
sys.path.insert(0, copfm_repo_path)
sys.path.insert(0, os.path.join(copfm_repo_path, 'src'))

# Clear any existing cached 'src' module or submodules
for k in list(sys.modules.keys()):
    if k == 'src' or k.startswith('src.'):
        del sys.modules[k]

try:
    from src.model_vit import vit_base_patch16, resize_abs_pos_embed
except ImportError as e:
    print(f"Warning: Could not import vit_base_patch16 or resize_abs_pos_embed. Ensure Copernicus-FM is cloned. Error: {e}")
    vit_base_patch16 = None
    resize_abs_pos_embed = None

# Restore sys.modules and sys.path
sys.modules.update(_original_modules)
for k in list(sys.modules.keys()):
    if k.startswith('src.') and k not in _original_modules:
        del sys.modules[k]

sys.path = _original_sys_path



class CopFMBackbone(nn.Module):
    def __init__(self, 
                 checkpoint_path: str,
                 freeze_mode: str = 'lora',   # 'full', 'lora', 'last4'
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 output_dim: int = 768):       # ViT-B hidden dim
        super().__init__()
        
        self.freeze_mode = freeze_mode
        
        # Instantiate the model with global_pool=False so it retains sequence features
        # num_classes=10 to match the checkpoint shape in the original codebase
        self.base_model = vit_base_patch16(num_classes=10, global_pool=False)
        
        # Load weights if checkpoint exists
        if os.path.exists(checkpoint_path):
            check_point = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in check_point:
                state_dict = check_point['model']
            else:
                state_dict = check_point
            self.base_model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Using initialized weights.")
            
        # Hook to extract patch tokens from norm layer
        self._tokens = None
        def hook(module, input, output):
            self._tokens = output
        self.base_model.norm.register_forward_hook(hook)
        
        if freeze_mode == 'full':
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        elif freeze_mode == 'last4':
            for param in self.base_model.parameters():
                param.requires_grad = False
            # Unfreeze last 4 layers
            for layer in self.base_model.blocks[-4:]:
                for param in layer.parameters():
                    param.requires_grad = True
                    
        elif freeze_mode == 'lora':
            if not PEFT_AVAILABLE:
                print("Warning: peft not installed. Proceeding with 'full' freeze mode.")
                for param in self.base_model.parameters():
                    param.requires_grad = False
            else:
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["qkv"],
                    lora_dropout=0.05,
                    bias="none"
                )
                self.base_model = get_peft_model(self.base_model, lora_config)
                
        # Print parameter counts
        total_params = sum(p.numel() for p in self.base_model.parameters())
        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        print(f"CopFMBackbone initialized. Mode: {freeze_mode}")
        print(f"Total Params: {total_params} | Trainable Params: {trainable_params}")

    def forward(self, 
                image: torch.Tensor,          # (B, C, H, W)
                wavelengths: list[float],     # list of C wavelength values
                bandwidths: list[float],      # list of C bandwidth values
                return_patch_tokens: bool = True,
                mask: torch.Tensor = None,    # (B, N_all) bool — True = masked
                meta_info: torch.Tensor = None):
        
        B = image.shape[0]
        device = image.device
        
        if meta_info is None:
            meta = torch.full((B, 4), float('nan'), device=device)
        else:
            meta = meta_info.to(device)
            
        if mask is None:
            # Standard full forward pass using base model
            _ = self.base_model(
                x=image, 
                meta_info=meta, 
                wave_list=wavelengths, 
                bandwidth=bandwidths, 
                language_embed=None, 
                input_mode='spectral', 
                kernel_size=16
            )
            tokens = self._tokens # (B, 1 + N, D)
            self._tokens = None  # Clear hook reference to prevent memory leak
            
            if return_patch_tokens:
                return tokens[:, 1:, :]
            else:
                return tokens[:, 0, :]
        else:
            # Masked forward pass: only process visible patches to prevent information leakage
            # Step 1: embed patches dynamically
            wavelist = torch.tensor(wavelengths, device=device).float()
            bandwidths_t = torch.tensor(bandwidths, device=device).float()
            self.base_model.waves = wavelist
            x, _ = self.base_model.patch_embed_spectral(image, wavelist, bandwidths_t, kernel_size=16)
            
            # Step 2: compute positional embeddings (spatial + temporal metadata if available)
            num_patches = x.size(1)
            num_patches_sqrt = int(num_patches ** 0.5)
            num_patches_sqrt_origin = int(self.base_model.num_patches ** 0.5)
            
            pos_embed = resize_abs_pos_embed(
                self.base_model.pos_embed, 
                num_patches_sqrt, 
                (num_patches_sqrt_origin, num_patches_sqrt_origin), 
                num_prefix_tokens=1
            ) # (1, 1 + N_all, D)
            
            # coord, scale and time pos embed
            lons, lats, times, areas = meta[:, 0], meta[:, 1], meta[:, 2], meta[:, 3]
            embed_dim = pos_embed.shape[-1]
            if torch.isnan(lons).any() or torch.isnan(lats).any():
                coord_embed = self.base_model.coord_token
            else:
                coord_embed = self.base_model.get_coord_pos_embed(lons, lats, embed_dim)
            coord_embed = self.base_model.coord_fc(coord_embed)
            
            if torch.isnan(areas).any():
                area_embed = self.base_model.scale_token
            else:   
                area_embed = self.base_model.get_area_pos_embed(areas, embed_dim)
            area_embed = self.base_model.scale_fc(area_embed)
            
            if torch.isnan(times).any():
                time_embed = self.base_model.time_token
            else:
                time_embed = self.base_model.get_time_pos_embed(times, embed_dim)
            time_embed = self.base_model.time_fc(time_embed)
            
            pos_embed = pos_embed + coord_embed + area_embed + time_embed
            
            # Step 3: Add position embeddings to patch tokens
            x = x + pos_embed[:, 1:, :] # (B, N_all, D)
            
            # Step 4: Keep only visible patch tokens to prevent information leakage
            # mask: (B, N_all) bool - True is masked, ~mask is visible
            x_vis = x[~mask].view(B, -1, embed_dim) # (B, N_vis, D)
            
            # Step 5: Append CLS token
            cls_token = self.base_model.cls_token + pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x_vis.shape[0], -1, -1)
            x_vis = torch.cat((cls_tokens, x_vis), dim=1) # (B, 1 + N_vis, D)
            
            # Step 6: Apply Transformer blocks
            for block in self.base_model.blocks:
                x_vis = block(x_vis)
                
            # Step 7: Apply layer normalization
            x_vis = self.base_model.norm(x_vis)
            self._tokens = None  # Clear hook reference to prevent memory leak
            
            if return_patch_tokens:
                return x_vis[:, 1:, :]
            else:
                return x_vis[:, 0, :]
