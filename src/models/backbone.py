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
sys.path.insert(0, copfm_repo_path)
sys.path.insert(0, os.path.join(copfm_repo_path, 'src'))

# Create __init__.py in reference repo's src dynamically if missing
init_file = os.path.join(copfm_repo_path, 'src', '__init__.py')
if not os.path.exists(init_file):
    try:
        with open(init_file, 'w') as f:
            pass
    except Exception:
        pass

# Pop all submodules of src to prevent import collisions
_original_src_mods = {}
for k in list(sys.modules.keys()):
    if k == 'src' or k.startswith('src.'):
        _original_src_mods[k] = sys.modules.pop(k)

try:
    from src.model_vit import vit_base_patch16
except ImportError as e:
    print(f"Warning: Could not import vit_base_patch16. Ensure Copernicus-FM is cloned. Error: {e}")
    vit_base_patch16 = None

# Clear reference repo imports of src and restore original src modules
for k in list(sys.modules.keys()):
    if k == 'src' or k.startswith('src.'):
        sys.modules.pop(k, None)

for k, mod in _original_src_mods.items():
    sys.modules[k] = mod

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
                meta_info: torch.Tensor = None):
        
        B = image.shape[0]
        device = image.device
        
        if meta_info is None:
            # Meta info: [lon, lat, time_doy, patch_area]
            # Since we don't have this info, set to NaN as per demo.ipynb
            meta_info = torch.full((B, 4), float('nan'), device=device)
        else:
            meta_info = meta_info.to(device)
        
        # Model forward
        _ = self.base_model(
            x=image, 
            meta_info=meta_info, 
            wave_list=wavelengths, 
            bandwidth=bandwidths, 
            language_embed=None, 
            input_mode='spectral', 
            kernel_size=16
        )
        
        if return_patch_tokens:
            # self._tokens shape is (B, 1 + N, D)
            # return N patch tokens (exclude cls token at index 0)
            return self._tokens[:, 1:, :]
        else:
            return self._tokens[:, 0, :]
