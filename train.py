import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm
import math

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.datasets.ben14k import BEN14KDataset
from src.datasets.cbrsir import CBRSIRDataset
from src.datasets.dsrsid import DSRSIDDataset
from src.models.copfm_retrieval import CopFMRetrieval
from src.losses import compute_total_loss
from src.utils.masking import random_block_mask
from src.wavelengths import get_wavelengths

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_dataloader(config, split='train'):
    dataset_name = config['data']['dataset_name']
    if 'BigEarthNet' in dataset_name:
        dataset = BEN14KDataset(config['data']['root_dir'], split=split, 
                                img_size=config['data']['img_size'], 
                                mask_ratio=config['model']['mask_ratio'])
    elif 'CBRSIR' in dataset_name:
        dataset = CBRSIRDataset(config['data']['root_dir'], split=split, 
                                img_size=config['data']['img_size'], 
                                mask_ratio=config['model']['mask_ratio'])
    elif 'DSRSID' in dataset_name:
        dataset = DSRSIDDataset(config['data']['root_dir'], split=split, 
                                img_size=config['data']['img_size'], 
                                mask_ratio=config['model']['mask_ratio'])
    else:
        raise ValueError("Unknown dataset")
        
    loader = DataLoader(dataset, batch_size=config['data']['batch_size'], 
                        shuffle=(split=='train'), num_workers=config['data']['num_workers'])
    return loader

def main(config_path):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config['logging']['use_wandb'] and WANDB_AVAILABLE:
        wandb.init(project=config['logging']['project'], config=config)
        
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    
    model = CopFMRetrieval(config['model']).to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=float(config['training']['lr']), 
                      weight_decay=float(config['training']['weight_decay']))
    
    # Simple cosine scheduler
    num_training_steps = config['training']['epochs']
    num_warmup_steps = config['training'].get('warmup_epochs', 0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    wl_a, bw_a = get_wavelengths(config['data']['modality_a'])
    wl_b, bw_b = get_wavelengths(config['data']['modality_b'])
    
    # Helper to get the correct keys from dummy loaders
    mod_a_key = config['data']['modality_a'].lower()
    mod_b_key = config['data']['modality_b'].lower()
    if 's1' in mod_a_key or 's2' in mod_a_key:
        mod_a_key, mod_b_key = 's1', 's2'
    elif 'rgb' in mod_a_key or 'sar' in mod_a_key:
        mod_a_key, mod_b_key = 'rgb', 'sar'
    elif 'pan' in mod_a_key or 'ms' in mod_a_key:
        mod_a_key, mod_b_key = 'pan', 'ms'
        
    os.makedirs('./checkpoints', exist_ok=True)
    
    print("Starting training loop...")
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_losses = {'total': 0, 'pred': 0, 'retr': 0, 'sigreg': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        for batch in pbar:
            B = batch['label'].shape[0]
            img_a = batch[mod_a_key].to(device)
            img_b = batch[mod_b_key].to(device)
            
            meta_a_key = mod_a_key + '_meta'
            meta_b_key = mod_b_key + '_meta'
            meta_info_a = batch[meta_a_key].to(device) if meta_a_key in batch else None
            meta_info_b = batch[meta_b_key].to(device) if meta_b_key in batch else None
            
            H, W = img_a.shape[-2], img_a.shape[-1]
            grid_h, grid_w = H // 16, W // 16
            
            mask_a = torch.stack([random_block_mask(grid_h, grid_w, config['model']['mask_ratio'], device=device) for _ in range(B)])
            mask_b = torch.stack([random_block_mask(grid_h, grid_w, config['model']['mask_ratio'], device=device) for _ in range(B)])
            
            output = model.forward_train(
                img_a, img_b, wl_a, wl_b, bw_a, bw_b, mask_a, mask_b,
                meta_info_a=meta_info_a, meta_info_b=meta_info_b
            )
            
            losses = compute_total_loss(output, **config['training']['loss_weights'])
            
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            
            for k in epoch_losses:
                epoch_losses[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()
                
            pbar.set_postfix({'loss': losses['total'].item() if not isinstance(losses['total'], float) else losses['total']})
            
            if config['logging']['use_wandb'] and WANDB_AVAILABLE:
                wandb.log({'train/total': losses['total'].item() if torch.is_tensor(losses['total']) else losses['total'],
                           'train/pred': losses['pred'],
                           'train/retr': losses['retr'],
                           'train/sigreg': losses['sigreg']})
                           
        scheduler.step()
        
        # Validation loop
        model.eval()
        val_losses = {'total': 0, 'pred': 0, 'retr': 0, 'sigreg': 0}
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{config['training']['epochs']}")
            for batch in val_pbar:
                B = batch['label'].shape[0]
                img_a = batch[mod_a_key].to(device)
                img_b = batch[mod_b_key].to(device)
                
                meta_a_key = mod_a_key + '_meta'
                meta_b_key = mod_b_key + '_meta'
                meta_info_a = batch[meta_a_key].to(device) if meta_a_key in batch else None
                meta_info_b = batch[meta_b_key].to(device) if meta_b_key in batch else None
                
                H, W = img_a.shape[-2], img_a.shape[-1]
                grid_h, grid_w = H // 16, W // 16
                
                mask_a = torch.stack([random_block_mask(grid_h, grid_w, config['model']['mask_ratio'], device=device) for _ in range(B)])
                mask_b = torch.stack([random_block_mask(grid_h, grid_w, config['model']['mask_ratio'], device=device) for _ in range(B)])
                
                output = model.forward_train(
                    img_a, img_b, wl_a, wl_b, bw_a, bw_b, mask_a, mask_b,
                    meta_info_a=meta_info_a, meta_info_b=meta_info_b
                )
                
                losses = compute_total_loss(output, **config['training']['loss_weights'])
                
                for k in val_losses:
                    val_losses[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()
                    
        if config['logging']['use_wandb'] and WANDB_AVAILABLE:
            wandb.log({'val/total': val_losses['total'] / len(val_loader),
                       'val/pred': val_losses['pred'] / len(val_loader),
                       'val/retr': val_losses['retr'] / len(val_loader),
                       'val/sigreg': val_losses['sigreg'] / len(val_loader)})
                       
        # Save checkpoint
        if (epoch + 1) % config['logging']['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, f"./checkpoints/epoch_{epoch+1}.pth")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    main(args.config)
