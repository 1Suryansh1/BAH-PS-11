import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
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
from src.utils.key_mapping import get_modality_key


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_dataloader(config, split='train'):
    dataset_name = config['data']['dataset_name']
    kwargs = dict(
        root_dir=config['data']['root_dir'],
        split=split,
        img_size=config['data']['img_size'],
        mask_ratio=config['model']['mask_ratio']
    )
    if 'BigEarthNet' in dataset_name:
        dataset = BEN14KDataset(**kwargs)
    elif 'CBRSIR' in dataset_name:
        dataset = CBRSIRDataset(**kwargs)
    elif 'DSRSID' in dataset_name:
        dataset = DSRSIDDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=(split == 'train')
    )
    return loader


def _to_device(x, device):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


def _loss_to_float(v):
    if torch.is_tensor(v):
        return v.item()
    return float(v)


def main(config_path):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config['logging']['use_wandb'] and WANDB_AVAILABLE:
        wandb.init(project=config['logging']['project'], config=config)

    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')

    model = CopFMRetrieval(config['model']).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "No trainable parameters found. Check freeze_mode config."

    optimizer = AdamW(
        trainable_params,
        lr=float(config['training']['lr']),
        weight_decay=float(config['training']['weight_decay'])
    )

    total_steps = config['training']['epochs'] * len(train_loader)
    warmup_steps = config['training'].get('warmup_epochs', 0) * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    wl_a, bw_a = get_wavelengths(config['data']['modality_a'])
    wl_b, bw_b = get_wavelengths(config['data']['modality_b'])

    dataset_name = config['data']['dataset_name']
    mod_a_key = get_modality_key(dataset_name, config['data']['modality_a'])
    mod_b_key = get_modality_key(dataset_name, config['data']['modality_b'])

    CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("Starting training loop...")
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_losses = {'total': 0.0, 'pred': 0.0, 'retr': 0.0, 'sigreg': 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        for batch in pbar:
            B = batch['label'].shape[0]
            img_a = batch[mod_a_key].to(device)
            img_b = batch[mod_b_key].to(device)

            H, W = img_a.shape[-2], img_a.shape[-1]
            grid_h, grid_w = H // 16, W // 16

            mask_a = torch.stack([
                random_block_mask(grid_h, grid_w, config['model']['mask_ratio'], device=device)
                for _ in range(B)
            ])
            mask_b = torch.stack([
                random_block_mask(grid_h, grid_w, config['model']['mask_ratio'], device=device)
                for _ in range(B)
            ])

            meta_a = _to_device(batch.get('meta_' + mod_a_key), device)
            meta_b = _to_device(batch.get('meta_' + mod_b_key), device)

            output = model.forward_train(
                img_a, img_b, wl_a, wl_b, bw_a, bw_b,
                mask_a, mask_b, meta_a=meta_a, meta_b=meta_b
            )

            losses = compute_total_loss(output, **config['training']['loss_weights'])

            if torch.isnan(losses['total']):
                raise RuntimeError(f"NaN loss at epoch {epoch+1}, step {pbar.n}. Stopping.")

            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            
            # Removed model.update_target_ema() as per instructions
            
            for k in epoch_losses:
                epoch_losses[k] += _loss_to_float(losses[k])

            pbar.set_postfix({'loss': _loss_to_float(losses['total'])})

            if config['logging']['use_wandb'] and WANDB_AVAILABLE:
                wandb.log({
                    'train/total': _loss_to_float(losses['total']),
                    'train/pred':  _loss_to_float(losses['pred']),
                    'train/retr':  _loss_to_float(losses['retr']),
                    'train/sigreg': _loss_to_float(losses['sigreg']),
                })

        model.eval()
        val_losses = {'total': 0.0, 'pred': 0.0, 'retr': 0.0, 'sigreg': 0.0}

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{config['training']['epochs']}")
            for batch in val_pbar:
                img_a = batch[mod_a_key].to(device)
                img_b = batch[mod_b_key].to(device)
                meta_a = _to_device(batch.get('meta_' + mod_a_key), device)
                meta_b = _to_device(batch.get('meta_' + mod_b_key), device)

                e_a = model.get_retrieval_embedding(img_a, wl_a, bw_a, mode='cross', meta=meta_a)
                e_b = model.get_retrieval_embedding(img_b, wl_b, bw_b, mode='cross', meta=meta_b)

                from src.losses.nce import info_nce_loss
                val_loss = info_nce_loss(e_a, e_b)
                val_losses['total'] += val_loss.item()

        if config['logging']['use_wandb'] and WANDB_AVAILABLE:
            n = max(len(val_loader), 1)
            wandb.log({
                'val/total':   val_losses['total'] / n,
            })

        if (epoch + 1) % config['logging']['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, os.path.join(CKPT_DIR, f"epoch_{epoch+1}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)
