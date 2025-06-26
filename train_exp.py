import os
import cv2
import numpy as np
import wandb
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader

from model.utils.utils import get_config
from model.trainers.trainer import Trainer
from dataset.dataset_ImagePair import ImagePairDataset
from logger import TrainingLogger

# === Configs ===
code_name = 'Dataset_OHAZE'
root_dir = f'./dataset/Dataset_OHAZE'
cfg_path = f'./config/Dataset_OHAZE.yaml'
wandb_project = 'TESIS_PAPER'

# === Load Base Config ===
args = get_config(cfg_path)
args['cfg_path'] = cfg_path
resume = args['resume']
img_size = (args['img_h'], args['img_w'])
batch_size = args['batch_size']
wandb_key = args['wandb']

# === wandb Login ===
if wandb_key is not None:
    print('\nSetting up wandb...\n')
    wandb.login(key=wandb_key)

# === Dataset (shared across all runs) ===
train_dataset = ImagePairDataset(
    root_dir=root_dir,
    phase='train',
    augment=args['aug'],
    stage=args['stage'],
    img_size=img_size,
    pairing='mix'
)
test_dataset = ImagePairDataset(
    root_dir=root_dir,
    phase='test',
    augment=False,
    return_gt=True,
    stage=args['stage'],
    img_size=img_size,
    pairing='direct'
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# === Grid Search Loops ===
block_list = [1, 2, 3]
flow_list = [2, 4, 6]

for n_block in block_list:
    for n_flow in flow_list:
        config = deepcopy(args)
        config['n_block'] = n_block
        config['n_flow'] = n_flow

        run_name = f"OHAZE-b{n_block}-f{n_flow}"
        print(f"\n=== Running: {run_name} ===")
        print('attention:', config['attention'])
        print('epochs  :', config['total_epoch'])

        if wandb_key is not None:
            wandb_run = wandb.init(
                project=wandb_project,
                entity="alfin-nurhalim",
                name=run_name,
                config=config,
                reinit=True
            )

        trainer = Trainer(config)
        if resume:
            trainer.load_model(resume, flow_only=False)
        logger = TrainingLogger(trainer.log_path)

        for epoch in range(config['total_epoch']):
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[{run_name}] Epoch {epoch}")
            last_loss = None

            for batch_id, (hazy, ref) in progress_bar:
                loss_list = trainer.train(batch_id, hazy, ref, epoch)
                if loss_list is not None:
                    loss, loss_c, loss_s, loss_r, loss_p, loss_smooth, loss_scm = loss_list
                    last_loss = loss
                    progress_bar.set_postfix({
                        'Loss': f'{loss:.4f}',
                        'L_c': f'{loss_c:.4f}',
                        'L_s': f'{loss_s:.4f}',
                        'L_r': f'{loss_r:.4f}'
                    })
            if wandb_key:
                wandb.log({
                    'train/loss_total': loss,
                    'train/loss_content': loss_c,
                    'train/loss_style': loss_s,
                    'train/loss_restoration': loss_r,
                    'train/loss_pixel': loss_p,
                    'train/loss_smooth': loss_smooth,
                    'train/loss_scm': loss_scm,
                }, step=epoch)

            if last_loss is not None:
                logger.log_epoch_loss(epoch, last_loss)

            if epoch % 10 == 0:
                print(f"\nTesting {run_name} at epoch {epoch}...")
                avg_psnr, avg_ssim = [], []
                test_bar = tqdm(enumerate(test_loader), total=len(test_loader))
                for batch_id, (hazy, ref, gt) in test_bar:
                    psnr, ssim = trainer.test(epoch, batch_id, hazy, ref, gt, len_data=len(test_loader))
                    avg_psnr.append(psnr)
                    avg_ssim.append(ssim)

                mean_psnr = sum(avg_psnr) / len(avg_psnr)
                mean_ssim = sum(avg_ssim) / len(avg_ssim)
                print(f"[{run_name}] Epoch {epoch} - PSNR: {mean_psnr:.4f}, SSIM: {mean_ssim:.4f}")
                if wandb_key:
                    wandb.log({'test/psnr': mean_psnr, 'test/ssim': mean_ssim}, step=epoch)

        if wandb_key:
            wandb_run.finish()
