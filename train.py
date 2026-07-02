import os
import cv2
import numpy as np
import wandb
import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from torch.utils.data import DataLoader

from model.utils.utils import get_config
from model.trainers.trainer import Trainer
from dataset.dataset_ImagePair import ImagePairDataset
from logger import TrainingLogger

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/Dataset_OHAZE.yaml')
args_cli = parser.parse_args()

cfg_path = args_cli.config

wandb_project = 'TESIS_PAPER'

args = get_config(cfg_path)
args['cfg_path'] = cfg_path

code_name = args['job_name']
root_dir = args.get('root_dir', f'./dataset/{code_name}')
resume = args['resume']
img_size = (args['img_h'],args['img_w']) #OpenCV
batch_size = args['batch_size']
wandb_key = args['wandb']
val_freq = args.get('val_freq', 10)
test_freq = args.get('test_freq', 10)


def has_split(root, phase):
    return (
        os.path.isdir(os.path.join(root, f'{phase}A')) and
        os.path.isdir(os.path.join(root, f'{phase}B'))
    )


def append_metrics_csv(path, row):
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_metrics_summary(path, summary):
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)


def mean_losses(loss_rows):
    arr = np.array(loss_rows, dtype=np.float64)
    return {
        'loss_total': float(arr[:, 0].mean()),
        'loss_content': float(arr[:, 1].mean()),
        'loss_style': float(arr[:, 2].mean()),
        'loss_recon': float(arr[:, 3].mean()),
        'loss_pixel': float(arr[:, 4].mean()),
        'loss_smooth': float(arr[:, 5].mean()),
        'loss_scm': float(arr[:, 6].mean()),
    }


def update_training_plots(log_path, train_metrics_path, validation_metrics_path, test_metrics_path):
    plot_dir = os.path.join(log_path, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    if os.path.exists(train_metrics_path):
        train_df = pd.read_csv(train_metrics_path)
        if not train_df.empty:
            plt.figure(figsize=(9, 5))
            for col in ['loss_total', 'loss_recon', 'loss_content', 'loss_smooth']:
                if col in train_df:
                    plt.plot(train_df['epoch'], train_df[col], marker='o', label=col)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Curves')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'train_losses.png'), dpi=180)
            plt.close()

            plt.figure(figsize=(8, 4.5))
            plt.plot(train_df['epoch'], train_df['lr'], marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Learning rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, 'learning_rate.png'), dpi=180)
            plt.close()

    metric_frames = []
    for path in [validation_metrics_path, test_metrics_path]:
        if os.path.exists(path):
            metric_frames.append(pd.read_csv(path))
    if metric_frames:
        metrics_df = pd.concat(metric_frames, ignore_index=True)
        for metric in ['psnr', 'ssim']:
            plt.figure(figsize=(8, 4.5))
            for split, group in metrics_df.groupby('split'):
                plt.plot(group['epoch'], group[metric], marker='o', label=split)
            plt.xlabel('Epoch')
            plt.ylabel(metric.upper())
            plt.title(f'{metric.upper()} Curves')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{metric}_curves.png'), dpi=180)
            plt.close()


def evaluate_loader(trainer, loader, epoch, split_name, metrics_path, summary_path):
    print(f'\n\nValidating on {split_name} set ......')
    avg_psnr = []
    avg_ssim = []
    eval_bar = tqdm(enumerate(loader),
                    total=len(loader),
                    desc=f"{split_name} Epoch {epoch}")

    for batch_id, (source_image, style_image, gt_image) in eval_bar:
        psnr, ssim = trainer.test(
            epoch,
            batch_id,
            source_image,
            style_image,
            gt_image,
            len_data=len(loader),
            suffix=split_name,
        )
        avg_psnr.append(psnr)
        avg_ssim.append(ssim)

    mean_psnr = sum(avg_psnr) / len(avg_psnr)
    mean_ssim = sum(avg_ssim) / len(avg_ssim)
    row = {
        'epoch': epoch,
        'split': split_name,
        'psnr': mean_psnr,
        'ssim': mean_ssim,
    }
    append_metrics_csv(metrics_path, row)
    write_metrics_summary(summary_path, row)

    print(f"\nEpoch: {epoch} - {split_name.upper()} Avg PSNR: {mean_psnr:.4f}, Avg SSIM: {mean_ssim:.4f} ")
    return mean_psnr, mean_ssim

if wandb_key is not None:
    print(f'\n\nSetting up wandb\n\n')
    wandb.login(key=wandb_key)

    wandb_run = wandb.init(
        project=wandb_project,
        entity="alfin-nurhalim",
        name=code_name,
        config=args,
    )

train_dataset_mixed = ImagePairDataset(root_dir=root_dir, 
                                        phase='train',
                                        augment=args['aug'],
                                        stage=args['stage'],
                                        img_size=img_size,
                                        pairing=args.get('train_pairing', 'mix'))

test_dataset_mixed = ImagePairDataset(root_dir=root_dir, 
                                        phase='test',
                                        augment=False, 
                                        return_gt=True,
                                        stage=args['stage'],
                                        img_size=img_size,
                                        pairing=args.get('test_pairing', 'direct'),
                                        )

val_dataset_mixed = None
if has_split(root_dir, 'val'):
    val_dataset_mixed = ImagePairDataset(root_dir=root_dir,
                                        phase='val',
                                        augment=False,
                                        return_gt=True,
                                        stage=args['stage'],
                                        img_size=img_size,
                                        pairing=args.get('val_pairing', args.get('test_pairing', 'direct')),
                                        )

print(f'\n\nTrain Dataset : {len(train_dataset_mixed)}')
if val_dataset_mixed is not None:
    print(f'Val Dataset : {len(val_dataset_mixed)}')
print(f'Test Dataset : {len(test_dataset_mixed)}\n\n')
train_loader_mixed = DataLoader(train_dataset_mixed, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader_mixed = None
if val_dataset_mixed is not None:
    val_loader_mixed = DataLoader(val_dataset_mixed, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader_mixed = DataLoader(test_dataset_mixed, batch_size=batch_size, shuffle=False, num_workers=2)

print('attention:',args['attention'])
print('FLOW :',args['n_flow'], 'BLOCK: ',args['n_block'])
print('epoch: ',args['total_epoch'])
print('\n\n')

trainer = Trainer(args)

if resume is not None:
  trainer.load_model(resume,flow_only=False)

training_logger = TrainingLogger(trainer.log_path)
validation_metrics_path = os.path.join(trainer.log_path, 'validation_metrics.csv')
validation_summary_path = os.path.join(trainer.log_path, 'validation_latest.json')
test_metrics_path = os.path.join(trainer.log_path, 'test_metrics.csv')
test_summary_path = os.path.join(trainer.log_path, 'test_latest.json')
train_metrics_path = os.path.join(trainer.log_path, 'training_metrics.csv')
best_metrics_path = os.path.join(trainer.log_path, 'best_metrics.json')
best_val_psnr = float('-inf')
best_val_ssim = float('-inf')
best_metrics = {}

for epoch in range(args['total_epoch']):
    progress_bar = tqdm(enumerate(train_loader_mixed),
                        total=len(train_loader_mixed),
                        desc=f"Epoch {epoch}")

    last_loss = None
    epoch_loss_rows = []
    for batch_id, (source_image, style_image) in progress_bar:
        loss_list = trainer.train(batch_id, source_image, style_image, epoch)
        
        loss,loss_c,loss_s,loss_r,loss_p,loss_smooth,loss_scm = loss_list if loss_list is not None else [0,0,0,0,0]

        last_loss = loss  # Update last_loss continuously
        epoch_loss_rows.append([loss, loss_c, loss_s, loss_r, loss_p, loss_smooth, loss_scm])

        progress_bar.set_postfix({'Loss': f'{loss:.4f}',
                                  'Loss_c': f'{loss_c:.4f}',
                                  'Loss_s': f'{loss_s:.4f}',
                                  'Loss_r': f'{loss_r:.4f}',
                                  'loss_p': f'{loss_p:.4f}',
                                  'Loss_smooth': f'{loss_smooth:.4f}',
                                  'loss_scm': f'{loss_scm:.4f}'})

    if loss_list is not None and wandb_key is not None:
            wandb.log({
                'train/loss_total': loss,
                'train/loss_content': loss_c,
                'train/loss_style': loss_s,
                'train/loss_restoration': loss_r,
                'train/loss_pixel': loss_p,
                'train/loss_smooth': loss_smooth,
                'train/loss_scm': loss_scm,
            },step=epoch)
            
    if last_loss is not None:
        training_logger.log_epoch_loss(epoch, last_loss)

    print(f"\nEpoch: {epoch} - Last Batch Loss: {last_loss:.4f}")
    train_row = {
        'epoch': epoch,
        'lr': trainer.get_lr(),
        'last_loss': last_loss,
    }
    if epoch_loss_rows:
        train_row.update(mean_losses(epoch_loss_rows))
    append_metrics_csv(train_metrics_path, train_row)

    if val_loader_mixed is not None and val_freq > 0 and epoch % val_freq == 0:
      val_psnr, val_ssim = evaluate_loader(
          trainer,
          val_loader_mixed,
          epoch,
          'val',
          validation_metrics_path,
          validation_summary_path,
      )
      if wandb_key is not None:
          wandb.log({
              'val/psnr': val_psnr,
              'val/ssim': val_ssim
          }, step=epoch)
      trainer.save_model(
          f'epoch_{epoch:03d}_val',
          epoch=epoch,
          metrics={'val_psnr': val_psnr, 'val_ssim': val_ssim},
      )
      if val_psnr > best_val_psnr:
          best_val_psnr = val_psnr
          best_metrics['best_val_psnr'] = {'epoch': epoch, 'psnr': val_psnr, 'ssim': val_ssim}
          trainer.save_model(
              'best_val_psnr',
              epoch=epoch,
              metrics={'val_psnr': val_psnr, 'val_ssim': val_ssim},
          )
      if val_ssim > best_val_ssim:
          best_val_ssim = val_ssim
          best_metrics['best_val_ssim'] = {'epoch': epoch, 'psnr': val_psnr, 'ssim': val_ssim}
          trainer.save_model(
              'best_val_ssim',
              epoch=epoch,
              metrics={'val_psnr': val_psnr, 'val_ssim': val_ssim},
          )
      write_metrics_summary(best_metrics_path, best_metrics)

    if test_freq > 0 and epoch % test_freq == 0:
      test_psnr, test_ssim = evaluate_loader(
          trainer,
          test_loader_mixed,
          epoch,
          'test',
          test_metrics_path,
          test_summary_path,
      )
      if wandb_key is not None:
          wandb.log({
              'test/psnr': test_psnr,
              'test/ssim': test_ssim
          },step=epoch)

    trainer.step_lr_scheduler()
    print(f"Learning rate after epoch {epoch}: {trainer.get_lr():.8f}")
    trainer.save_model('last', epoch=epoch, metrics=train_row)
    update_training_plots(trainer.log_path, train_metrics_path, validation_metrics_path, test_metrics_path)

    print('\n\n')

if wandb_key is not None:
    wandb_run.finish()
