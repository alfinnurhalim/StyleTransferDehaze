import argparse
import csv
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.dataset_ImagePair import ImagePairDataset
from model.trainers.trainer import Trainer
from model.utils.utils import get_config


def append_metrics_csv(path, row):
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    return value


def write_metrics_summary(path, summary):
    with open(path, 'w') as f:
        json.dump(to_jsonable(summary), f, indent=2)


def default_checkpoint_path(cfg):
    run_dir = os.path.join(
        cfg['output'],
        f"{cfg['job_name']}_{int(cfg.get('keep_ratio', 1) * 100)}_{cfg['n_flow']}_{cfg['n_block']}",
    )
    return os.path.join(run_dir, 'model_save', f"{cfg['job_name']}.pth.tar")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--phase', default='val', choices=['val', 'test'])
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    args_cli = parser.parse_args()

    cfg = get_config(args_cli.config)
    cfg['cfg_path'] = args_cli.config

    root_dir = cfg.get('root_dir', f"./dataset/{cfg['job_name']}")
    img_size = (cfg['img_h'], cfg['img_w'])
    batch_size = args_cli.batch_size or cfg['batch_size']
    pairing = cfg.get(f'{args_cli.phase}_pairing', cfg.get('test_pairing', 'direct'))

    dataset = ImagePairDataset(
        root_dir=root_dir,
        phase=args_cli.phase,
        augment=False,
        return_gt=True,
        stage=cfg['stage'],
        img_size=img_size,
        pairing=pairing,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    trainer = Trainer(cfg)
    checkpoint_path = args_cli.checkpoint or default_checkpoint_path(cfg)
    trainer.load_model(checkpoint_path, flow_only=False)

    avg_psnr = []
    avg_ssim = []
    eval_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Validate {args_cli.phase}")
    for batch_id, (source_image, style_image, gt_image) in eval_bar:
        psnr, ssim = trainer.test(
            epoch=0,
            batch_id=batch_id,
            content_imgs=source_image,
            style_imgs=style_image,
            gt_imgs=gt_image,
            len_data=len(loader),
            suffix=args_cli.phase,
        )
        avg_psnr.append(psnr)
        avg_ssim.append(ssim)

    mean_psnr = float(sum(avg_psnr) / len(avg_psnr))
    mean_ssim = float(sum(avg_ssim) / len(avg_ssim))
    row = {
        'epoch': 0,
        'split': args_cli.phase,
        'psnr': mean_psnr,
        'ssim': mean_ssim,
        'checkpoint': checkpoint_path,
    }

    metrics_path = os.path.join(trainer.log_path, f'{args_cli.phase}_metrics_manual.csv')
    summary_path = os.path.join(trainer.log_path, f'{args_cli.phase}_latest_manual.json')
    append_metrics_csv(metrics_path, row)
    write_metrics_summary(summary_path, row)

    print(f"{args_cli.phase.upper()} Avg PSNR: {mean_psnr:.4f}, Avg SSIM: {mean_ssim:.4f}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == '__main__':
    main()
