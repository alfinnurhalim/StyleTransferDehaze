import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from model.trainers.trainer import Trainer
from model.utils.utils import get_config


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def iter_images(root, condition, split, limit=None):
    base = Path(root) / 'images' / condition / split
    images = sorted(p for p in base.rglob('*') if p.suffix.lower() in IMG_EXTS)
    if limit is not None:
        images = images[:limit]
    return images


def load_rgb(path, size):
    img = Image.open(path).convert('RGB').resize(size, Image.BICUBIC)
    return transforms.ToTensor()(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--acdc-root', default='./data/acdc')
    parser.add_argument('--condition', default='fog', choices=['fog', 'night', 'rain', 'snow'])
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--reference', required=True, help='Clear reference/GT image used for style code.')
    parser.add_argument('--output-root', default='./output/acdc_dehazed')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    cfg = get_config(args.config)
    cfg['cfg_path'] = args.config

    trainer = Trainer(cfg)
    trainer.load_model(args.checkpoint, flow_only=False)
    trainer.model.eval()
    trainer.encoder.eval()

    model_size = (cfg['img_w'], cfg['img_h'])
    ref = load_rgb(args.reference, model_size).unsqueeze(0).cuda()

    images = iter_images(args.acdc_root, args.condition, args.split, args.limit)
    out_root = Path(args.output_root)

    with torch.no_grad():
        style_code = trainer.encoder.cat_tensor(ref)
        for path in tqdm(images, desc=f'Dehaze {args.condition}/{args.split}'):
            rel = path.relative_to(Path(args.acdc_root) / 'images')
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            original = Image.open(path).convert('RGB')
            source = original.resize(model_size, Image.BICUBIC)
            source_tensor = transforms.ToTensor()(source).unsqueeze(0).cuda()

            z_c = trainer.model.glow(source_tensor, forward=True)
            output = trainer.model.glow(z_c, forward=False, style=style_code)
            output = torch.clamp(output, 0, 1).cpu()

            output = torch.nn.functional.interpolate(
                output,
                size=(original.height, original.width),
                mode='bilinear',
                align_corners=False,
            )
            save_image(output[0], out_path)

    print(f'Saved dehazed images to {out_root}')


if __name__ == '__main__':
    main()
