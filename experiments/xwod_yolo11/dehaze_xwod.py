import argparse
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


def iter_prefixed_images(images_dir, prefix, limit=None):
    images = sorted(
        path for path in Path(images_dir).iterdir()
        if path.is_file()
        and path.suffix.lower() in IMG_EXTS
        and path.name.startswith(prefix)
    )
    if limit is not None:
        images = images[:limit]
    return images


def load_rgb_tensor(path, size):
    image = Image.open(path).convert('RGB').resize(size, Image.BICUBIC)
    return transforms.ToTensor()(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--xwod-root', default='./data/XWOD')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--prefix', default='fog_')
    parser.add_argument('--reference', required=True, help='Clear reference/GT image used for style code.')
    parser.add_argument('--output-root', default='./output/xwod_dehazed')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    cfg = get_config(args.config)
    cfg['cfg_path'] = args.config

    trainer = Trainer(cfg)
    trainer.load_model(args.checkpoint, flow_only=False)
    trainer.model.eval()
    trainer.encoder.eval()

    model_size = (cfg['img_w'], cfg['img_h'])
    reference = load_rgb_tensor(args.reference, model_size).unsqueeze(0).cuda()
    images_dir = Path(args.xwod_root) / args.split / 'images'
    images = iter_prefixed_images(images_dir, args.prefix, args.limit)
    out_dir = Path(args.output_root) / args.split / 'images'
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        style_code = trainer.encoder.cat_tensor(reference)
        for path in tqdm(images, desc=f'Dehaze XWOD {args.split}/{args.prefix}'):
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
            save_image(output[0], out_dir / path.name)

    print(f'Saved {len(images)} dehazed images to {out_dir}')


if __name__ == '__main__':
    main()
