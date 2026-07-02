import argparse
import os
import shutil
from pathlib import Path

import yaml


def link_or_copy(src, dst, copy=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def read_names(data_yaml):
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']


def write_yaml(path, dataset_dir, names):
    with open(path, 'w') as f:
        yaml.safe_dump(
            {
                'path': str(Path(dataset_dir).resolve()),
                'train': 'test/images',
                'val': 'test/images',
                'test': 'test/images',
                'nc': len(names),
                'names': names,
            },
            f,
            sort_keys=False,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xwod-root', default='./data/XWOD')
    parser.add_argument('--image-root', required=True, help='Source images directory, e.g. data/XWOD/test/images or output/xwod_dehazed/test/images.')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--prefix', default='fog_')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--copy', action='store_true', help='Copy files instead of symlinking.')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    xwod_root = Path(args.xwod_root)
    image_root = Path(args.image_root)
    label_root = xwod_root / args.split / 'labels'
    output_root = Path(args.output_root)
    output_images = output_root / 'test' / 'images'
    output_labels = output_root / 'test' / 'labels'
    names = read_names(xwod_root / 'data.yaml')

    images = sorted(
        path for path in image_root.iterdir()
        if path.is_file() and path.name.startswith(args.prefix)
    )
    if args.limit is not None:
        images = images[:args.limit]

    kept = 0
    for image_path in images:
        label_path = label_root / f'{image_path.stem}.txt'
        if not label_path.exists():
            continue
        link_or_copy(image_path, output_images / image_path.name, copy=args.copy)
        link_or_copy(label_path, output_labels / label_path.name, copy=args.copy)
        kept += 1

    yaml_path = output_root / 'data.yaml'
    write_yaml(yaml_path, output_root, names)
    print(f'Prepared {kept} images at {output_root}')
    print(f'YOLO data yaml: {yaml_path}')


if __name__ == '__main__':
    main()
