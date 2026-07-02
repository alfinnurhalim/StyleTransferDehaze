import argparse
import os
import re
import shutil
from pathlib import Path


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
SPLITS = {
    'train': range(1, 36),
    'val': range(36, 41),
    'test': range(41, 46),
}


def image_id(path):
    match = re.match(r'(\d+)', path.name)
    if not match:
        raise ValueError(f'Cannot parse numeric id from {path}')
    return int(match.group(1))


def collect_by_id(root):
    files = sorted(p for p in Path(root).iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    return {image_id(p): p for p in files}


def reset_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src, dst, copy=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def make_ohaze(raw_root, output_root, copy=False):
    hazy = collect_by_id(Path(raw_root) / 'hazy')
    gt = collect_by_id(Path(raw_root) / 'GT')
    ids = sorted(set(hazy) & set(gt))
    missing = sorted((set(hazy) ^ set(gt)))
    if missing:
        raise RuntimeError(f'Missing O-HAZE pairs for ids: {missing}')

    for phase in SPLITS:
        reset_dir(Path(output_root) / f'{phase}A')
        reset_dir(Path(output_root) / f'{phase}B')

    manifest = []
    for phase, split_ids in SPLITS.items():
        for idx in split_ids:
            if idx not in ids:
                continue
            out_name = f'{idx:02d}'
            a_dst = Path(output_root) / f'{phase}A' / f'{out_name}_hazy{hazy[idx].suffix.lower()}'
            b_dst = Path(output_root) / f'{phase}B' / f'{out_name}_GT{gt[idx].suffix.lower()}'
            link_or_copy(hazy[idx], a_dst, copy=copy)
            link_or_copy(gt[idx], b_dst, copy=copy)
            manifest.append((phase, idx, str(hazy[idx]), str(gt[idx])))

    return manifest


def iter_pairs(dataset_root, phase):
    a_dir = Path(dataset_root) / f'{phase}A'
    b_dir = Path(dataset_root) / f'{phase}B'
    a_files = sorted(p for p in a_dir.iterdir() if p.is_file() or p.is_symlink())
    b_files = sorted(p for p in b_dir.iterdir() if p.is_file() or p.is_symlink())
    if len(a_files) != len(b_files):
        raise RuntimeError(f'Pair count mismatch for {dataset_root} {phase}: {len(a_files)} vs {len(b_files)}')
    return zip(a_files, b_files)


def make_combined(ohaze_dataset, nhhaze_dataset, output_root, copy=False):
    for phase in SPLITS:
        reset_dir(Path(output_root) / f'{phase}A')
        reset_dir(Path(output_root) / f'{phase}B')

        count = 0
        for prefix, dataset_root in [('ohaze', ohaze_dataset), ('nhhaze', nhhaze_dataset)]:
            phase_a = Path(output_root) / f'{phase}A'
            phase_b = Path(output_root) / f'{phase}B'
            source_phase = phase
            if not (Path(dataset_root) / f'{source_phase}A').exists():
                continue
            for a_src, b_src in iter_pairs(dataset_root, source_phase):
                count += 1
                stem = f'{prefix}_{count:03d}'
                link_or_copy(a_src.resolve(), phase_a / f'{stem}_hazy{a_src.suffix.lower()}', copy=copy)
                link_or_copy(b_src.resolve(), phase_b / f'{stem}_GT{b_src.suffix.lower()}', copy=copy)


def count_split(root):
    rows = []
    for phase in SPLITS:
        a_count = len(list((Path(root) / f'{phase}A').glob('*'))) if (Path(root) / f'{phase}A').exists() else 0
        b_count = len(list((Path(root) / f'{phase}B').glob('*'))) if (Path(root) / f'{phase}B').exists() else 0
        rows.append((phase, a_count, b_count))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-ohaze', default='./data/OHAZE')
    parser.add_argument('--nhhaze', default='./data/Dataset_NHHAZE')
    parser.add_argument('--ohaze-out', default='./data/Dataset_OHAZE')
    parser.add_argument('--combined-out', default='./data/Dataset_OHAZE_NHHAZE')
    parser.add_argument('--copy', action='store_true', help='Copy files instead of symlinking.')
    args = parser.parse_args()

    make_ohaze(args.raw_ohaze, args.ohaze_out, copy=args.copy)
    make_combined(args.ohaze_out, args.nhhaze, args.combined_out, copy=args.copy)

    print('Prepared O-HAZE dataset:')
    for phase, a_count, b_count in count_split(args.ohaze_out):
        print(f'  {phase}: {a_count} hazy / {b_count} GT')

    print('Prepared O-HAZE + NH-HAZE combined dataset:')
    for phase, a_count, b_count in count_split(args.combined_out):
        print(f'  {phase}: {a_count} hazy / {b_count} GT')


if __name__ == '__main__':
    main()
