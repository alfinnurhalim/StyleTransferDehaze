import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', default='fog', choices=['fog', 'night', 'rain', 'snow'])
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--acdc-root', default='./data/acdc')
    parser.add_argument('--yolo-model', default='yolo11n.pt')
    parser.add_argument('--dehazed-root', default=None)
    parser.add_argument('--output-root', default='./output/acdc_yolo11_comparison')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--imgsz', type=int, default=1280)
    args = parser.parse_args()

    base_cmd = [
        sys.executable,
        'experiments/acdc_yolo11/eval_yolo11_acdc.py',
        '--acdc-root', args.acdc_root,
        '--condition', args.condition,
        '--split', args.split,
        '--model', args.yolo_model,
        '--imgsz', str(args.imgsz),
    ]
    if args.limit is not None:
        base_cmd += ['--limit', str(args.limit)]

    original_out = Path(args.output_root) / args.condition / args.split / 'original'
    run(base_cmd + [
        '--images-root', str(Path(args.acdc_root) / 'images'),
        '--output-dir', str(original_out),
    ])

    if args.dehazed_root:
        dehazed_out = Path(args.output_root) / args.condition / args.split / 'dehazed'
        run(base_cmd + [
            '--images-root', args.dehazed_root,
            '--output-dir', str(dehazed_out),
        ])


if __name__ == '__main__':
    main()
