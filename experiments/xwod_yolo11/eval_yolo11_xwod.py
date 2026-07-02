import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolo11n.pt')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output-root', default='./output/xwod_yolo11_comparison')
    parser.add_argument('--name', required=True)
    parser.add_argument('--imgsz', type=int, default=256)
    args = parser.parse_args()

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        split='test',
        imgsz=args.imgsz,
        project=args.output_root,
        name=args.name,
        exist_ok=True,
    )

    summary = {
        'model': args.model,
        'data': args.data,
        'imgsz': args.imgsz,
        'box_map': float(metrics.box.map),
        'box_map50': float(metrics.box.map50),
        'box_map75': float(metrics.box.map75),
        'box_mp': float(metrics.box.mp),
        'box_mr': float(metrics.box.mr),
    }
    out_dir = Path(args.output_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
