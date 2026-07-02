import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


COCO_TO_ACDC = {
    0: 24,   # person
    1: 33,   # bicycle
    2: 26,   # car
    3: 32,   # motorcycle
    5: 28,   # bus
    6: 31,   # train
    7: 27,   # truck
}


def load_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            'Missing dependency: ultralytics. Install it in your experiment env with '
            '`pip install ultralytics`.'
        ) from exc
    return YOLO


def find_annotation(acdc_root, condition, split):
    labels_dir = Path(acdc_root) / 'labels' / condition
    candidates = sorted(labels_dir.glob(f'*_{condition}_{split}_*.json'))
    if not candidates:
        raise FileNotFoundError(f'No annotation JSON found in {labels_dir} for split={split}')
    return candidates[0]


def resolve_image_path(images_root, image_info):
    rel = Path(image_info['file_name'])
    path = Path(images_root) / rel
    if path.exists():
        return path

    # Allows evaluating dehazed outputs that preserve paths under condition/split/sequence.
    condition_split_rel = Path(*rel.parts[:])
    path = Path(images_root) / condition_split_rel
    if path.exists():
        return path

    raise FileNotFoundError(f'Image not found for {image_info["file_name"]} under {images_root}')


def write_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_coco_eval(annotation_json, detections_json, output_dir):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print('pycocotools is not installed; skipping mAP computation.')
        return {}

    coco_gt = COCO(str(annotation_json))
    if Path(detections_json).stat().st_size == 0:
        return {'mAP_50_95': 0.0, 'mAP_50': 0.0}

    detections = json.load(open(detections_json))
    if not detections:
        return {'mAP_50_95': 0.0, 'mAP_50': 0.0}

    coco_dt = coco_gt.loadRes(str(detections_json))
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        'mAP_50_95': float(coco_eval.stats[0]),
        'mAP_50': float(coco_eval.stats[1]),
        'mAP_75': float(coco_eval.stats[2]),
        'mAP_small': float(coco_eval.stats[3]),
        'mAP_medium': float(coco_eval.stats[4]),
        'mAP_large': float(coco_eval.stats[5]),
    }
    with open(Path(output_dir) / 'coco_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--acdc-root', default='./data/acdc')
    parser.add_argument('--images-root', default='./data/acdc/images')
    parser.add_argument('--condition', default='fog', choices=['fog', 'night', 'rain', 'snow'])
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--model', default='yolo11n.pt')
    parser.add_argument('--output-dir', default='./output/acdc_yolo11_eval')
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    YOLO = load_ultralytics()
    model = YOLO(args.model)

    annotation_json = find_annotation(args.acdc_root, args.condition, args.split)
    coco = json.load(open(annotation_json))
    images = coco['images'][:args.limit] if args.limit is not None else coco['images']

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detections_path = output_dir / 'detections_coco.json'

    detections = []
    for image_info in tqdm(images, desc=f'YOLO11 {args.condition}/{args.split}'):
        image_path = resolve_image_path(args.images_root, image_info)
        result = model.predict(
            source=str(image_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )[0]

        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, cls_id in zip(boxes, scores, classes):
            if cls_id not in COCO_TO_ACDC:
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                'image_id': image_info['id'],
                'category_id': COCO_TO_ACDC[cls_id],
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': float(score),
            })

    with detections_path.open('w') as f:
        json.dump(detections, f)

    summary = {
        'condition': args.condition,
        'split': args.split,
        'model': args.model,
        'images_root': args.images_root,
        'num_images': len(images),
        'num_detections': len(detections),
        'detections_json': str(detections_path),
    }
    if args.split != 'test':
        summary.update(run_coco_eval(annotation_json, detections_path, output_dir))

    with (output_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=2)
    write_csv(output_dir / 'summary.csv', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
