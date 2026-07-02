import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm


RTTS_CLASSES = ['person', 'bicycle', 'car', 'motorbike', 'bus']
RTTS_CLASS_TO_ID = {name: idx + 1 for idx, name in enumerate(RTTS_CLASSES)}
COCO_TO_RTTS = {
    0: RTTS_CLASS_TO_ID['person'],
    1: RTTS_CLASS_TO_ID['bicycle'],
    2: RTTS_CLASS_TO_ID['car'],
    3: RTTS_CLASS_TO_ID['motorbike'],
    5: RTTS_CLASS_TO_ID['bus'],
}
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def load_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit('Missing dependency: ultralytics. Install with `pip install ultralytics`.') from exc
    return YOLO


def read_image_set(rtts_root, split, limit=None):
    split_path = Path(rtts_root) / 'ImageSets' / 'Main' / f'{split}.txt'
    stems = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    if limit is not None:
        stems = stems[:limit]
    return stems


def resolve_image(images_root, stem):
    for ext in IMG_EXTS:
        path = Path(images_root) / f'{stem}{ext}'
        if path.exists():
            return path
    raise FileNotFoundError(f'Could not find image for stem {stem} under {images_root}')


def voc_to_coco(rtts_root, split, output_json, limit=None):
    stems = read_image_set(rtts_root, split, limit)
    images = []
    annotations = []
    ann_id = 1
    for image_id, stem in enumerate(stems, start=1):
        xml_path = Path(rtts_root) / 'Annotations' / f'{stem}.xml'
        root = ET.parse(xml_path).getroot()
        filename = root.findtext('filename')
        size = root.find('size')
        width = int(float(size.findtext('width')))
        height = int(float(size.findtext('height')))
        images.append({
            'id': image_id,
            'file_name': filename,
            'width': width,
            'height': height,
            'stem': stem,
        })
        for obj in root.findall('object'):
            name = obj.findtext('name')
            if name not in RTTS_CLASS_TO_ID:
                continue
            box = obj.find('bndbox')
            xmin = max(0.0, float(box.findtext('xmin')))
            ymin = max(0.0, float(box.findtext('ymin')))
            xmax = min(float(width), float(box.findtext('xmax')))
            ymax = min(float(height), float(box.findtext('ymax')))
            bw = max(0.0, xmax - xmin)
            bh = max(0.0, ymax - ymin)
            if bw <= 0 or bh <= 0:
                continue
            annotations.append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': RTTS_CLASS_TO_ID[name],
                'bbox': [xmin, ymin, bw, bh],
                'area': bw * bh,
                'iscrowd': 0,
            })
            ann_id += 1

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': idx + 1, 'name': name} for idx, name in enumerate(RTTS_CLASSES)],
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open('w') as f:
        json.dump(coco, f)
    return coco


def run_coco_eval(annotation_json, detections_json, output_dir):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print('pycocotools is not installed; skipping mAP computation.')
        return {}

    detections = json.load(open(detections_json))
    if not detections:
        return {'mAP_50_95': 0.0, 'mAP_50': 0.0}

    coco_gt = COCO(str(annotation_json))
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


def write_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtts-root', default='./data/RTTS')
    parser.add_argument('--images-root', default='./data/RTTS/JPEGImages')
    parser.add_argument('--split', default='test')
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--output-dir', default='./output/rtts_yolov8_eval')
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default=None, help='Ultralytics device, e.g. 0, cpu, cuda:0.')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    YOLO = load_ultralytics()
    model = YOLO(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_json = output_dir / f'rtts_{args.split}_coco_gt.json'
    coco = voc_to_coco(args.rtts_root, args.split, annotation_json, args.limit)
    detections_path = output_dir / 'detections_coco.json'

    detections = []
    for image_info in tqdm(coco['images'], desc=f'YOLOv8 RTTS {args.split}'):
        stem = image_info['stem']
        image_path = resolve_image(args.images_root, stem)
        result = model.predict(
            source=str(image_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, cls_id in zip(boxes, scores, classes):
            if cls_id not in COCO_TO_RTTS:
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                'image_id': image_info['id'],
                'category_id': COCO_TO_RTTS[cls_id],
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': float(score),
            })

    with detections_path.open('w') as f:
        json.dump(detections, f)

    summary = {
        'split': args.split,
        'model': args.model,
        'images_root': str(args.images_root),
        'num_images': len(coco['images']),
        'num_annotations': len(coco['annotations']),
        'num_detections': len(detections),
        'detections_json': str(detections_path),
    }
    summary.update(run_coco_eval(annotation_json, detections_path, output_dir))
    with (output_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=2)
    write_csv(output_dir / 'summary.csv', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
