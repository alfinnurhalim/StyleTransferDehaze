import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw


METRICS = ['mAP_50_95', 'mAP_50', 'mAP_75', 'mAP_small', 'mAP_medium', 'mAP_large']
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
COLORS = {
    'original': '#5B8DEF',
    'dehazed': '#F28E2B',
    'delta_pos': '#2CA02C',
    'delta_neg': '#D62728',
}


def load_json(path):
    with Path(path).open('r') as f:
        return json.load(f)


def discover_summaries(results_root):
    rows = []
    for path in sorted(Path(results_root).glob('*/*/summary.json')):
        variant = path.parent.name
        model_dir = path.parent.parent.name
        if variant not in {'original', 'dehazed'}:
            continue
        summary = load_json(path)
        row = {
            'model_dir': model_dir,
            'model': Path(str(summary.get('model', model_dir))).stem,
            'variant': variant,
            'summary_path': str(path),
            'eval_dir': str(path.parent),
        }
        row.update(summary)
        rows.append(row)
    return pd.DataFrame(rows)


def metric_columns(df):
    return [metric for metric in METRICS if metric in df.columns and df[metric].notna().any()]


def make_delta_table(df, metrics):
    if df.empty:
        return pd.DataFrame()
    rows = []
    for model_dir, group in df.groupby('model_dir'):
        by_variant = {row.variant: row for row in group.itertuples()}
        if 'original' not in by_variant or 'dehazed' not in by_variant:
            continue
        original = by_variant['original']
        dehazed = by_variant['dehazed']
        row = {
            'model_dir': model_dir,
            'model': dehazed.model,
            'num_images': getattr(dehazed, 'num_images', None),
            'num_annotations': getattr(dehazed, 'num_annotations', None),
            'detections_original': getattr(original, 'num_detections', None),
            'detections_dehazed': getattr(dehazed, 'num_detections', None),
            'detections_delta': getattr(dehazed, 'num_detections', 0) - getattr(original, 'num_detections', 0),
        }
        for metric in metrics:
            before = getattr(original, metric, None)
            after = getattr(dehazed, metric, None)
            if before is not None and after is not None:
                row[f'{metric}_original'] = before
                row[f'{metric}_dehazed'] = after
                row[f'{metric}_delta'] = after - before
        rows.append(row)
    return pd.DataFrame(rows).sort_values('model_dir')


def save_bar_plot(df, metric, out_path):
    pivot = df.pivot(index='model_dir', columns='variant', values=metric).sort_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    pivot[['original', 'dehazed']].plot(
        kind='bar',
        ax=ax,
        color=[COLORS['original'], COLORS['dehazed']],
        width=0.72,
    )
    ax.set_xlabel('YOLO model')
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_title(f'{metric.replace("_", " ")}: original vs dehazed')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title='')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_delta_plot(delta, metric, out_path):
    col = f'{metric}_delta'
    plot_df = delta[['model_dir', col]].dropna().sort_values('model_dir')
    colors = [COLORS['delta_pos'] if value >= 0 else COLORS['delta_neg'] for value in plot_df[col]]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(plot_df['model_dir'], plot_df[col], color=colors)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('YOLO model')
    ax.set_ylabel(f'Delta {metric.replace("_", " ")}')
    ax.set_title(f'Dehazed - original {metric.replace("_", " ")}')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_size_ap_plot(df, out_path):
    size_metrics = [m for m in ['mAP_small', 'mAP_medium', 'mAP_large'] if m in df.columns and df[m].notna().any()]
    if not size_metrics:
        return False
    rows = []
    for row in df.itertuples():
        for metric in size_metrics:
            rows.append({
                'model_dir': row.model_dir,
                'variant': row.variant,
                'metric': metric.replace('mAP_', ''),
                'value': getattr(row, metric),
            })
    plot_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(size_metrics), figsize=(4.4 * len(size_metrics), 4.4), sharey=True)
    if len(size_metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, [m.replace('mAP_', '') for m in size_metrics]):
        pivot = plot_df[plot_df['metric'] == metric].pivot(index='model_dir', columns='variant', values='value')
        pivot[['original', 'dehazed']].plot(
            kind='bar',
            ax=ax,
            color=[COLORS['original'], COLORS['dehazed']],
            width=0.72,
            legend=False,
        )
        ax.set_title(metric.title())
        ax.set_xlabel('YOLO model')
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('AP')
    axes[-1].legend(['Original', 'Dehazed'], title='')
    fig.suptitle('AP by object size')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def save_detection_count_plot(df, out_path):
    if 'num_detections' not in df.columns:
        return False
    pivot = df.pivot(index='model_dir', columns='variant', values='num_detections').sort_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    pivot[['original', 'dehazed']].plot(
        kind='bar',
        ax=ax,
        color=[COLORS['original'], COLORS['dehazed']],
        width=0.72,
    )
    ax.set_xlabel('YOLO model')
    ax.set_ylabel('Detections')
    ax.set_title('Number of YOLO detections')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title='')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def save_delta_heatmap(delta, metrics, out_path):
    delta_cols = [f'{metric}_delta' for metric in metrics if f'{metric}_delta' in delta.columns]
    if not delta_cols:
        return False
    heat = delta.set_index('model_dir')[delta_cols].rename(columns=lambda c: c.replace('_delta', ''))
    fig, ax = plt.subplots(figsize=(1.7 * len(delta_cols) + 2, 1.0 * len(heat) + 2))
    im = ax.imshow(heat.values, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels([c.replace('_', ' ') for c in heat.columns], rotation=35, ha='right')
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f'{heat.iloc[i, j]:+.4f}', ha='center', va='center', fontsize=9)
    ax.set_title('Dehazed - original metric deltas')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def resolve_image(root, file_name):
    root = Path(root)
    candidate = root / file_name
    if candidate.exists():
        return candidate
    stem = Path(file_name).stem
    for ext in IMG_EXTS:
        candidate = root / f'{stem}{ext}'
        if candidate.exists():
            return candidate
    return None


def group_detections(detections, conf):
    grouped = defaultdict(list)
    for det in detections:
        if det.get('score', 0.0) >= conf:
            grouped[det['image_id']].append(det)
    return grouped


def draw_boxes(image_path, detections, categories, label, max_boxes=25):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    width = max(2, image.width // 320)
    detections = sorted(detections, key=lambda d: d.get('score', 0.0), reverse=True)[:max_boxes]
    for det in detections:
        x, y, w, h = det['bbox']
        x2, y2 = x + w, y + h
        cls = categories.get(det['category_id'], str(det['category_id']))
        text = f'{cls} {det.get("score", 0.0):.2f}'
        draw.rectangle([x, y, x2, y2], outline=(255, 140, 0), width=width)
        text_w, text_h = draw.textsize(text)
        draw.rectangle([x, y, x + text_w + 4, y + text_h + 4], fill=(255, 140, 0))
        draw.text((x + 2, y + 2), text, fill=(0, 0, 0))
    header = 34
    canvas = Image.new('RGB', (image.width, image.height + header), 'white')
    canvas.paste(image, (0, header))
    ImageDraw.Draw(canvas).text((8, 9), label, fill=(0, 0, 0))
    return canvas


def resize_height(image, height):
    scale = height / image.height
    return image.resize((max(1, int(image.width * scale)), height), Image.BICUBIC)


def make_detection_examples(df, out_dir, max_examples, vis_conf):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    complete_models = []
    for model_dir, group in df.groupby('model_dir'):
        variants = set(group['variant'])
        if {'original', 'dehazed'} <= variants:
            complete_models.append(model_dir)
    if not complete_models:
        return []
    model_dir = sorted(complete_models)[0]
    original_row = df[(df['model_dir'] == model_dir) & (df['variant'] == 'original')].iloc[0]
    dehazed_row = df[(df['model_dir'] == model_dir) & (df['variant'] == 'dehazed')].iloc[0]
    gt = load_json(Path(original_row['eval_dir']) / 'rtts_test_coco_gt.json')
    categories = {cat['id']: cat['name'] for cat in gt.get('categories', [])}
    original_dets = group_detections(load_json(original_row['detections_json']), vis_conf)
    dehazed_dets = group_detections(load_json(dehazed_row['detections_json']), vis_conf)
    exported = []
    for image_info in gt.get('images', [])[: max_examples * 10]:
        image_id = image_info['id']
        if not original_dets.get(image_id) and not dehazed_dets.get(image_id):
            continue
        original_path = resolve_image(original_row['images_root'], image_info['file_name'])
        dehazed_path = resolve_image(dehazed_row['images_root'], image_info['file_name'])
        if original_path is None or dehazed_path is None:
            continue
        left = draw_boxes(original_path, original_dets.get(image_id, []), categories, f'{model_dir} original')
        right = draw_boxes(dehazed_path, dehazed_dets.get(image_id, []), categories, f'{model_dir} dehazed')
        height = 330
        left = resize_height(left, height)
        right = resize_height(right, height)
        gap = 10
        canvas = Image.new('RGB', (left.width + right.width + gap, height), 'white')
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width + gap, 0))
        path = out_dir / f'{image_info.get("stem", Path(image_info["file_name"]).stem)}_{model_dir}_detections.jpg'
        canvas.save(path, quality=92)
        exported.append(path)
        if len(exported) >= max_examples:
            break
    return exported


def write_markdown(path, df, delta, metrics, plots, examples, title):
    lines = [f'# {title}', '']
    if df.empty:
        lines.append('No YOLO comparison summaries were found.')
        Path(path).write_text('\n'.join(lines))
        return

    num_images = int(df['num_images'].dropna().iloc[0]) if 'num_images' in df and df['num_images'].notna().any() else 'n/a'
    num_annotations = int(df['num_annotations'].dropna().iloc[0]) if 'num_annotations' in df and df['num_annotations'].notna().any() else 'n/a'
    lines.append('## Dataset and Setup')
    lines.append('')
    lines.append(f'- Images evaluated: `{num_images}`.')
    lines.append(f'- Ground-truth boxes: `{num_annotations}`.')
    lines.append(f'- Detector variants: `{", ".join(sorted(df["model_dir"].unique()))}`.')
    lines.append('- Each detector is evaluated on the original hazy images and on the dehazed outputs from the same image list.')
    lines.append('')

    if metrics:
        lines.append('## Main Quantitative Results')
        lines.append('')
        display_cols = ['model_dir']
        for metric in ['mAP_50_95', 'mAP_50']:
            for suffix in ['original', 'dehazed', 'delta']:
                col = f'{metric}_{suffix}'
                if col in delta.columns:
                    display_cols.append(col)
        lines.append(delta[display_cols].to_markdown(index=False, floatfmt='.4f'))
        lines.append('')
    else:
        lines.append('## Main Quantitative Results')
        lines.append('')
        lines.append('COCO mAP metrics are not present in the saved summaries, so this report is limited to detector counts.')
        lines.append('')

    if 'detections_delta' in delta.columns:
        lines.append('## Detection Count')
        lines.append('')
        lines.append(delta[['model_dir', 'detections_original', 'detections_dehazed', 'detections_delta']].to_markdown(index=False))
        lines.append('')

    lines.append('## Interpretation')
    lines.append('')
    if metrics and 'mAP_50_95_delta' in delta.columns:
        mean_delta = float(delta['mAP_50_95_delta'].mean())
        best = delta.loc[delta['mAP_50_95_delta'].idxmax()]
        worst = delta.loc[delta['mAP_50_95_delta'].idxmin()]
        direction = 'improved' if mean_delta > 0 else 'reduced'
        if best['mAP_50_95_delta'] > 0:
            contrast = (
                f'The largest gain is `{best["mAP_50_95_delta"]:+.4f}` on `{best["model_dir"]}`, '
                f'while the largest drop is `{worst["mAP_50_95_delta"]:+.4f}` on `{worst["model_dir"]}`.'
            )
        else:
            contrast = (
                f'The smallest drop is `{best["mAP_50_95_delta"]:+.4f}` on `{best["model_dir"]}`, '
                f'while the largest drop is `{worst["mAP_50_95_delta"]:+.4f}` on `{worst["model_dir"]}`.'
            )
        lines.append(
            f'Across the tested YOLO models, dehazing {direction} mean mAP@50:95 by `{mean_delta:+.4f}`. '
            f'{contrast}'
        )
        if mean_delta < 0:
            lines.append(
                'For this RTTS run, the dehazed images do not improve downstream detection mAP overall. '
                'That is still useful thesis evidence: it suggests the current dehazing checkpoint may preserve perceptual quality '
                'but can shift textures or contrast in ways that pretrained detectors do not benefit from.'
            )
        else:
            lines.append(
                'For this RTTS run, the downstream detector benefits from the dehazed inputs overall, supporting the claim that '
                'the dehazing model improves task-relevant visibility.'
            )
    else:
        lines.append(
            'Use this report as a sanity check of detector behavior. For thesis-level claims, rerun the evaluator with '
            '`pycocotools` installed so mAP metrics are written into each summary.'
        )
    lines.append('')

    if plots:
        lines.append('## Plots')
        lines.append('')
        for plot in plots:
            rel = Path(plot).relative_to(Path(path).parent)
            lines.append(f'![{Path(plot).stem}]({rel.as_posix()})')
            lines.append('')

    if examples:
        lines.append('## Qualitative Detection Examples')
        lines.append('')
        for example in examples:
            rel = Path(example).relative_to(Path(path).parent)
            lines.append(f'![{Path(example).stem}]({rel.as_posix()})')
            lines.append('')

    lines.append('## Reuse')
    lines.append('')
    lines.append('Regenerate this report with:')
    lines.append('')
    lines.append('```bash')
    lines.append(f'python experiments/report_tools/summarize_yolo_comparison.py --results-root {Path(df.iloc[0]["summary_path"]).parents[2]} --output-dir {Path(path).parent}')
    lines.append('```')
    lines.append('')
    Path(path).write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', default='./output/rtts_yolov8_comparison')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--title', default='RTTS YOLOv8 Downstream Evaluation')
    parser.add_argument('--max-examples', type=int, default=8)
    parser.add_argument('--vis-conf', type=float, default=0.25)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else results_root / 'thesis_summary'
    plots_dir = output_dir / 'plots'
    examples_dir = output_dir / 'qualitative_detections'
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = discover_summaries(results_root)
    metrics = metric_columns(df)
    delta = make_delta_table(df, metrics)
    df.to_csv(output_dir / 'all_results_long.csv', index=False)
    delta.to_csv(output_dir / 'original_vs_dehazed.csv', index=False)

    plots = []
    for metric in metrics:
        path = plots_dir / f'{metric}_original_vs_dehazed.png'
        save_bar_plot(df, metric, path)
        plots.append(path)
        delta_path = plots_dir / f'{metric}_delta.png'
        save_delta_plot(delta, metric, delta_path)
        plots.append(delta_path)
    if save_size_ap_plot(df, plots_dir / 'object_size_ap.png'):
        plots.append(plots_dir / 'object_size_ap.png')
    if save_detection_count_plot(df, plots_dir / 'detection_counts.png'):
        plots.append(plots_dir / 'detection_counts.png')
    if save_delta_heatmap(delta, metrics, plots_dir / 'metric_delta_heatmap.png'):
        plots.append(plots_dir / 'metric_delta_heatmap.png')

    examples = []
    if args.max_examples > 0:
        examples = make_detection_examples(df, examples_dir, args.max_examples, args.vis_conf)

    summary = {
        'results_root': str(results_root),
        'output_dir': str(output_dir),
        'num_rows': int(len(df)),
        'models': sorted(df['model_dir'].unique().tolist()) if not df.empty else [],
        'metrics': metrics,
        'plots': [str(p) for p in plots],
        'examples': [str(p) for p in examples],
    }
    with (output_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=2)
    write_markdown(output_dir / 'README.md', df, delta, metrics, plots, examples, args.title)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
