import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd


MODEL_ORDER = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']
METRICS = ['mAP_50_95', 'mAP_50', 'mAP_75', 'mAP_small', 'mAP_medium', 'mAP_large']


def model_rank(model):
    try:
        return MODEL_ORDER.index(model)
    except ValueError:
        return len(MODEL_ORDER)


def collect_rows(root):
    rows = []
    for summary_path in Path(root).glob('*/**/summary.json'):
        with summary_path.open() as f:
            row = json.load(f)
        parts = summary_path.parts
        row['variant'] = summary_path.parent.name
        row['summary_path'] = str(summary_path)
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f'No summary.json files found under {root}')

    df = pd.DataFrame(rows)
    df['model_rank'] = df['model'].map(model_rank)
    df = df.sort_values(['condition', 'split', 'model_rank', 'variant']).reset_index(drop=True)
    return df


def make_pivot(df, metric):
    pivot = df.pivot_table(
        index=['condition', 'split', 'model'],
        columns='variant',
        values=metric,
        aggfunc='first',
    ).reset_index()
    if 'dehazed' in pivot.columns and 'original' in pivot.columns:
        pivot[f'{metric}_delta'] = pivot['dehazed'] - pivot['original']
        pivot[f'{metric}_delta_pct'] = 100 * pivot[f'{metric}_delta'] / pivot['original'].replace(0, pd.NA)
    pivot['model_rank'] = pivot['model'].map(model_rank)
    return pivot.sort_values(['condition', 'split', 'model_rank']).reset_index(drop=True)


def save_metric_bars(df, metric, out_dir):
    for (condition, split), group in df.groupby(['condition', 'split']):
        pivot = group.pivot(index='model', columns='variant', values=metric).reindex(MODEL_ORDER)
        pivot = pivot.dropna(how='all')
        if pivot.empty:
            continue

        ax = pivot.plot(kind='bar', figsize=(9, 5), width=0.78)
        ax.set_title(f'{metric} on ACDC {condition}/{split}')
        ax.set_xlabel('YOLO11 model')
        ax.set_ylabel(metric)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(title='Input')
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f'{condition}_{split}_{metric}_bar.png', dpi=180)
        plt.close()


def save_delta_plot(pivots, metric, out_dir):
    for (condition, split), group in pivots.groupby(['condition', 'split']):
        if f'{metric}_delta' not in group:
            continue
        group = group.sort_values('model_rank')
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.axhline(0, color='black', linewidth=1)
        ax.bar(group['model'], group[f'{metric}_delta'])
        ax.set_title(f'Dehazed - original delta for {metric} on ACDC {condition}/{split}')
        ax.set_xlabel('YOLO11 model')
        ax.set_ylabel(f'Delta {metric}')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f'{condition}_{split}_{metric}_delta.png', dpi=180)
        plt.close()


def save_detection_count_plot(df, out_dir):
    for (condition, split), group in df.groupby(['condition', 'split']):
        pivot = group.pivot(index='model', columns='variant', values='num_detections').reindex(MODEL_ORDER)
        pivot = pivot.dropna(how='all')
        if pivot.empty:
            continue

        ax = pivot.plot(kind='bar', figsize=(9, 5), width=0.78)
        ax.set_title(f'Detection count on ACDC {condition}/{split}')
        ax.set_xlabel('YOLO11 model')
        ax.set_ylabel('Number of detections')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(title='Input')
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f'{condition}_{split}_detections_bar.png', dpi=180)
        plt.close()


def save_metric_heatmap(pivots, metric, out_dir):
    delta_col = f'{metric}_delta'
    if delta_col not in pivots.columns:
        return

    for (condition, split), group in pivots.groupby(['condition', 'split']):
        table = group.pivot_table(index='model', values=delta_col, aggfunc='first').reindex(MODEL_ORDER)
        table = table.dropna(how='all')
        if table.empty:
            continue

        fig, ax = plt.subplots(figsize=(4.5, 5.2))
        values = table.values
        vmax = max(abs(float(pd.Series(values.flatten()).dropna().max())),
                   abs(float(pd.Series(values.flatten()).dropna().min())))
        im = ax.imshow(values, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(f'Delta heatmap: {metric}\nACDC {condition}/{split}')
        ax.set_xticks([0])
        ax.set_xticklabels(['dehazed - original'])
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)
        for i, val in enumerate(values[:, 0]):
            ax.text(0, i, f'{val:.3f}', ha='center', va='center', color='black')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f'{condition}_{split}_{metric}_delta_heatmap.png', dpi=180)
        plt.close()


def save_markdown_report(df, pivots, out_dir, primary_metric):
    lines = ['# ACDC YOLO11 Downstream Summary', '']
    for (condition, split), group in pivots.groupby(['condition', 'split']):
        lines.append(f'## {condition}/{split}')
        lines.append('')
        cols = ['model', 'original', 'dehazed', f'{primary_metric}_delta', f'{primary_metric}_delta_pct']
        available = [c for c in cols if c in group.columns]
        lines.append(group[available].to_markdown(index=False, floatfmt='.4f'))
        lines.append('')

    lines.append('## Generated Figures')
    lines.append('')
    for png in sorted(Path(out_dir).glob('*.png')):
        lines.append(f'- `{png.name}`')
    lines.append('')

    with (Path(out_dir) / 'report.md').open('w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', default='./output/acdc_yolo11_comparison')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--primary-metric', default='mAP_50_95', choices=METRICS)
    args = parser.parse_args()

    out_dir = Path(args.output_dir or Path(args.results_root) / 'summary_plots')
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_rows(args.results_root)
    df.to_csv(out_dir / 'all_results_long.csv', index=False)

    pivot_frames = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        pivot = make_pivot(df, metric)
        pivot.to_csv(out_dir / f'{metric}_original_vs_dehazed.csv', index=False)
        pivot_frames.append(pivot.assign(metric=metric))
        save_metric_bars(df, metric, out_dir)
        save_delta_plot(pivot, metric, out_dir)
        save_metric_heatmap(pivot, metric, out_dir)

    save_detection_count_plot(df, out_dir)

    primary_pivot = make_pivot(df, args.primary_metric)
    save_markdown_report(df, primary_pivot, out_dir, args.primary_metric)

    print(f'Saved summary CSVs and plots to {out_dir}')


if __name__ == '__main__':
    main()
