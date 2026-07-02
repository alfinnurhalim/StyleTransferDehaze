import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd


MODEL_ORDER = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']
METRICS = ['box_map', 'box_map50', 'box_map75', 'box_mp', 'box_mr']


def model_rank(model):
    try:
        return MODEL_ORDER.index(model)
    except ValueError:
        return len(MODEL_ORDER)


def parse_summary_path(summary_path, root):
    rel = summary_path.relative_to(root)
    parts = rel.parts
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2]
    if len(parts) >= 3:
        return 'unknown_prefix', parts[0], parts[1]
    return 'unknown_prefix', 'unknown_model', summary_path.parent.name


def collect_rows(root):
    root = Path(root)
    rows = []
    for summary_path in sorted(root.glob('**/summary.json')):
        with summary_path.open() as f:
            row = json.load(f)
        prefix, model_dir, variant = parse_summary_path(summary_path, root)
        row['prefix'] = prefix
        row['model_dir'] = model_dir
        row['variant'] = variant
        row['summary_path'] = str(summary_path)
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f'No summary.json files found under {root}')

    df = pd.DataFrame(rows)
    df['model_rank'] = df['model'].map(model_rank)
    return df.sort_values(['prefix', 'model_rank', 'variant']).reset_index(drop=True)


def make_pivot(df, metric):
    pivot = df.pivot_table(
        index=['prefix', 'model'],
        columns='variant',
        values=metric,
        aggfunc='first',
    ).reset_index()
    if 'dehazed' in pivot.columns and 'original' in pivot.columns:
        pivot[f'{metric}_delta'] = pivot['dehazed'] - pivot['original']
        pivot[f'{metric}_delta_pct'] = 100 * pivot[f'{metric}_delta'] / pivot['original'].replace(0, pd.NA)
    pivot['model_rank'] = pivot['model'].map(model_rank)
    return pivot.sort_values(['prefix', 'model_rank']).reset_index(drop=True)


def save_metric_bars(df, metric, out_dir):
    for prefix, group in df.groupby('prefix'):
        pivot = group.pivot(index='model', columns='variant', values=metric).reindex(MODEL_ORDER)
        pivot = pivot.dropna(how='all')
        if pivot.empty:
            continue

        ax = pivot.plot(kind='bar', figsize=(9, 5), width=0.78)
        ax.set_title(f'{metric} on XWOD {prefix}')
        ax.set_xlabel('YOLO11 model')
        ax.set_ylabel(metric)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(title='Input')
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f'{prefix}_{metric}_bar.png', dpi=180)
        plt.close()


def save_delta_plot(pivot, metric, out_dir):
    delta_col = f'{metric}_delta'
    if delta_col not in pivot.columns:
        return

    for prefix, group in pivot.groupby('prefix'):
        group = group.sort_values('model_rank')
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.axhline(0, color='black', linewidth=1)
        ax.bar(group['model'], group[delta_col])
        ax.set_title(f'Dehazed - original delta for {metric} on XWOD {prefix}')
        ax.set_xlabel('YOLO11 model')
        ax.set_ylabel(f'Delta {metric}')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f'{prefix}_{metric}_delta.png', dpi=180)
        plt.close()


def save_delta_heatmap(pivot, metric, out_dir):
    delta_col = f'{metric}_delta'
    if delta_col not in pivot.columns:
        return

    for prefix, group in pivot.groupby('prefix'):
        table = group.pivot_table(index='model', values=delta_col, aggfunc='first').reindex(MODEL_ORDER)
        table = table.dropna(how='all')
        if table.empty:
            continue

        values = table.values
        finite = pd.Series(values.flatten()).dropna()
        if finite.empty:
            continue
        vmax = max(abs(float(finite.max())), abs(float(finite.min())))
        vmax = vmax if vmax > 0 else 1.0

        fig, ax = plt.subplots(figsize=(4.5, 5.2))
        im = ax.imshow(values, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(f'Delta heatmap: {metric}\nXWOD {prefix}')
        ax.set_xticks([0])
        ax.set_xticklabels(['dehazed - original'])
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)
        for i, val in enumerate(values[:, 0]):
            if pd.notna(val):
                ax.text(0, i, f'{val:.4f}', ha='center', va='center', color='black')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f'{prefix}_{metric}_delta_heatmap.png', dpi=180)
        plt.close()


def save_metric_overview(pivots, out_dir):
    rows = []
    for metric, pivot in pivots.items():
        delta_col = f'{metric}_delta'
        if delta_col not in pivot.columns:
            continue
        for _, row in pivot.iterrows():
            rows.append({
                'prefix': row['prefix'],
                'model': row['model'],
                'metric': metric,
                'delta': row[delta_col],
            })
    overview = pd.DataFrame(rows)
    if overview.empty:
        return

    for prefix, group in overview.groupby('prefix'):
        table = group.pivot_table(index='model', columns='metric', values='delta', aggfunc='first').reindex(MODEL_ORDER)
        table = table.dropna(how='all')
        if table.empty:
            continue

        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        values = table.values
        finite = pd.Series(values.flatten()).dropna()
        vmax = max(abs(float(finite.max())), abs(float(finite.min())))
        vmax = vmax if vmax > 0 else 1.0
        im = ax.imshow(values, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(f'All metric deltas on XWOD {prefix}')
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(table.columns, rotation=30, ha='right')
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                if pd.notna(val):
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f'{prefix}_all_metric_deltas_heatmap.png', dpi=180)
        plt.close()


def save_markdown_report(df, primary_pivot, out_dir, primary_metric):
    lines = ['# XWOD YOLO11 Downstream Summary', '']
    for prefix, group in primary_pivot.groupby('prefix'):
        lines.append(f'## {prefix}')
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
    parser.add_argument('--results-root', default='./output/xwod_yolo11_comparison')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--primary-metric', default='box_map', choices=METRICS)
    args = parser.parse_args()

    out_dir = Path(args.output_dir or Path(args.results_root) / 'summary_plots')
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_rows(args.results_root)
    df.to_csv(out_dir / 'all_results_long.csv', index=False)

    pivots = {}
    for metric in METRICS:
        if metric not in df.columns:
            continue
        pivot = make_pivot(df, metric)
        pivot.to_csv(out_dir / f'{metric}_original_vs_dehazed.csv', index=False)
        pivots[metric] = pivot
        save_metric_bars(df, metric, out_dir)
        save_delta_plot(pivot, metric, out_dir)
        save_delta_heatmap(pivot, metric, out_dir)

    save_metric_overview(pivots, out_dir)
    save_markdown_report(df, pivots[args.primary_metric], out_dir, args.primary_metric)

    print(f'Saved summary CSVs and plots to {out_dir}')


if __name__ == '__main__':
    main()
