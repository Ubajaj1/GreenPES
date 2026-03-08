"""
Saturation experiment analysis.

Reads results/saturation_results.json and produces:
  - results/figures/fig_sat1_scaling_curves.png   — quality vs. tokens per task (4 subplots)
  - results/figures/fig_sat2_saturation_points.png — saturation token count heatmap (model × task)
  - results/saturation_summary.csv                 — per (model, task) fit params + saturation points

Usage:
    python experiments/saturation_analysis.py
    python experiments/saturation_analysis.py --input results/saturation_results.json
    python experiments/saturation_analysis.py --output-dir results/
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr  # noqa: F401 — used in print_correlation_stats

# ── Constants ─────────────────────────────────────────────────────────────────

TASKS = ['qa', 'summarization', 'classification', 'instruction_following']
TASK_LABELS = {
    'qa': 'QA',
    'summarization': 'Summarization',
    'classification': 'Classification',
    'instruction_following': 'Instruction Following',
}

MODEL_ORDER = [
    'llama-3.1-8b',
    'gemini-flash',
    'qwen3-32b',
    'llama-3.3-70b',
    'kimi-k2',
    'gpt-4o-mini',
    'claude-haiku',
]

MODEL_COLORS = {
    'llama-3.1-8b':  '#e41a1c',
    'gemini-flash':  '#ff7f00',
    'qwen3-32b':     '#4daf4a',
    'llama-3.3-70b': '#984ea3',
    'kimi-k2':       '#a65628',
    'gpt-4o-mini':   '#377eb8',
    'claude-haiku':  '#f781bf',
}

# Saturation defined as the token count where quality ≥ 95% of asymptote
SATURATION_THRESHOLD = 0.95

# ── Curve models ──────────────────────────────────────────────────────────────

def log_curve(x, a, b, c):
    """Quality = a * log(b * x) + c  (logarithmic growth)"""
    return a * np.log(np.maximum(b * x, 1e-6)) + c


def sigmoid_curve(x, L, k, x0, c):
    """Quality = L / (1 + exp(-k*(x - x0))) + c  (sigmoid growth)"""
    return L / (1 + np.exp(-k * (x - x0))) + c


def fit_best_curve(tokens: np.ndarray, quality: np.ndarray) -> dict:
    """
    Fit both logarithmic and sigmoid curves; return the better fit (lower RMSE).
    Returns a dict with: model_type, params, rmse, r2, saturation_tokens.
    """
    results = {}

    # Logarithmic fit
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p0_log = [0.1, 0.01, quality.mean()]
            popt_log, _ = curve_fit(log_curve, tokens, quality, p0=p0_log,
                                    maxfev=5000, bounds=([-1, 1e-6, -1], [2, 1, 2]))
        pred_log = log_curve(tokens, *popt_log)
        rmse_log = float(np.sqrt(np.mean((quality - pred_log) ** 2)))
        ss_res = np.sum((quality - pred_log) ** 2)
        ss_tot = np.sum((quality - quality.mean()) ** 2)
        r2_log = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        results['log'] = {'params': popt_log, 'rmse': rmse_log, 'r2': r2_log}
    except Exception:
        results['log'] = None

    # Sigmoid fit
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            L0 = float(quality.max() - quality.min())
            x0_0 = float(np.median(tokens))
            p0_sig = [L0, 0.05, x0_0, float(quality.min())]
            popt_sig, _ = curve_fit(sigmoid_curve, tokens, quality, p0=p0_sig,
                                    maxfev=5000,
                                    bounds=([0, 0, tokens.min(), -0.5],
                                            [2, 1, tokens.max() * 2, 1.5]))
        pred_sig = sigmoid_curve(tokens, *popt_sig)
        rmse_sig = float(np.sqrt(np.mean((quality - pred_sig) ** 2)))
        ss_res = np.sum((quality - pred_sig) ** 2)
        ss_tot = np.sum((quality - quality.mean()) ** 2)
        r2_sig = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        results['sig'] = {'params': popt_sig, 'rmse': rmse_sig, 'r2': r2_sig}
    except Exception:
        results['sig'] = None

    # Pick winner
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        return {'model_type': 'none', 'params': None, 'rmse': np.nan, 'r2': np.nan,
                'saturation_tokens': np.nan}

    best_key = min(valid, key=lambda k: valid[k]['rmse'])
    best = valid[best_key]
    model_type = 'logarithmic' if best_key == 'log' else 'sigmoid'

    # Compute saturation point: smallest x where fitted quality ≥ 95% of asymptote
    x_range = np.linspace(tokens.min(), tokens.max() * 2, 1000)
    if model_type == 'logarithmic':
        y_hat = log_curve(x_range, *best['params'])
    else:
        y_hat = sigmoid_curve(x_range, *best['params'])

    asymptote = float(y_hat.max())
    threshold_q = SATURATION_THRESHOLD * asymptote
    sat_mask = y_hat >= threshold_q
    saturation_tokens = float(x_range[sat_mask][0]) if sat_mask.any() else float(tokens.max())

    return {
        'model_type': model_type,
        'params': best['params'].tolist() if hasattr(best['params'], 'tolist') else list(best['params']),
        'rmse': best['rmse'],
        'r2': best['r2'],
        'saturation_tokens': saturation_tokens,
        'asymptote': asymptote,
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Saturation results not found: {path}")
    with open(p) as f:
        raw = json.load(f)
    good = [r for r in raw if 'error' not in r]
    n_err = len(raw) - len(good)
    if n_err:
        print(f"  Dropped {n_err} error records")
    df = pd.DataFrame(good)
    required = {'model', 'task', 'level', 'prompt_tokens', 'quality'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean quality and tokens per (model, task, level)."""
    agg = (df.groupby(['model', 'task', 'level'])
             .agg(mean_quality=('quality', 'mean'),
                  mean_tokens=('prompt_tokens', 'mean'),
                  n=('quality', 'count'))
             .reset_index())
    return agg


# ── Figure 1: Scaling curves ──────────────────────────────────────────────────

def plot_scaling_curves(agg: pd.DataFrame, fits: dict, out_path: str) -> None:
    """
    4-subplot grid. Each subplot = one task.
    One line per model (aggregated mean across examples); fitted curve overlaid.
    """
    models_present = [m for m in MODEL_ORDER if m in agg['model'].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax_idx, task in enumerate(TASKS):
        ax = axes[ax_idx]
        task_data = agg[agg['task'] == task]

        for model in models_present:
            md = task_data[task_data['model'] == model].sort_values('mean_tokens')
            if md.empty:
                continue
            color = MODEL_COLORS.get(model, '#333333')
            ax.scatter(md['mean_tokens'], md['mean_quality'],
                       color=color, s=30, zorder=3)
            ax.plot(md['mean_tokens'], md['mean_quality'],
                    color=color, linewidth=1.2, alpha=0.7, label=model)

            # Overlay fitted curve if available
            key = (model, task)
            if key in fits and fits[key]['params'] is not None:
                fit = fits[key]
                x_range = np.linspace(md['mean_tokens'].min() * 0.9,
                                      md['mean_tokens'].max() * 1.1, 200)
                if fit['model_type'] == 'logarithmic':
                    y_fit = log_curve(x_range, *fit['params'])
                else:
                    y_fit = sigmoid_curve(x_range, *fit['params'])
                ax.plot(x_range, y_fit, color=color, linewidth=0.8,
                        linestyle='--', alpha=0.5)

                # Mark saturation point
                sat_x = fit['saturation_tokens']
                ax.axvline(sat_x, color=color, linewidth=0.5, linestyle=':', alpha=0.4)

        ax.set_title(TASK_LABELS[task], fontsize=11, fontweight='bold')
        ax.set_xlabel('Prompt Tokens', fontsize=9)
        ax.set_ylabel('Mean Quality', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    # Legend on last subplot or figure
    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    fig.legend(list(seen.values()), list(seen.keys()),
               loc='lower center', ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Quality vs. Prompt Tokens by Task and Model\n(dashed = fitted curve, dotted = saturation point)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=(0, 0.05, 1, 0.96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 2: Saturation points heatmap ──────────────────────────────────────

def plot_saturation_heatmap(fits: dict, models: list[str], out_path: str) -> None:
    """
    Heatmap: rows = models, columns = tasks, values = saturation token count.
    """
    task_labels = [TASK_LABELS[t] for t in TASKS]
    sat_matrix = np.full((len(models), len(TASKS)), np.nan)
    for i, model in enumerate(models):
        for j, task in enumerate(TASKS):
            key = (model, task)
            if key in fits and not np.isnan(fits[key].get('saturation_tokens', np.nan)):
                sat_matrix[i, j] = fits[key]['saturation_tokens']

    fig, ax = plt.subplots(figsize=(9, 5))
    # Mask NaN cells
    masked = np.ma.masked_invalid(sat_matrix)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad('#e0e0e0')
    im = ax.imshow(masked, cmap=cmap, aspect='auto')

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(TASKS)):
            val = sat_matrix[i, j]
            txt = f'{val:.0f}' if not np.isnan(val) else 'N/A'
            color = 'white' if (not np.isnan(val) and val > masked.max() * 0.6) else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                    color=color, fontweight='bold')

    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)
    ax.set_title(f'Saturation Token Count by Model × Task\n'
                 f'(tokens at which quality ≥ {SATURATION_THRESHOLD*100:.0f}% of asymptote)',
                 fontsize=11, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Saturation Tokens', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Stats summary ─────────────────────────────────────────────────────────────

def build_summary(fits: dict, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        for task in TASKS:
            key = (model, task)
            fit = fits.get(key, {})
            rows.append({
                'model': model,
                'task': task,
                'fit_type': fit.get('model_type', 'none'),
                'r2': fit.get('r2', np.nan),
                'rmse': fit.get('rmse', np.nan),
                'saturation_tokens': fit.get('saturation_tokens', np.nan),
                'asymptote': fit.get('asymptote', np.nan),
            })
    return pd.DataFrame(rows)


# ── Correlation: level vs quality (by task) ───────────────────────────────────

def print_correlation_stats(df: pd.DataFrame) -> None:
    print("\n── Pearson r (prompt_tokens vs quality) by task ──")
    for task in TASKS:
        td = df[df['task'] == task]
        if len(td) < 3:
            continue
        r, p = pearsonr(td['prompt_tokens'], td['quality'])
        print(f"  {task:25s}  r={r:+.3f}  p={p:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Saturation experiment analysis')
    parser.add_argument('--input',      default='results/saturation_results.json')
    parser.add_argument('--output-dir', default='results/')
    args = parser.parse_args()

    fig_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print(f"\nLoading: {args.input}")
    df = load_data(args.input)
    print(f"  {len(df)} valid records | "
          f"{df['model'].nunique()} models | "
          f"{df['task'].nunique()} tasks | "
          f"levels {sorted(df['level'].unique())}")

    models_present = [m for m in MODEL_ORDER if m in df['model'].unique()]
    print(f"  Models: {models_present}")

    # Aggregate to level-means per (model, task)
    agg = aggregate(df)

    # Fit curves per (model, task)
    print("\nFitting curves …")
    fits: dict = {}
    for model in models_present:
        for task in TASKS:
            sub = agg[(agg['model'] == model) & (agg['task'] == task)]
            if len(sub) < 3:
                fits[(model, task)] = {'model_type': 'none', 'params': None,
                                       'rmse': np.nan, 'r2': np.nan,
                                       'saturation_tokens': np.nan, 'asymptote': np.nan}
                continue
            tokens = sub['mean_tokens'].values.astype(float)
            quality = sub['mean_quality'].values.astype(float)
            fit = fit_best_curve(tokens, quality)
            fits[(model, task)] = fit
            print(f"  {model:20s} | {task:22s} | {fit['model_type']:11s} "
                  f"R²={fit['r2']:.3f}  sat={fit['saturation_tokens']:.0f} tokens")

    # Correlation stats
    print_correlation_stats(df)

    # Figure 1: scaling curves
    print("\nGenerating figures …")
    plot_scaling_curves(agg, fits,
                        os.path.join(fig_dir, 'fig_sat1_scaling_curves.png'))

    # Figure 2: saturation heatmap
    plot_saturation_heatmap(fits, models_present,
                            os.path.join(fig_dir, 'fig_sat2_saturation_points.png'))

    # Summary CSV
    summary = build_summary(fits, models_present)
    csv_path = os.path.join(args.output_dir, 'saturation_summary.csv')
    summary.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Print quick summary
    print("\n── Saturation points (tokens) ──")
    pivot = summary.pivot(index='model', columns='task', values='saturation_tokens')
    pivot = pivot.reindex(models_present)
    print(pivot.to_string(float_format='{:.0f}'.format))

    r2_mean = summary['r2'].dropna().mean()
    print(f"\nMean R² across all fits: {r2_mean:.3f}")
    print("\nDone.")


if __name__ == '__main__':
    main()
