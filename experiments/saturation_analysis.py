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
from scipy.stats import pearsonr, f as f_dist  # noqa: F401 — used in print_correlation_stats

# ── Constants ─────────────────────────────────────────────────────────────────

TASKS = ['qa', 'summarization', 'classification', 'instruction_following', 'math_reasoning', 'product_extraction']
TASK_LABELS = {
    'qa': 'QA',
    'summarization': 'Summarization',
    'classification': 'Classification',
    'instruction_following': 'Instruction Following',
    'math_reasoning': 'Math Reasoning',
    'product_extraction': 'Product Extraction',
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


# ── Null model F-test ─────────────────────────────────────────────────────────

def null_model_ftest(tokens: np.ndarray, quality: np.ndarray, fit: dict) -> dict:
    """
    Test whether the fitted curve is significantly better than a flat line (null model).

    Null model: quality = mean (1 parameter).
    Fitted model: log (3 params) or sigmoid (4 params).

    Returns dict with ftest_F, ftest_p, ftest_significant.
    """
    n = len(quality)
    if fit.get('params') is None or fit.get('model_type') == 'none':
        return {'ftest_F': np.nan, 'ftest_p': np.nan, 'ftest_significant': False}

    # Null model residuals
    ss_null = float(np.sum((quality - quality.mean()) ** 2))

    # Fitted model residuals
    if fit['model_type'] == 'logarithmic':
        params = fit['params']
        pred = log_curve(tokens, *params)
        p_full = 3
    else:
        params = fit['params']
        pred = sigmoid_curve(tokens, *params)
        p_full = 4

    ss_fit = float(np.sum((quality - pred) ** 2))
    p_null = 1  # just the mean

    dfn = p_full - p_null  # numerator degrees of freedom
    dfd = n - p_full       # denominator degrees of freedom

    if dfd <= 0 or ss_fit <= 0 or ss_null <= ss_fit:
        return {'ftest_F': np.nan, 'ftest_p': np.nan, 'ftest_significant': False}

    f_stat = ((ss_null - ss_fit) / dfn) / (ss_fit / dfd)
    p_value = float(f_dist.sf(f_stat, dfn, dfd))

    return {
        'ftest_F': float(f_stat),
        'ftest_p': p_value,
        'ftest_significant': p_value < 0.05,
    }


# ── Bootstrap confidence intervals ───────────────────────────────────────────

def bootstrap_saturation_ci(
    df_raw: pd.DataFrame,
    model: str,
    task: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap CIs on saturation point by resampling example indices.

    Resamples the same 20 example indices (with replacement) and applies them
    across all 7 levels, preserving the paired structure. Then re-aggregates,
    re-fits, and extracts the saturation point.

    Returns dict with sat_median, sat_ci_lower, sat_ci_upper, bootstrap_fit_rate.
    """
    sub = df_raw[(df_raw['model'] == model) & (df_raw['task'] == task)].copy()
    if sub.empty:
        return {'sat_median': np.nan, 'sat_ci_lower': np.nan,
                'sat_ci_upper': np.nan, 'bootstrap_fit_rate': 0.0}

    example_ids = sorted(sub['example_id'].unique())
    n_examples = len(example_ids)
    levels = sorted(sub['level'].unique())

    # Pre-index for fast lookup: (level, example_id) -> (quality, prompt_tokens)
    sub_indexed = sub.set_index(['level', 'example_id'])

    rng = np.random.default_rng(seed)
    sat_points = []

    for _ in range(n_bootstrap):
        # Resample example indices with replacement — same set for all levels
        boot_ids = rng.choice(example_ids, size=n_examples, replace=True)

        # Build aggregated points per level
        level_tokens = []
        level_quality = []
        valid = True

        for level in levels:
            qualities = []
            tokens = []
            for eid in boot_ids:
                try:
                    row = sub_indexed.loc[(level, eid)]
                    # Handle potential duplicate index (same example_id at same level)
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    qualities.append(row['quality'])
                    tokens.append(row['prompt_tokens'])
                except KeyError:
                    continue

            if len(qualities) < 5:  # too few records at this level
                valid = False
                break

            level_quality.append(np.mean(qualities))
            level_tokens.append(np.mean(tokens))

        if not valid or len(level_quality) < 4:
            continue

        tok_arr = np.array(level_tokens, dtype=float)
        qual_arr = np.array(level_quality, dtype=float)

        fit = fit_best_curve(tok_arr, qual_arr)
        if fit['params'] is not None and not np.isnan(fit.get('saturation_tokens', np.nan)):
            sat_points.append(fit['saturation_tokens'])

    if not sat_points:
        return {'sat_median': np.nan, 'sat_ci_lower': np.nan,
                'sat_ci_upper': np.nan, 'bootstrap_fit_rate': 0.0}

    sat_arr = np.array(sat_points)
    return {
        'sat_median': float(np.median(sat_arr)),
        'sat_ci_lower': float(np.percentile(sat_arr, 2.5)),
        'sat_ci_upper': float(np.percentile(sat_arr, 97.5)),
        'bootstrap_fit_rate': len(sat_points) / n_bootstrap,
    }


# ── Figure 3: Forest plot with CIs ──────────────────────────────────────────

TASK_COLORS = {
    'qa': '#377eb8',
    'summarization': '#4daf4a',
    'classification': '#e41a1c',
    'instruction_following': '#ff7f00',
    'math_reasoning': '#984ea3',
    'product_extraction': '#a65628',
}


def plot_forest(summary: pd.DataFrame, out_path: str) -> None:
    """
    Forest plot: one row per (model, task) showing saturation point + 95% CI.
    Only includes pairs where F-test is significant.
    """
    sig = summary[summary['ftest_significant'] == True].copy()
    if sig.empty:
        print("  No significant pairs for forest plot — skipping.")
        return

    sig = sig.sort_values(['task', 'saturation_tokens'])
    sig['label'] = sig['model'] + ' — ' + sig['task'].map(TASK_LABELS)

    fig, ax = plt.subplots(figsize=(10, max(4, len(sig) * 0.4)))

    y_positions = range(len(sig))
    for i, (_, row) in enumerate(sig.iterrows()):
        color = TASK_COLORS.get(row['task'], '#333')
        ci_lo = row.get('sat_ci_lower', np.nan)
        ci_hi = row.get('sat_ci_upper', np.nan)
        sat = row['saturation_tokens']

        # Point estimate
        ax.plot(sat, i, 'o', color=color, markersize=7, zorder=3)

        # CI whiskers
        if not np.isnan(ci_lo) and not np.isnan(ci_hi):
            ax.plot([ci_lo, ci_hi], [i, i], '-', color=color,
                    linewidth=2, alpha=0.6, zorder=2)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(sig['label'].tolist(), fontsize=8)
    ax.set_xlabel('Saturation Point (tokens)', fontsize=10)
    ax.set_title('Saturation Points with 95% Bootstrap CI\n(only pairs where curve significantly beats flat line, F-test p<0.05)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    # Legend for task colors
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], marker='o', color=c, linestyle='', markersize=7, label=TASK_LABELS[t])
                      for t, c in TASK_COLORS.items()]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


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
    Subplot grid. Each subplot = one task.
    One line per model (aggregated mean across examples); fitted curve overlaid.
    """
    models_present = [m for m in MODEL_ORDER if m in agg['model'].unique()]
    tasks_present = [t for t in TASKS if t in agg['task'].unique()]
    n_tasks = len(tasks_present)
    ncols = min(n_tasks, 3)
    nrows = (n_tasks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_tasks == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for ax_idx, task in enumerate(tasks_present):
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
    tasks_present = [t for t in TASKS if any((m, t) in fits for m in models)]
    task_labels = [TASK_LABELS[t] for t in tasks_present]
    sat_matrix = np.full((len(models), len(tasks_present)), np.nan)
    for i, model in enumerate(models):
        for j, task in enumerate(tasks_present):
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
        for j in range(len(tasks_present)):
            val = sat_matrix[i, j]
            txt = f'{val:.0f}' if not np.isnan(val) else 'N/A'
            color = 'white' if (not np.isnan(val) and val > masked.max() * 0.6) else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                    color=color, fontweight='bold')

    ax.set_xticks(range(len(tasks_present)))
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

def build_summary(fits: dict, ftests: dict, boot_cis: dict, models: list[str], tasks: list[str] | None = None) -> pd.DataFrame:
    tasks_to_use = tasks if tasks is not None else TASKS
    rows = []
    for model in models:
        for task in tasks_to_use:
            key = (model, task)
            fit = fits.get(key, {})
            ft = ftests.get(key, {})
            ci = boot_cis.get(key, {})
            rows.append({
                'model': model,
                'task': task,
                'fit_type': fit.get('model_type', 'none'),
                'r2': fit.get('r2', np.nan),
                'rmse': fit.get('rmse', np.nan),
                'saturation_tokens': fit.get('saturation_tokens', np.nan),
                'asymptote': fit.get('asymptote', np.nan),
                'ftest_F': ft.get('ftest_F', np.nan),
                'ftest_p': ft.get('ftest_p', np.nan),
                'ftest_significant': ft.get('ftest_significant', False),
                'sat_median': ci.get('sat_median', np.nan),
                'sat_ci_lower': ci.get('sat_ci_lower', np.nan),
                'sat_ci_upper': ci.get('sat_ci_upper', np.nan),
                'bootstrap_fit_rate': ci.get('bootstrap_fit_rate', 0.0),
            })
    return pd.DataFrame(rows)


# ── Correlation: level vs quality (by task) ───────────────────────────────────

def print_correlation_stats(df: pd.DataFrame) -> None:
    tasks_in_data = [t for t in TASKS if t in df['task'].unique()]
    print("\n── Pearson r (prompt_tokens vs quality) by task ──")
    for task in tasks_in_data:
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
    tasks_present = [t for t in TASKS if t in df['task'].unique()]
    print(f"  Models: {models_present}")

    # Aggregate to level-means per (model, task)
    agg = aggregate(df)

    # Fit curves per (model, task)
    print("\nFitting curves …")
    fits: dict = {}
    for model in models_present:
        for task in tasks_present:
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

    # ── Fix #1: Null model F-tests ──────────────────────────────────────────
    print("\nRunning null model F-tests …")
    ftests: dict = {}
    for model in models_present:
        for task in tasks_present:
            key = (model, task)
            sub = agg[(agg['model'] == model) & (agg['task'] == task)]
            if len(sub) < 3 or fits[key].get('params') is None:
                ftests[key] = {'ftest_F': np.nan, 'ftest_p': np.nan, 'ftest_significant': False}
                continue
            tokens = sub['mean_tokens'].values.astype(float)
            quality = sub['mean_quality'].values.astype(float)
            ft = null_model_ftest(tokens, quality, fits[key])
            ftests[key] = ft
            sig_str = "*** SIGNIFICANT" if ft['ftest_significant'] else "    not significant"
            print(f"  {model:20s} | {task:22s} | F={ft['ftest_F']:7.2f}  p={ft['ftest_p']:.4f}  {sig_str}")

    n_sig = sum(1 for v in ftests.values() if v.get('ftest_significant'))
    print(f"  {n_sig}/{len(ftests)} pairs have significant curve fit (p<0.05)")

    # ── Fix #3: Bootstrap confidence intervals ──────────────────────────────
    print("\nRunning bootstrap CIs (1000 iterations per pair) …")
    boot_cis: dict = {}
    for model in models_present:
        for task in tasks_present:
            key = (model, task)
            ci = bootstrap_saturation_ci(df, model, task, n_bootstrap=1000)
            boot_cis[key] = ci
            if not np.isnan(ci['sat_median']):
                print(f"  {model:20s} | {task:22s} | "
                      f"median={ci['sat_median']:6.1f}  "
                      f"CI=[{ci['sat_ci_lower']:.1f}, {ci['sat_ci_upper']:.1f}]  "
                      f"fit_rate={ci['bootstrap_fit_rate']:.1%}")
            else:
                print(f"  {model:20s} | {task:22s} | bootstrap failed")

    # ── Figures ─────────────────────────────────────────────────────────────
    print("\nGenerating figures …")

    # Figure 1: scaling curves (original)
    plot_scaling_curves(agg, fits,
                        os.path.join(fig_dir, 'fig_sat1_scaling_curves.png'))

    # Figure 2: saturation heatmap (original)
    plot_saturation_heatmap(fits, models_present,
                            os.path.join(fig_dir, 'fig_sat2_saturation_points.png'))

    # Summary CSV (with F-test + bootstrap columns)
    summary = build_summary(fits, ftests, boot_cis, models_present, tasks_present)
    csv_path = os.path.join(args.output_dir, 'saturation_summary.csv')
    summary.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Figure 3: Forest plot with CIs (new)
    plot_forest(summary, os.path.join(fig_dir, 'fig_sat3_forest_plot.png'))

    # ── Print summary ───────────────────────────────────────────────────────
    print("\n── Saturation points (tokens) ──")
    pivot = summary.pivot(index='model', columns='task', values='saturation_tokens')
    pivot = pivot.reindex(models_present)
    print(pivot.to_string(float_format='{:.0f}'.format))

    print("\n── F-test results ──")
    for _, row in summary.iterrows():
        sig = "✓" if row['ftest_significant'] else "✗"
        ci_str = ""
        if not np.isnan(row.get('sat_ci_lower', np.nan)):
            ci_str = f"  CI=[{row['sat_ci_lower']:.0f}, {row['sat_ci_upper']:.0f}]"
        print(f"  {sig} {row['model']:20s} | {row['task']:22s} | "
              f"p={row['ftest_p']:.4f}  sat={row['saturation_tokens']:.0f}{ci_str}")

    r2_mean = summary['r2'].dropna().mean()
    print(f"\nMean R² across all fits: {r2_mean:.3f}")
    print(f"Significant pairs: {n_sig}/{len(ftests)}")
    print("\nDone.")


if __name__ == '__main__':
    main()
