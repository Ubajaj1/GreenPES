"""
Analysis script for GreenPES benchmark results.

Reads results/benchmark_results.json and produces:
  - results/stats_summary.csv    (all test statistics)
  - results/figures/fig1_strategy_heatmap.png
  - results/figures/fig2_model_comparison.png
  - results/figures/fig3_quality_efficiency_scatter.png
  - results/figures/fig4_greenpes_distribution.png

Usage:
    python experiments/analysis.py
    python experiments/analysis.py --input results/benchmark_results.json
    python experiments/analysis.py --output-dir results/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving files
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import f_oneway, kendalltau, pearsonr, ttest_rel, tukey_hsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# ── Constants ─────────────────────────────────────────────────────────────────

STRATEGIES = ['zero_shot', 'zero_shot_verbose', 'few_shot', 'cot', 'concise']
TASKS = ['qa', 'summarization', 'classification', 'instruction_following']

# Approximate capability tier order (small → large) for RQ3 figure
MODEL_ORDER = [
    'llama-3.1-8b',
    'gemini-flash',
    'qwen3-32b',
    'llama-3.3-70b',
    'kimi-k2',
    'gpt-4o-mini',
    'claude-haiku',
]

REQUIRED_COLS = {
    'model', 'task', 'strategy', 'greenpes',
    'quality', 'input_tokens', 'output_tokens', 'total_tokens',
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    """Load benchmark_results.json, drop error records, validate schema."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(p) as f:
        raw = json.load(f)

    # Drop records that represent failed API calls
    good = [r for r in raw if 'error' not in r]
    n_errors = len(raw) - len(good)
    if n_errors:
        print(f"  Dropped {n_errors} error records (failed API calls)")

    df = pd.DataFrame(good)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Results missing required columns: {missing}")

    # Data summary
    print(f"\nData summary ({len(df)} successful runs):")
    for col, vals in [('model', df['model'].unique()), ('task', df['task'].unique()),
                      ('strategy', df['strategy'].unique())]:
        print(f"  {col}s ({len(vals)}): {', '.join(sorted(vals))}")

    return df


# ── RQ functions ──────────────────────────────────────────────────────────────

def rq1_strategy_effect(df: pd.DataFrame) -> tuple[Figure, list[dict]]:
    """
    RQ1: Does prompting strategy significantly affect GreenPES?

    - One-way ANOVA across strategy groups
    - Post-hoc Tukey HSD for pairwise comparisons
    - Effect size: eta-squared
    - Figure: heatmap of mean GreenPES by strategy × task
    """
    print("\n── RQ1: Strategy effect on GreenPES ──")

    strategies = df['strategy'].unique()
    groups = [df[df['strategy'] == s]['greenpes'].to_numpy() for s in strategies]  # type: ignore[union-attr]

    # One-way ANOVA
    f_stat, p_val = f_oneway(*groups)
    # Eta-squared: SS_between / SS_total
    grand_mean = df['greenpes'].mean()
    ss_between = sum(
        len(g) * (g.mean() - grand_mean) ** 2
        for g in groups
    )
    ss_total = ((df['greenpes'] - grand_mean) ** 2).sum()
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    print(f"  ANOVA: F={f_stat:.3f}, p={p_val:.4f}, η²={eta_sq:.3f}")

    stats = [{
        'rq': 'RQ1',
        'test': 'one-way ANOVA',
        'statistic': round(f_stat, 4),
        'p_value': round(p_val, 4),
        'effect_size': round(eta_sq, 4),
        'effect_metric': 'eta_squared',
        'notes': f'groups={list(strategies)}',
    }]

    # Tukey HSD post-hoc (scipy >= 1.8)
    tukey = tukey_hsd(*groups)
    for i, s1 in enumerate(strategies):
        for j, s2 in enumerate(strategies):
            if j <= i:
                continue
            p = tukey.pvalue[i, j]
            stats.append({
                'rq': 'RQ1',
                'test': 'Tukey HSD',
                'statistic': round(tukey.statistic[i, j], 4),
                'p_value': round(p, 4),
                'effect_size': None,
                'effect_metric': None,
                'notes': f'{s1} vs {s2}',
            })
            sig = '✓' if p < 0.05 else '✗'
            print(f"    {sig} {s1} vs {s2}: p={p:.4f}")

    # Figure 1: heatmap of mean GreenPES by strategy × task
    pivot: pd.DataFrame = df.groupby(['strategy', 'task'])['greenpes'].mean().unstack()  # type: ignore[assignment]
    # Ensure consistent ordering
    present_strategies = [s for s in STRATEGIES if s in pivot.index]
    present_tasks = [t for t in TASKS if t in pivot.columns]
    pivot = pivot.reindex(index=present_strategies, columns=present_tasks)

    fig, ax = plt.subplots(figsize=(8, 5))
    if pivot.empty:
        ax.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Figure 1: Mean GreenPES by Strategy × Task', fontsize=13, pad=12)
        fig.tight_layout()
        return fig, stats
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        linewidths=0.5,
        cbar_kws={'label': 'Mean GreenPES'},
    )
    ax.set_title('Figure 1: Mean GreenPES by Strategy × Task', fontsize=13, pad=12)
    ax.set_xlabel('Task')
    ax.set_ylabel('Strategy')
    fig.tight_layout()

    return fig, stats


def rq2_token_efficiency(df: pd.DataFrame) -> tuple[Figure, list[dict]]:
    """
    RQ2: Which strategy is most token-efficient per task?

    Groups by (task, strategy), reports mean GreenPES and mean tokens.
    Figure: grouped bar chart — tasks × strategies, y = mean GreenPES.
    """
    print("\n── RQ2: Token efficiency by strategy × task ──")

    agg = (
        df.groupby(['task', 'strategy'])
        .agg(mean_greenpes=('greenpes', 'mean'),
             std_greenpes=('greenpes', 'std'),
             mean_tokens=('total_tokens', 'mean'))
        .reset_index()
    )

    stats = []
    for task in df['task'].unique():
        sub = agg[agg['task'] == task].sort_values('mean_greenpes', ascending=False)  # type: ignore[call-overload]
        winner = sub.iloc[0]
        print(f"  {task}: best strategy = {winner['strategy']} "
              f"(GreenPES={winner['mean_greenpes']:.2f}, tokens={winner['mean_tokens']:.0f})")
        stats.append({
            'rq': 'RQ2',
            'test': 'winner',
            'statistic': round(float(winner['mean_greenpes']), 4),
            'p_value': None,
            'effect_size': None,
            'effect_metric': None,
            'notes': f"task={task}, strategy={winner['strategy']}, "
                     f"mean_tokens={round(float(winner['mean_tokens']), 1)}",
        })
        for _, row in sub.iterrows():
            stats.append({
                'rq': 'RQ2',
                'test': 'mean_greenpes',
                'statistic': round(float(row['mean_greenpes']), 4),  # type: ignore[arg-type]
                'p_value': None,
                'effect_size': None,
                'effect_metric': None,
                'notes': f"task={task}, strategy={row['strategy']}, "
                         f"mean_tokens={round(float(row['mean_tokens']), 1)}",  # type: ignore[arg-type]
            })

    # Figure 2: grouped bar chart
    present_tasks = [t for t in TASKS if t in agg['task'].values]
    present_strategies = [s for s in STRATEGIES if s in agg['strategy'].values]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(present_tasks))
    bar_width = 0.15
    for i, strategy in enumerate(present_strategies):
        sub = agg[agg['strategy'] == strategy].set_index('task')
        heights = [float(sub.loc[t, 'mean_greenpes']) if t in sub.index else 0.0  # type: ignore[arg-type]
                   for t in present_tasks]
        errs = [float(sub.loc[t, 'std_greenpes']) if t in sub.index else 0.0  # type: ignore[arg-type]
                for t in present_tasks]
        offset = (i - len(present_strategies) / 2) * bar_width + bar_width / 2
        ax.bar([xi + offset for xi in x], heights, bar_width,
               label=strategy, yerr=errs, capsize=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(present_tasks)
    ax.set_ylabel('Mean GreenPES')
    ax.set_title('Figure 2: GreenPES by Strategy per Task', fontsize=13)
    ax.legend(title='Strategy', bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.tight_layout()

    return fig, stats


def rq3_model_comparison(df: pd.DataFrame) -> tuple[Figure, list[dict]]:
    """
    RQ3: Do smaller models achieve competitive GreenPES vs larger ones?

    - Groups by model, computes mean GreenPES ± std
    - Orders models by MODEL_ORDER (small → large) where possible
    - Figure: horizontal bar chart with error bars (one row per model)
    - Stats: one row per model
    """
    print("\n── RQ3: Model comparison ──")

    agg = (
        df.groupby('model')
        .agg(mean_greenpes=('greenpes', 'mean'),
             std_greenpes=('greenpes', 'std'),
             n=('greenpes', 'count'))
        .reset_index()
    )

    # Order by MODEL_ORDER where possible; unknown models appended at end
    known = [m for m in MODEL_ORDER if m in agg['model'].values]
    unknown = [m for m in agg['model'].values if m not in MODEL_ORDER]
    order = known + unknown
    agg = agg.set_index('model').reindex(order).reset_index()

    stats = []
    for _, row in agg.iterrows():
        mean_val = float(row['mean_greenpes'])  # type: ignore[arg-type]
        std_raw = float(row['std_greenpes'])    # type: ignore[arg-type]
        std_val = std_raw if std_raw == std_raw else 0.0  # guard NaN
        n_val = int(row['n'])                   # type: ignore[arg-type]
        model_name = str(row['model'])          # type: ignore[arg-type]
        print(f"  {model_name}: mean={mean_val:.2f}, std={std_val:.2f}, n={n_val}")
        stats.append({
            'rq': 'RQ3',
            'test': 'mean_greenpes',
            'statistic': round(mean_val, 4),
            'p_value': None,
            'effect_size': round(std_val, 4),   # std used here (unlike RQ1 eta-squared)
            'effect_metric': 'std',
            'notes': f'model={model_name}, n={n_val}',
        })

    # Figure 3: horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    means  = agg['mean_greenpes'].tolist()
    stds   = [v if v == v else 0.0 for v in agg['std_greenpes'].fillna(0.0).tolist()]
    labels = agg['model'].tolist()
    y = range(len(labels))
    ax.barh(list(y), means, xerr=stds, capsize=4, color='steelblue', alpha=0.85)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Mean GreenPES')
    ax.set_title('Figure 3: Mean GreenPES by Model (small → large)', fontsize=13)
    ax.invert_yaxis()   # smallest model at top
    fig.tight_layout()

    return fig, stats


def rq4_quality_tradeoff(df: pd.DataFrame) -> tuple[Figure, list[dict]]:
    """
    RQ4: Is there a quality–efficiency tradeoff across strategies?

    - Pearson r between total_tokens and quality (all records)
    - Scatter: one point per (strategy, task) mean
    - Pareto-efficient points annotated
    - Figure: quality vs tokens scatter
    """
    print("\n── RQ4: Quality–efficiency tradeoff ──")

    # Pearson r on all records
    if df['total_tokens'].std() == 0 or df['quality'].std() == 0:
        print("  Pearson r: skipped (constant column, insufficient variance)")
        r, p = float('nan'), float('nan')
    else:
        r, p = pearsonr(df['total_tokens'], df['quality'])
        print(f"  Pearson r (tokens vs quality): r={r:.3f}, p={p:.4f}")

    stats: list[dict] = [{
        'rq': 'RQ4',
        'test': 'Pearson r',
        'statistic': round(float(r), 4),  # type: ignore[arg-type]
        'p_value': round(float(p), 4),    # type: ignore[arg-type]
        'effect_size': None,
        'effect_metric': None,
        'notes': 'total_tokens vs quality, all records',
    }]

    # Aggregate to (strategy, task) level for scatter
    agg = (
        df.groupby(['strategy', 'task'])
        .agg(mean_tokens=('total_tokens', 'mean'),
             mean_quality=('quality', 'mean'))
        .reset_index()
    )

    # Pearson r at aggregate level (matches what the scatter plot shows)
    r_agg: float = float('nan')
    if len(agg) >= 2 and agg['mean_tokens'].std() > 0 and agg['mean_quality'].std() > 0:
        _res = pearsonr(agg['mean_tokens'], agg['mean_quality'])  # type: ignore[call-overload]
        r_agg = float(_res[0])  # type: ignore[arg-type]

    # Pareto frontier: for each token budget, highest quality
    agg_sorted = agg.sort_values('mean_tokens')  # type: ignore[call-overload]
    pareto: list[dict] = []
    best_q = -1.0
    for _, row in agg_sorted.iterrows():
        q = float(row['mean_quality'])   # type: ignore[arg-type]
        if q > best_q:
            best_q = q
            pareto.append({
                'strategy': str(row['strategy']),   # type: ignore[arg-type]
                'task': str(row['task']),            # type: ignore[arg-type]
                'mean_tokens': float(row['mean_tokens']),  # type: ignore[arg-type]
                'mean_quality': q,
            })

    # Figure 4: scatter
    strategy_colors = {s: c for s, c in zip(
        STRATEGIES, sns.color_palette('tab10', len(STRATEGIES))
    )}
    task_markers = {'qa': 'o', 'summarization': 's',
                    'classification': '^', 'instruction_following': 'D'}

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in agg.iterrows():
        color = strategy_colors.get(str(row['strategy']), 'grey')   # type: ignore[arg-type]
        marker = task_markers.get(str(row['task']), 'o')             # type: ignore[arg-type]
        ax.scatter(float(row['mean_tokens']), float(row['mean_quality']),  # type: ignore[arg-type]
                   color=color, marker=marker, s=80, zorder=3)

    # Pareto frontier line
    if len(pareto) > 1:
        px = [pt['mean_tokens'] for pt in pareto]
        py = [pt['mean_quality'] for pt in pareto]
        ax.plot(px, py, 'k--', linewidth=1, label='Pareto frontier', zorder=2)
        for pt in pareto:
            ax.annotate(f"{pt['strategy']}\n({pt['task']})",
                        (pt['mean_tokens'], pt['mean_quality']),
                        textcoords='offset points', xytext=(5, 5), fontsize=7)

    # Legend for strategies (color)
    for strategy, color in strategy_colors.items():
        if strategy in agg['strategy'].values:
            ax.scatter([], [], color=color, label=strategy, s=60)
    # Legend for tasks (marker)
    for task, marker in task_markers.items():
        if task in agg['task'].values:
            ax.scatter([], [], color='grey', marker=marker, label=task, s=60)

    ax.set_xlabel('Mean Total Tokens')
    ax.set_ylabel('Mean Quality Score')
    r_agg_label = f'{r_agg:.2f}' if r_agg == r_agg else 'N/A'
    ax.set_title(f'Figure 4: Quality vs Token Cost (agg r={r_agg_label})', fontsize=13)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.tight_layout()

    return fig, stats


def rq5_strategy_transfer(df: pd.DataFrame) -> tuple[Figure, list[dict]]:
    """
    RQ5: Does the optimal prompting strategy for one model transfer to others?

    - Two-way ANOVA: quality ~ model + strategy + model:strategy (interaction effect)
    - Per-model best strategy table
    - Transfer matrix [src, tgt]: quality retention when using src's best strategy on tgt
    - Figure 5: 7×7 transfer heatmap
    """
    print("\n── RQ5: Strategy transfer across models ──")

    # Filter to core strategies only
    df5 = df[df['strategy'].isin(STRATEGIES)].copy()

    stats: list[dict] = []

    # Two-way ANOVA with interaction
    try:
        fit = ols('quality ~ C(model) + C(strategy) + C(model):C(strategy)', data=df5).fit()
        anova_table = anova_lm(fit, typ=2)
        interaction_key = 'C(model):C(strategy)'
        F_int = float(anova_table.loc[interaction_key, 'F'])
        p_int = float(anova_table.loc[interaction_key, 'PR(>F)'])
    except Exception as e:
        print(f"  Two-way ANOVA failed: {e}")
        F_int, p_int = float('nan'), float('nan')

    print(f"  Interaction (model × strategy): F={F_int:.3f}, p={p_int:.4f}")
    stats.append({
        'rq': 'RQ5',
        'test': 'two-way ANOVA interaction',
        'statistic': round(F_int, 4) if F_int == F_int else None,
        'p_value': round(p_int, 4) if p_int == p_int else None,
        'effect_size': None,
        'effect_metric': None,
        'notes': 'quality ~ C(model) + C(strategy) + C(model):C(strategy)',
    })

    # Per-model best strategy (highest mean quality)
    quality_pivot: pd.DataFrame = (  # type: ignore[assignment]
        df5.groupby(['model', 'strategy'])['quality']
        .mean()
        .unstack(fill_value=float('nan'))  # type: ignore[union-attr]
    )
    best_strats: dict[str, str] = {}
    for model_name in quality_pivot.index:
        row = quality_pivot.loc[model_name]
        best = str(row.idxmax())
        best_strats[model_name] = best
        print(f"  {model_name}: best strategy = {best} "
              f"(quality={row.max():.3f})")
        stats.append({
            'rq': 'RQ5',
            'test': 'best_strategy',
            'statistic': round(float(row.max()), 4),
            'p_value': None,
            'effect_size': None,
            'effect_metric': None,
            'notes': f'model={model_name}, best={best}',
        })

    # Transfer matrix: cell[src, tgt] = quality(tgt, best_src) / quality(tgt, best_tgt)
    models = sorted(quality_pivot.index)
    transfer = pd.DataFrame(index=models, columns=models, dtype=float)
    for src in models:
        for tgt in models:
            if src == tgt:
                transfer.loc[src, tgt] = 1.0
                continue
            src_best = best_strats.get(src)
            if src_best is None or src_best not in quality_pivot.columns:
                transfer.loc[src, tgt] = float('nan')
                continue
            q_with_src = float(quality_pivot.loc[tgt, src_best]) if tgt in quality_pivot.index else float('nan')
            q_best_tgt = float(quality_pivot.loc[tgt].max()) if tgt in quality_pivot.index else float('nan')
            transfer.loc[src, tgt] = q_with_src / q_best_tgt if q_best_tgt > 0 else float('nan')

    # Figure 5: transfer heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        transfer.astype(float),
        ax=ax,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={'label': 'Quality Retention'},
    )
    ax.set_title(
        'Figure 5: Strategy Transfer Matrix\n'
        '(row = source model\'s best strategy applied to column model)',
        fontsize=12,
    )
    ax.set_xlabel('Target Model')
    ax.set_ylabel('Source Model (strategy donor)')
    fig.tight_layout()

    return fig, stats


def rq6_model_strategy_interaction(df: pd.DataFrame) -> tuple[Figure, list[dict]]:
    """
    RQ6: Do models agree on which strategy is best?

    - Kendall's tau for every pair of models on their strategy rankings
    - Universality index: fraction of pairs with tau > 0.8
    - Figure 6: interaction plot (x=strategy, y=quality, one line per model)
    """
    print("\n── RQ6: Model–strategy interaction ──")

    df6 = df[df['strategy'].isin(STRATEGIES)].copy()

    quality_pivot: pd.DataFrame = (  # type: ignore[assignment]
        df6.groupby(['model', 'strategy'])['quality']
        .mean()
        .unstack(fill_value=float('nan'))  # type: ignore[union-attr]
    )
    models = sorted(quality_pivot.index)
    common_strategies = [s for s in STRATEGIES if s in quality_pivot.columns]

    tau_rows: list[dict] = []
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if j <= i:
                continue
            r1 = quality_pivot.loc[m1, common_strategies].tolist()
            r2 = quality_pivot.loc[m2, common_strategies].tolist()
            kt = kendalltau(r1, r2)
            tau_val = float(kt.statistic)  # type: ignore[union-attr]
            p_val = float(kt.pvalue)       # type: ignore[union-attr]
            sig = '✓' if p_val < 0.05 else '✗'
            print(f"  {sig} {m1} vs {m2}: tau={tau_val:.3f}, p={p_val:.4f}")
            tau_rows.append({'m1': m1, 'm2': m2, 'tau': tau_val, 'p': p_val})

    universality = (
        sum(1 for t in tau_rows if t['tau'] > 0.8) / len(tau_rows)
        if tau_rows else float('nan')
    )
    print(f"  Universality index (tau > 0.8): {universality:.3f}")

    stats: list[dict] = [{
        'rq': 'RQ6',
        'test': 'universality_index',
        'statistic': round(universality, 4) if universality == universality else None,
        'p_value': None,
        'effect_size': None,
        'effect_metric': None,
        'notes': f'fraction of model pairs with tau > 0.8 (n_pairs={len(tau_rows)})',
    }]
    for t in tau_rows:
        stats.append({
            'rq': 'RQ6',
            'test': 'Kendall tau',
            'statistic': round(t['tau'], 4),
            'p_value': round(t['p'], 4),
            'effect_size': None,
            'effect_metric': None,
            'notes': f"{t['m1']} vs {t['m2']}",
        })

    # Figure 6: interaction plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name in models:
        means = [
            float(quality_pivot.loc[model_name, s]) if s in quality_pivot.columns else float('nan')
            for s in common_strategies
        ]
        ax.plot(common_strategies, means, marker='o', label=model_name)

    ax.set_xlabel('Strategy')
    ax.set_ylabel('Mean Quality Score')
    ax.set_title(
        'Figure 6: Model × Strategy Interaction\n(crossing lines = strategies do not transfer universally)',
        fontsize=12,
    )
    ax.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha='right')
    fig.tight_layout()

    return fig, stats


def rq7_scaling_laws(df: pd.DataFrame) -> tuple[tuple[Figure, Figure], list[dict]]:
    """
    RQ7: Is there a predictable relationship between prompt token investment and quality?

    For each (model, task) pair, fits three curves to (total_tokens, quality):
      - Power law:   q = a * tokens^b + c
      - Logarithmic: q = a * log(tokens) + b
      - Sigmoid:     q = L / (1 + exp(-k*(tokens - x0)))

    Selects best fit by AIC. Extracts saturation point (tokens where
    marginal quality gain < 0.001/token).

    Figures:
      - Figure 7: scatter + best-fit curves, 4 subplots (one per task, models colour-coded)
      - Figure 8: saturation point bar chart, per task × model
    """
    print("\n── RQ7: Token efficiency scaling laws ──")

    # ── Curve definitions ────────────────────────────────────────────────────
    def power_law(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.power(np.maximum(x, 1e-6), b) + c

    def logarithmic(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * np.log(np.maximum(x, 1e-6)) + b

    def sigmoid(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
        return L / (1.0 + np.exp(-k * (x - x0)))

    def aic(n: int, k_params: int, residuals: np.ndarray) -> float:
        """AIC = n*log(SSR/n) + 2*k  (Gaussian log-likelihood approximation)."""
        ssr = float(np.sum(residuals ** 2))
        if ssr <= 0 or n <= 0:
            return float('inf')
        return n * np.log(ssr / n) + 2 * k_params

    def saturation_point(x_dense: np.ndarray, y_dense: np.ndarray, threshold: float = 0.001) -> float:
        """Token count where marginal quality gain drops below threshold/token."""
        dq = np.diff(y_dense)
        dx = np.diff(x_dense)
        marginal = dq / np.maximum(dx, 1e-9)
        below = np.where(marginal < threshold)[0]
        return float(x_dense[below[0]]) if len(below) > 0 else float(x_dense[-1])

    # ── Per (model, task) fitting ─────────────────────────────────────────────
    present_tasks = [t for t in TASKS if t in df['task'].unique()]
    present_models = [m for m in MODEL_ORDER if m in df['model'].unique()] + \
                     [m for m in df['model'].unique() if m not in MODEL_ORDER]

    fit_results: list[dict] = []
    stats: list[dict] = []

    agg = (
        df.groupby(['model', 'task', 'strategy'])
        .agg(mean_tokens=('total_tokens', 'mean'), mean_quality=('quality', 'mean'))
        .reset_index()
    )

    for model_name in present_models:
        for task_name in present_tasks:
            sub = agg[(agg['model'] == model_name) & (agg['task'] == task_name)]
            if len(sub) < 3:
                continue

            x = sub['mean_tokens'].to_numpy(dtype=float)  # type: ignore[union-attr]
            y = sub['mean_quality'].to_numpy(dtype=float)  # type: ignore[union-attr]
            sort_idx = np.argsort(x)
            x, y = x[sort_idx], y[sort_idx]

            best_name = 'none'
            best_aic = float('inf')
            best_fn = None
            best_params: list[float] = []

            # Try each curve
            fits: dict[str, tuple] = {}

            try:
                p, _ = curve_fit(power_law, x, y, p0=[1.0, 0.3, 0.0], maxfev=5000)
                resid = y - power_law(x, *p)
                a_val = aic(len(x), 3, resid)
                fits['power_law'] = (power_law, list(p), a_val)
            except Exception:
                pass

            try:
                p, _ = curve_fit(logarithmic, x, y, p0=[0.1, 0.5], maxfev=5000)
                resid = y - logarithmic(x, *p)
                a_val = aic(len(x), 2, resid)
                fits['logarithmic'] = (logarithmic, list(p), a_val)
            except Exception:
                pass

            try:
                p0 = [float(y.max()), 0.1, float(np.median(x))]
                p, _ = curve_fit(sigmoid, x, y, p0=p0, maxfev=5000)
                resid = y - sigmoid(x, *p)
                a_val = aic(len(x), 3, resid)
                fits['sigmoid'] = (sigmoid, list(p), a_val)
            except Exception:
                pass

            for fname, (fn, params, a_val) in fits.items():
                if a_val < best_aic:
                    best_aic = a_val
                    best_name = fname
                    best_fn = fn
                    best_params = params

            # Compute saturation point using best fit
            sat_point: float = float('nan')
            if best_fn is not None and best_params:
                x_dense = np.linspace(x.min(), x.max() * 2, 500)
                y_dense = best_fn(x_dense, *best_params)
                sat_point = saturation_point(x_dense, y_dense)

            fit_results.append({
                'model': model_name,
                'task': task_name,
                'best_fit': best_name,
                'best_aic': round(best_aic, 3) if best_aic != float('inf') else None,
                'saturation_tokens': round(sat_point, 1) if sat_point == sat_point else None,
                'x': x.tolist(),
                'y': y.tolist(),
                'best_fn': best_fn,
                'best_params': best_params,
            })

            print(f"  {model_name}/{task_name}: best={best_name}, "
                  f"AIC={best_aic:.2f}, saturation≈{sat_point:.0f} tokens")

            stats.append({
                'rq': 'RQ7',
                'test': 'curve_fit',
                'statistic': round(best_aic, 3) if best_aic != float('inf') else None,
                'p_value': None,
                'effect_size': round(sat_point, 1) if sat_point == sat_point else None,
                'effect_metric': 'saturation_tokens',
                'notes': f'model={model_name}, task={task_name}, best_fit={best_name}',
            })

    # ── Figure 7: scatter + fit curves, one subplot per task ─────────────────
    n_tasks = len(present_tasks)
    fig7, axes7 = plt.subplots(1, max(n_tasks, 1), figsize=(5 * max(n_tasks, 1), 4), squeeze=False)
    colors = sns.color_palette('tab10', len(present_models))
    model_color = {m: colors[i] for i, m in enumerate(present_models)}

    for ti, task_name in enumerate(present_tasks):
        ax = axes7[0, ti]
        task_fits = [r for r in fit_results if r['task'] == task_name]
        for r in task_fits:
            color = model_color.get(r['model'], 'grey')
            ax.scatter(r['x'], r['y'], color=color, s=40, zorder=3, label=r['model'])
            if r['best_fn'] is not None and r['best_params'] and len(r['x']) >= 2:
                x_dense = np.linspace(min(r['x']) * 0.9, max(r['x']) * 1.2, 200)
                try:
                    y_dense = r['best_fn'](x_dense, *r['best_params'])
                    ax.plot(x_dense, y_dense, color=color, linewidth=1.2, alpha=0.7)
                except Exception:
                    pass
        ax.set_title(task_name, fontsize=11)
        ax.set_xlabel('Mean Total Tokens')
        ax.set_ylabel('Mean Quality')

    # Deduplicate legend entries (one per model)
    handles, labels = axes7[0, 0].get_legend_handles_labels()
    seen: dict[str, int] = {}
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = 1
            unique_h.append(h)
            unique_l.append(l)
    if unique_h:
        axes7[0, -1].legend(unique_h, unique_l, title='Model',
                            bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

    fig7.suptitle('Figure 7: Token Efficiency Scaling Curves', fontsize=13)
    fig7.tight_layout()

    # ── Figure 8: saturation point bar chart ─────────────────────────────────
    sat_data = [
        {'model': r['model'], 'task': r['task'], 'saturation_tokens': r['saturation_tokens']}
        for r in fit_results
        if r['saturation_tokens'] is not None
    ]

    fig8, ax8 = plt.subplots(figsize=(10, 5))
    if sat_data:
        sat_df = pd.DataFrame(sat_data)
        x_pos = range(len(present_tasks))
        bar_width = 0.8 / max(len(present_models), 1)
        for mi, model_name in enumerate(present_models):
            sub = sat_df[sat_df['model'] == model_name].set_index('task')
            heights = [
                float(sub.loc[t, 'saturation_tokens']) if t in sub.index else 0.0
                for t in present_tasks
            ]
            offset = (mi - len(present_models) / 2) * bar_width + bar_width / 2
            ax8.bar(
                [xi + offset for xi in x_pos],
                heights, bar_width,
                label=model_name,
                color=model_color.get(model_name, 'grey'),
                alpha=0.85,
            )
        ax8.set_xticks(list(x_pos))
        ax8.set_xticklabels(present_tasks)
        ax8.set_ylabel('Saturation Token Count')
        ax8.set_title('Figure 8: Prompt Token Saturation Point by Task × Model', fontsize=12)
        ax8.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    else:
        ax8.text(0.5, 0.5, 'Insufficient data for saturation analysis',
                 ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('Figure 8: Saturation Points (insufficient data)', fontsize=12)

    fig8.tight_layout()

    return (fig7, fig8), stats


def rq8_optimizer_effectiveness(
    df: pd.DataFrame,
) -> tuple[tuple[Figure, Figure], list[dict]]:
    """
    RQ8: Can automatic prompt compression maintain quality while reducing tokens?

    Reads optimizer_results.json-style data (columns: method, compression_ratio,
    quality_retained, strategy, model, task).

    - Figure 9: Compression ratio vs. quality retention scatter.
                Color = method; Pareto frontier highlighted.
    - Figure 10: Mean compression ratio bar chart, grouped by original strategy.
    - Stats: Paired t-test comparing LLM optimizer vs. each baseline on
             quality_retained, per (model, task, strategy, example_id) triplet.
    """
    print("\n── RQ8: Optimizer effectiveness ──")

    OPTIMIZER_METHODS = ['remove_filler', 'truncate_examples', 'add_concise_suffix', 'llm_optimizer']
    stats: list[dict] = []

    # ── Paired t-tests: llm_optimizer vs. each baseline ──────────────────────
    # Key for pairing: (model, task, strategy, example_id)
    pair_keys = ['model', 'task', 'strategy', 'example_id']

    if 'method' in df.columns and 'quality_retained' in df.columns:
        llm_rows = df[df['method'] == 'llm_optimizer'].copy()
        for baseline in ['remove_filler', 'truncate_examples', 'add_concise_suffix']:
            base_rows = df[df['method'] == baseline].copy()
            # Merge on pair keys to get matched pairs
            merged = llm_rows.merge(
                base_rows[pair_keys + ['quality_retained', 'compression_ratio']],  # type: ignore[index]
                on=pair_keys,
                suffixes=('_llm', f'_{baseline}'),
            )
            if len(merged) < 2:
                print(f"  Insufficient paired data for llm_optimizer vs {baseline}")
                stats.append({
                    'rq': 'RQ8',
                    'test': f'paired_t_test vs {baseline}',
                    'statistic': None,
                    'p_value': None,
                    'effect_size': None,
                    'effect_metric': None,
                    'notes': f'insufficient paired data (n={len(merged)})',
                })
                continue
            llm_qr = merged['quality_retained_llm'].tolist()
            base_qr = merged[f'quality_retained_{baseline}'].tolist()
            t_stat, p_val = ttest_rel(llm_qr, base_qr)
            mean_diff = float(np.mean(np.array(llm_qr) - np.array(base_qr)))
            sig = '✓' if p_val < 0.05 else '✗'
            print(
                f"  {sig} llm_optimizer vs {baseline}: "
                f"t={t_stat:.3f}, p={p_val:.4f}, mean_diff={mean_diff:+.3f}"
            )
            stats.append({
                'rq': 'RQ8',
                'test': f'paired_t_test vs {baseline}',
                'statistic': round(float(t_stat), 4),
                'p_value': round(float(p_val), 4),
                'effect_size': round(mean_diff, 4),
                'effect_metric': 'mean_quality_retained_diff (llm - baseline)',
                'notes': f'n_pairs={len(merged)}',
            })

        # Mean compression ratio per method
        for method in OPTIMIZER_METHODS:
            sub = df[df['method'] == method]
            if len(sub) == 0:
                continue
            mean_cr = float(sub['compression_ratio'].mean())  # type: ignore[arg-type]
            mean_qr = float(sub['quality_retained'].mean())  # type: ignore[arg-type]
            print(f"  {method}: mean_compression={mean_cr:.3f}, mean_quality_retained={mean_qr:.3f}")
            stats.append({
                'rq': 'RQ8',
                'test': 'mean_compression',
                'statistic': round(mean_cr, 4),
                'p_value': None,
                'effect_size': round(mean_qr, 4),
                'effect_metric': 'mean_quality_retained',
                'notes': f'method={method}, n={len(sub)}',
            })
    else:
        print("  WARNING: optimizer results DataFrame missing 'method' or 'quality_retained' columns")
        stats.append({
            'rq': 'RQ8',
            'test': 'data_check',
            'statistic': None,
            'p_value': None,
            'effect_size': None,
            'effect_metric': None,
            'notes': 'missing required columns',
        })

    # ── Figure 9: compression ratio vs quality retention scatter ─────────────
    fig9, ax9 = plt.subplots(figsize=(9, 6))

    if 'method' in df.columns and 'compression_ratio' in df.columns and 'quality_retained' in df.columns:
        palette = sns.color_palette('tab10', len(OPTIMIZER_METHODS))
        method_color = {m: palette[i] for i, m in enumerate(OPTIMIZER_METHODS)}
        markers = {'zero_shot_verbose': 'o', 'few_shot': 's', 'cot': '^', 'original': 'D'}

        for method in OPTIMIZER_METHODS:
            sub = df[df['method'] == method]
            if len(sub) == 0:
                continue
            for orig_strat, grp in sub.groupby('strategy') if 'strategy' in sub.columns else [('all', sub)]:
                marker = markers.get(str(orig_strat), 'o')
                ax9.scatter(
                    grp['compression_ratio'],
                    grp['quality_retained'],
                    color=method_color.get(method, 'grey'),
                    marker=marker,
                    alpha=0.6,
                    s=40,
                    label=method,
                )

        # Pareto frontier: highest quality_retained for each compression_ratio bucket
        non_orig = df[df['method'] != 'original'].copy() if 'method' in df.columns else df.copy()
        if len(non_orig) > 1:
            pareto_pts = non_orig.sort_values('compression_ratio')[  # type: ignore[call-overload]
                ['compression_ratio', 'quality_retained']
            ].values
            # Keep only points not dominated (higher compression AND higher quality)
            pareto_front: list[tuple[float, float]] = []
            best_qr = -float('inf')
            for cr, qr in sorted(pareto_pts, key=lambda p: p[0]):
                if qr > best_qr:
                    pareto_front.append((float(cr), float(qr)))
                    best_qr = qr
            if len(pareto_front) > 1:
                px, py = zip(*pareto_front)
                ax9.step(px, py, where='post', color='black', linewidth=1.5,
                         linestyle='--', label='Pareto frontier', zorder=5)

        # Deduplicate legend
        handles, labels = ax9.get_legend_handles_labels()
        seen: dict[str, bool] = {}
        uniq_h, uniq_l = [], []
        for h, lbl in zip(handles, labels):
            if lbl not in seen:
                seen[lbl] = True
                uniq_h.append(h)
                uniq_l.append(lbl)
        ax9.legend(uniq_h, uniq_l, title='Method', bbox_to_anchor=(1.01, 1),
                   loc='upper left', fontsize=8)
    else:
        ax9.text(0.5, 0.5, 'No optimizer data available',
                 ha='center', va='center', transform=ax9.transAxes)

    ax9.axhline(1.0, color='grey', linestyle=':', linewidth=0.8, alpha=0.7)
    ax9.set_xlabel('Compression Ratio (original_tokens / compressed_tokens)')
    ax9.set_ylabel('Quality Retained (compressed / original)')
    ax9.set_title(
        'Figure 9: Compression Ratio vs. Quality Retention\n'
        '(right + up = better; Pareto frontier = ideal trade-off)',
        fontsize=12,
    )
    fig9.tight_layout()

    # ── Figure 10: mean compression ratio bar chart ───────────────────────────
    fig10, ax10 = plt.subplots(figsize=(10, 5))

    if 'method' in df.columns and 'strategy' in df.columns and 'compression_ratio' in df.columns:
        non_orig10 = df[df['method'] != 'original'].copy()
        if len(non_orig10) > 0:
            agg10: pd.DataFrame = (  # type: ignore[assignment]
                non_orig10.groupby(['strategy', 'method'])['compression_ratio']
                .mean()
                .reset_index()  # type: ignore[union-attr]
            )
            orig_strategies = sorted(agg10['strategy'].unique())
            methods_present = [m for m in OPTIMIZER_METHODS if m != 'original' and m in agg10['method'].unique()]
            x_pos = range(len(orig_strategies))
            bar_width = 0.8 / max(len(methods_present), 1)
            palette10 = sns.color_palette('tab10', len(OPTIMIZER_METHODS))
            mcolor10 = {m: palette10[i] for i, m in enumerate(OPTIMIZER_METHODS)}

            for mi, method in enumerate(methods_present):
                sub = agg10[agg10['method'] == method].set_index('strategy')
                heights = [
                    float(sub.loc[s, 'compression_ratio']) if s in sub.index else 0.0
                    for s in orig_strategies
                ]
                offset = (mi - len(methods_present) / 2) * bar_width + bar_width / 2
                ax10.bar(
                    [xi + offset for xi in x_pos],
                    heights,
                    bar_width,
                    label=method,
                    color=mcolor10.get(method, 'grey'),
                    alpha=0.85,
                )

            ax10.set_xticks(list(x_pos))
            ax10.set_xticklabels(orig_strategies, rotation=15, ha='right')
            ax10.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            ax10.set_ylabel('Mean Compression Ratio')
            ax10.legend(title='Method', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        else:
            ax10.text(0.5, 0.5, 'No non-original data available',
                      ha='center', va='center', transform=ax10.transAxes)
    else:
        ax10.text(0.5, 0.5, 'No optimizer data available',
                  ha='center', va='center', transform=ax10.transAxes)

    ax10.set_title(
        'Figure 10: Mean Compression Ratio by Method × Original Strategy',
        fontsize=12,
    )
    fig10.tight_layout()

    return (fig9, fig10), stats


# ── Output helpers ────────────────────────────────────────────────────────────

def save_stats_csv(all_stats: list[dict], output_dir: str) -> None:
    """Write all collected stats rows to output_dir/stats_summary.csv."""
    path = Path(output_dir) / 'stats_summary.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_stats).to_csv(path, index=False)
    print(f"\nStats saved → {path}")


def save_figures(figs: list[tuple[str, Figure]], output_dir: str) -> None:
    """Save all figures as 300 DPI PNGs to output_dir/figures/."""
    figures_dir = Path(output_dir) / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    for fname, fig in figs:
        out = figures_dir / fname
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure saved → {out}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='GreenPES analysis — RQ1–RQ8 stats + figures',
    )
    parser.add_argument(
        '--input', default='results/benchmark_results.json',
        help='Path to benchmark_results.json (default: results/benchmark_results.json)',
    )
    parser.add_argument(
        '--optimizer-input', default=None,
        help='Path to optimizer_results.json for RQ8 (default: results/optimizer_results.json if exists)',
    )
    parser.add_argument(
        '--output-dir', default='results/',
        help='Directory for stats CSV and figures/ subfolder (default: results/)',
    )
    parser.add_argument(
        '--rqs', default='all',
        help='Comma-separated RQs to run, e.g. "1,2,5,6,8", or "all" (default: all)',
    )
    args = parser.parse_args()

    rqs_to_run = (
        set(range(1, 9))
        if args.rqs == 'all'
        else {int(r.strip()) for r in args.rqs.split(',')}
    )

    print(f"Loading results from: {args.input}")
    df = load_and_clean(args.input)

    all_stats: list[dict] = []
    figures: list[tuple[str, Figure]] = []

    if 1 in rqs_to_run:
        fig1, s1 = rq1_strategy_effect(df)
        figures.append(('fig1_strategy_heatmap.png', fig1))
        all_stats.extend(s1)

    if 2 in rqs_to_run:
        fig2, s2 = rq2_token_efficiency(df)
        figures.append(('fig2_token_efficiency.png', fig2))
        all_stats.extend(s2)

    if 3 in rqs_to_run:
        fig3, s3 = rq3_model_comparison(df)
        figures.append(('fig3_model_comparison.png', fig3))
        all_stats.extend(s3)

    if 4 in rqs_to_run:
        fig4, s4 = rq4_quality_tradeoff(df)
        figures.append(('fig4_quality_efficiency_scatter.png', fig4))
        all_stats.extend(s4)

    if 5 in rqs_to_run:
        fig5, s5 = rq5_strategy_transfer(df)
        figures.append(('fig5_transfer_heatmap.png', fig5))
        all_stats.extend(s5)

    if 6 in rqs_to_run:
        fig6, s6 = rq6_model_strategy_interaction(df)
        figures.append(('fig6_interaction_plot.png', fig6))
        all_stats.extend(s6)

    if 7 in rqs_to_run:
        (fig7a, fig7b), s7 = rq7_scaling_laws(df)
        figures.append(('fig7_scaling_curves.png', fig7a))
        figures.append(('fig8_saturation_points.png', fig7b))
        all_stats.extend(s7)

    if 8 in rqs_to_run:
        # Load optimizer results (separate file)
        opt_path = args.optimizer_input
        if opt_path is None:
            opt_path = 'results/optimizer_results.json'
        opt_df: pd.DataFrame
        if Path(opt_path).exists():
            print(f"\nLoading optimizer results from: {opt_path}")
            with open(opt_path) as f:
                opt_records = json.load(f)
            opt_df = pd.DataFrame([r for r in opt_records if 'error' not in r])
        else:
            print(f"\nOptimizer results not found at {opt_path} — running RQ8 with empty DataFrame")
            opt_df = pd.DataFrame()
        (fig9, fig10), s8 = rq8_optimizer_effectiveness(opt_df)
        figures.append(('fig9_compression_scatter.png', fig9))
        figures.append(('fig10_compression_bars.png', fig10))
        all_stats.extend(s8)

    save_stats_csv(all_stats, args.output_dir)
    save_figures(figures, args.output_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
