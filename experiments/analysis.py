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

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving files
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway, pearsonr, tukey_hsd

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
    if len(agg) >= 2 and agg['mean_tokens'].std() > 0 and agg['mean_quality'].std() > 0:
        r_agg, _ = pearsonr(agg['mean_tokens'], agg['mean_quality'])  # type: ignore[call-overload]
    else:
        r_agg = float('nan')

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


# ── Output helpers ────────────────────────────────────────────────────────────

def save_stats_csv(all_stats: list[dict], output_dir: str) -> None:
    pass  # Task 7


def save_figures(figs: list[tuple[str, Figure]], output_dir: str) -> None:
    pass  # Task 7


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    pass  # Task 7


if __name__ == '__main__':
    main()
