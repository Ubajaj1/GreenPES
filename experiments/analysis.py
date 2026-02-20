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
    groups = [df[df['strategy'] == s]['greenpes'].values for s in strategies]

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
    pivot = df.groupby(['strategy', 'task'])['greenpes'].mean().unstack()
    # Ensure consistent ordering
    present_strategies = [s for s in STRATEGIES if s in pivot.index]
    present_tasks = [t for t in TASKS if t in pivot.columns]
    pivot = pivot.reindex(index=present_strategies, columns=present_tasks)

    fig, ax = plt.subplots(figsize=(8, 5))
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
    pass  # Task 4


def rq3_model_comparison(df: pd.DataFrame) -> tuple[Figure, list[dict]]:
    pass  # Task 5


def rq4_quality_tradeoff(df: pd.DataFrame) -> tuple[Figure, list[dict]]:
    pass  # Task 6


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
