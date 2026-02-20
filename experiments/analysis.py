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
    pass  # Task 2


# ── RQ functions ──────────────────────────────────────────────────────────────

def rq1_strategy_effect(df: pd.DataFrame) -> tuple[plt.Figure, list[dict]]:
    pass  # Task 3


def rq2_token_efficiency(df: pd.DataFrame) -> tuple[plt.Figure, list[dict]]:
    pass  # Task 4


def rq3_model_comparison(df: pd.DataFrame) -> tuple[plt.Figure, list[dict]]:
    pass  # Task 5


def rq4_quality_tradeoff(df: pd.DataFrame) -> tuple[plt.Figure, list[dict]]:
    pass  # Task 6


# ── Output helpers ────────────────────────────────────────────────────────────

def save_stats_csv(all_stats: list[dict], output_dir: str) -> None:
    pass  # Task 7


def save_figures(figs: list[tuple[str, plt.Figure]], output_dir: str) -> None:
    pass  # Task 7


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    pass  # Task 7


if __name__ == '__main__':
    main()
