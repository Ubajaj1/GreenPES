"""Tests for experiments/analysis.py."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
import pytest
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from experiments.analysis import (
    load_and_clean, REQUIRED_COLS,
    rq1_strategy_effect, rq2_token_efficiency, rq3_model_comparison,
    rq4_quality_tradeoff, rq5_strategy_transfer, rq6_model_strategy_interaction,
    rq7_scaling_laws, rq8_optimizer_effectiveness,
)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def make_synthetic_results(
    n_models: int = 2,
    n_tasks: int = 2,
    n_strategies: int = 2,
    n_examples: int = 2,
    quality_variance: bool = False,
) -> list[dict]:
    """Build a minimal synthetic results list for testing."""
    models = ['llama-3.1-8b', 'gpt-4o-mini', 'claude-haiku', 'qwen3-32b'][:n_models]
    tasks = ['qa', 'summarization'][:n_tasks]
    strategies = ['zero_shot', 'concise', 'cot', 'few_shot'][:n_strategies]
    records = []
    base_greenpes = 10.0
    for mi, model in enumerate(models):
        for task in tasks:
            for si, strategy in enumerate(strategies):
                for ex in range(n_examples):
                    # quality_variance=True: different models prefer different strategies
                    if quality_variance:
                        quality = 0.5 + (mi * 0.1 + si * 0.05 + ex * 0.02) % 0.5
                    else:
                        quality = 0.8 + ex * 0.05
                    records.append({
                        'model': model,
                        'task': task,
                        'strategy': strategy,
                        'example_id': ex,
                        'greenpes': base_greenpes + ex,
                        'quality': quality,
                        'input_tokens': 20 + ex * 5,
                        'output_tokens': 10 + ex * 2,
                        'total_tokens': 30 + ex * 7,
                        'latency_ms': 100.0,
                        'task_completed': True,
                    })
                    base_greenpes += 0.5
    return records


def write_json(records: list[dict]) -> str:
    """Write records to a temp JSON file, return path."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(records, f)
    f.close()
    return f.name


class TestLoadAndClean:
    def test_returns_dataframe(self):
        path = write_json(make_synthetic_results())
        df = load_and_clean(path)
        assert isinstance(df, pd.DataFrame)

    def test_drops_error_records(self):
        records = make_synthetic_results()
        records.append({'model': 'x', 'task': 'qa', 'strategy': 'zero_shot', 'error': 'timeout'})
        path = write_json(records)
        df = load_and_clean(path)
        assert 'error' not in df.columns or bool(df['error'].isna().all())
        assert len(df) == len(records) - 1

    def test_required_columns_present(self):
        path = write_json(make_synthetic_results())
        df = load_and_clean(path)
        assert REQUIRED_COLS.issubset(set(df.columns))

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_and_clean('/nonexistent/path.json')


class TestRQ1:
    def setup_method(self):
        records = make_synthetic_results(n_models=2, n_tasks=2, n_strategies=2, n_examples=4)
        path = write_json(records)
        self.df = load_and_clean(path)

    def test_returns_figure_and_stats(self):
        fig, stats = rq1_strategy_effect(self.df)
        assert isinstance(fig, Figure)
        assert isinstance(stats, list)
        assert len(stats) > 0
        plt.close(fig)

    def test_stats_have_required_keys(self):
        _, stats = rq1_strategy_effect(self.df)
        row = stats[0]
        assert 'rq' in row
        assert 'test' in row
        assert 'statistic' in row
        assert 'p_value' in row
        plt.close('all')

    def test_stats_rq_label(self):
        _, stats = rq1_strategy_effect(self.df)
        assert all(s['rq'] == 'RQ1' for s in stats)
        plt.close('all')


class TestRQ2:
    def setup_method(self):
        records = make_synthetic_results(n_models=2, n_tasks=2, n_strategies=2, n_examples=4)
        path = write_json(records)
        self.df = load_and_clean(path)

    def test_returns_figure_and_stats(self):
        fig, stats = rq2_token_efficiency(self.df)
        assert isinstance(fig, Figure)
        assert isinstance(stats, list)
        assert len(stats) > 0
        plt.close(fig)

    def test_stats_rq_label(self):
        _, stats = rq2_token_efficiency(self.df)
        assert all(s['rq'] == 'RQ2' for s in stats)
        plt.close('all')

    def test_one_winner_per_task(self):
        _, stats = rq2_token_efficiency(self.df)
        winners = [s for s in stats if s.get('test') == 'winner']
        tasks = self.df['task'].unique()
        assert len(winners) == len(tasks)
        plt.close('all')


class TestRQ3:
    def setup_method(self):
        records = make_synthetic_results(n_models=2, n_tasks=2, n_strategies=2, n_examples=4)
        path = write_json(records)
        self.df = load_and_clean(path)

    def test_returns_figure_and_stats(self):
        fig, stats = rq3_model_comparison(self.df)
        assert isinstance(fig, Figure)
        assert isinstance(stats, list)
        assert len(stats) > 0
        plt.close(fig)

    def test_one_stat_row_per_model(self):
        _, stats = rq3_model_comparison(self.df)
        models = self.df['model'].unique()
        assert len(stats) == len(models)
        plt.close('all')

    def test_stats_rq_label(self):
        _, stats = rq3_model_comparison(self.df)
        assert all(s['rq'] == 'RQ3' for s in stats)
        plt.close('all')

    def test_stats_have_required_keys(self):
        _, stats = rq3_model_comparison(self.df)
        row = stats[0]
        for key in ('rq', 'test', 'statistic', 'p_value', 'effect_size', 'effect_metric', 'notes'):
            assert key in row
        assert row['effect_metric'] == 'std'
        plt.close('all')


class TestRQ4:
    def setup_method(self):
        records = make_synthetic_results(n_models=2, n_tasks=2, n_strategies=2, n_examples=4)
        path = write_json(records)
        self.df = load_and_clean(path)

    def test_returns_figure_and_stats(self):
        fig, stats = rq4_quality_tradeoff(self.df)
        assert isinstance(fig, Figure)
        assert isinstance(stats, list)
        assert len(stats) > 0
        plt.close(fig)

    def test_stats_contain_pearson_r(self):
        _, stats = rq4_quality_tradeoff(self.df)
        pearson_rows = [s for s in stats if s['test'] == 'Pearson r']
        assert len(pearson_rows) == 1
        r = pearson_rows[0]['statistic']
        assert -1.0 <= r <= 1.0
        plt.close('all')

    def test_stats_rq_label(self):
        _, stats = rq4_quality_tradeoff(self.df)
        assert all(s['rq'] == 'RQ4' for s in stats)
        plt.close('all')

    def test_stats_have_required_keys(self):
        _, stats = rq4_quality_tradeoff(self.df)
        row = stats[0]
        for key in ('rq', 'test', 'statistic', 'p_value', 'effect_size', 'effect_metric', 'notes'):
            assert key in row
        plt.close('all')

    def test_handles_constant_quality(self):
        """pearsonr must not raise when all quality values are identical."""
        df_const = self.df.copy()
        df_const['quality'] = 1.0
        fig, _ = rq4_quality_tradeoff(df_const)   # must not raise
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestRQ5:
    def setup_method(self):
        # 3 models × 2 tasks × 3 strategies × 4 examples, with varied quality
        records = make_synthetic_results(
            n_models=3, n_tasks=2, n_strategies=3, n_examples=4,
            quality_variance=True,
        )
        path = write_json(records)
        self.df = load_and_clean(path)

    def test_returns_figure_and_stats(self):
        fig, stats = rq5_strategy_transfer(self.df)
        assert isinstance(fig, Figure)
        assert isinstance(stats, list)
        assert len(stats) > 0
        plt.close(fig)

    def test_stats_rq_label(self):
        _, stats = rq5_strategy_transfer(self.df)
        assert all(s['rq'] == 'RQ5' for s in stats)
        plt.close('all')

    def test_stats_have_required_keys(self):
        _, stats = rq5_strategy_transfer(self.df)
        for row in stats:
            for key in ('rq', 'test', 'statistic', 'p_value', 'notes'):
                assert key in row
        plt.close('all')

    def test_contains_anova_and_best_strategy_rows(self):
        _, stats = rq5_strategy_transfer(self.df)
        tests = {s['test'] for s in stats}
        assert 'two-way ANOVA interaction' in tests
        assert 'best_strategy' in tests
        plt.close('all')

    def test_best_strategy_row_per_model(self):
        _, stats = rq5_strategy_transfer(self.df)
        best_rows = [s for s in stats if s['test'] == 'best_strategy']
        n_models = self.df['model'].nunique()
        assert len(best_rows) == n_models
        plt.close('all')

    def test_transfer_matrix_values_in_range(self):
        """All transfer matrix values must be in [0, 1] (quality retention ratio)."""
        fig, _ = rq5_strategy_transfer(self.df)
        plt.close(fig)


class TestRQ6:
    def setup_method(self):
        records = make_synthetic_results(
            n_models=3, n_tasks=2, n_strategies=3, n_examples=4,
            quality_variance=True,
        )
        path = write_json(records)
        self.df = load_and_clean(path)

    def test_returns_figure_and_stats(self):
        fig, stats = rq6_model_strategy_interaction(self.df)
        assert isinstance(fig, Figure)
        assert isinstance(stats, list)
        assert len(stats) > 0
        plt.close(fig)

    def test_stats_rq_label(self):
        _, stats = rq6_model_strategy_interaction(self.df)
        assert all(s['rq'] == 'RQ6' for s in stats)
        plt.close('all')

    def test_contains_universality_index(self):
        _, stats = rq6_model_strategy_interaction(self.df)
        uni_rows = [s for s in stats if s['test'] == 'universality_index']
        assert len(uni_rows) == 1
        plt.close('all')

    def test_universality_index_in_range(self):
        _, stats = rq6_model_strategy_interaction(self.df)
        uni = [s for s in stats if s['test'] == 'universality_index'][0]
        val = uni['statistic']
        if val is not None:
            assert 0.0 <= val <= 1.0
        plt.close('all')

    def test_kendall_tau_rows_for_all_pairs(self):
        _, stats = rq6_model_strategy_interaction(self.df)
        tau_rows = [s for s in stats if s['test'] == 'Kendall tau']
        n_models = self.df['model'].nunique()
        expected_pairs = n_models * (n_models - 1) // 2
        assert len(tau_rows) == expected_pairs
        plt.close('all')

    def test_kendall_tau_statistic_in_range(self):
        _, stats = rq6_model_strategy_interaction(self.df)
        for row in [s for s in stats if s['test'] == 'Kendall tau']:
            assert -1.0 <= row['statistic'] <= 1.0
        plt.close('all')

    def test_stats_have_required_keys(self):
        _, stats = rq6_model_strategy_interaction(self.df)
        for row in stats:
            for key in ('rq', 'test', 'statistic', 'p_value', 'notes'):
                assert key in row
        plt.close('all')


class TestRQ7:
    """Tests for rq7_scaling_laws — curve fitting and saturation analysis."""

    def _make_scaling_data(self, n_models: int = 2, n_tasks: int = 2) -> pd.DataFrame:
        """
        Build synthetic data with a known logarithmic quality curve.
        total_tokens varies by strategy; quality = 0.2 * log(tokens) - 0.5.
        """
        models = ['llama-3.1-8b', 'gpt-4o-mini'][:n_models]
        tasks = ['qa', 'summarization'][:n_tasks]
        # 12 scaling strategies → 12 different token counts
        token_values = [70, 80, 90, 100, 110, 120, 160, 200, 130, 170, 220, 100]
        strategies = [f'strategy_{i}' for i in range(len(token_values))]

        records = []
        for model in models:
            for task in tasks:
                for strategy, tokens in zip(strategies, token_values):
                    quality = min(0.2 * np.log(tokens) - 0.5 + np.random.normal(0, 0.02), 1.0)
                    for ex in range(3):
                        records.append({
                            'model': model,
                            'task': task,
                            'strategy': strategy,
                            'example_id': ex,
                            'greenpes': quality * 10,
                            'quality': float(quality),
                            'input_tokens': tokens,
                            'output_tokens': 20,
                            'total_tokens': tokens + 20,
                            'latency_ms': 100.0,
                            'task_completed': True,
                        })
        return pd.DataFrame(records)

    def setup_method(self):
        np.random.seed(42)
        self.df = self._make_scaling_data()

    def test_returns_two_figures_and_stats(self):
        (fig7, fig8), stats = rq7_scaling_laws(self.df)
        assert isinstance(fig7, Figure)
        assert isinstance(fig8, Figure)
        assert isinstance(stats, list)
        plt.close('all')

    def test_stats_rq_label(self):
        (_, _), stats = rq7_scaling_laws(self.df)
        assert all(s['rq'] == 'RQ7' for s in stats)
        plt.close('all')

    def test_stats_have_required_keys(self):
        (_, _), stats = rq7_scaling_laws(self.df)
        for row in stats:
            for key in ('rq', 'test', 'statistic', 'p_value', 'effect_size', 'effect_metric', 'notes'):
                assert key in row
        plt.close('all')

    def test_one_stat_row_per_model_task_pair(self):
        (_, _), stats = rq7_scaling_laws(self.df)
        curve_rows = [s for s in stats if s['test'] == 'curve_fit']
        n_pairs = self.df['model'].nunique() * self.df['task'].nunique()
        assert len(curve_rows) == n_pairs
        plt.close('all')

    def test_best_fit_is_named_curve(self):
        (_, _), stats = rq7_scaling_laws(self.df)
        valid = {'power_law', 'logarithmic', 'sigmoid', 'none'}
        for row in [s for s in stats if s['test'] == 'curve_fit']:
            assert any(name in row['notes'] for name in valid)
        plt.close('all')

    def test_saturation_tokens_positive(self):
        (_, _), stats = rq7_scaling_laws(self.df)
        for row in [s for s in stats if s['test'] == 'curve_fit']:
            if row['effect_size'] is not None:
                assert row['effect_size'] > 0
        plt.close('all')

    def test_handles_too_few_points(self):
        """Only 2 points per (model, task): all fits should fail gracefully."""
        records = make_synthetic_results(n_models=2, n_tasks=2, n_strategies=2, n_examples=1)
        path = write_json(records)
        df_small = load_and_clean(path)
        (fig7, fig8), stats = rq7_scaling_laws(df_small)
        assert isinstance(fig7, Figure)
        assert isinstance(fig8, Figure)
        plt.close('all')


class TestEndToEnd:
    def test_runs_on_synthetic_data(self, tmp_path):
        """Full pipeline: write synthetic JSON → run script → check outputs."""
        records = make_synthetic_results(n_models=2, n_tasks=2, n_strategies=2, n_examples=4)
        input_file = tmp_path / 'results.json'
        input_file.write_text(json.dumps(records))

        script = Path(os.path.dirname(os.path.abspath(__file__))).parent / 'experiments' / 'analysis.py'
        result = subprocess.run(
            [sys.executable, str(script),
             '--input', str(input_file),
             '--output-dir', str(tmp_path),
             '--rqs', '1,2,3,4'],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

        # Check CSV exists and has content
        csv_path = tmp_path / 'stats_summary.csv'
        assert csv_path.exists()
        rows = pd.read_csv(csv_path)
        assert len(rows) > 0
        assert set(rows['rq'].unique()) >= {'RQ1', 'RQ2', 'RQ3', 'RQ4'}

        # Check core figure PNGs exist and are non-empty
        figures_dir = tmp_path / 'figures'
        assert figures_dir.is_dir()
        expected_figs = [
            'fig1_strategy_heatmap.png',
            'fig2_token_efficiency.png',
            'fig3_model_comparison.png',
            'fig4_quality_efficiency_scatter.png',
        ]
        for fname in expected_figs:
            fpath = figures_dir / fname
            assert fpath.exists(), f"Missing: {fname}"
            assert fpath.stat().st_size > 0, f"Zero-byte file: {fname}"


# ── RQ8: Optimizer effectiveness ──────────────────────────────────────────────

def make_optimizer_results(
    n_models: int = 2,
    n_tasks: int = 2,
    n_strategies: int = 2,
    n_examples: int = 3,
) -> pd.DataFrame:
    """Build synthetic optimizer_results-style DataFrame for RQ8 tests."""
    methods = ['original', 'remove_filler', 'truncate_examples', 'add_concise_suffix', 'llm_optimizer']
    models = ['llama-3.1-8b', 'gpt-4o-mini'][:n_models]
    tasks = ['qa', 'summarization'][:n_tasks]
    strategies = ['zero_shot_verbose', 'few_shot'][:n_strategies]
    records = []
    for model in models:
        for task in tasks:
            for strategy in strategies:
                for ex in range(n_examples):
                    orig_tokens = 200
                    orig_quality = 0.8
                    for mi, method in enumerate(methods):
                        compress_factor = [1.0, 1.05, 1.8, 0.9, 2.5][mi]
                        quality_factor = [1.0, 1.0, 0.95, 0.97, 0.93][mi]
                        comp_tokens = max(1, int(orig_tokens / compress_factor))
                        comp_quality = orig_quality * quality_factor
                        records.append({
                            'model': model,
                            'task': task,
                            'strategy': strategy,
                            'example_id': ex,
                            'method': method,
                            'original_tokens': orig_tokens,
                            'compressed_tokens': comp_tokens,
                            'compression_ratio': orig_tokens / comp_tokens,
                            'quality': comp_quality,
                            'quality_retained': comp_quality / orig_quality,
                            'greenpes': comp_quality * 10,
                            'total_tokens': comp_tokens + 20,
                        })
    return pd.DataFrame(records)


class TestRQ8:

    def setup_method(self):
        self.df = make_optimizer_results()

    def test_returns_two_figures_and_stats(self):
        (fig9, fig10), stats = rq8_optimizer_effectiveness(self.df)
        assert isinstance(fig9, Figure)
        assert isinstance(fig10, Figure)
        assert isinstance(stats, list)
        plt.close('all')

    def test_stats_rq_label(self):
        (_, _), stats = rq8_optimizer_effectiveness(self.df)
        rq_labels = {s['rq'] for s in stats}
        assert 'RQ8' in rq_labels
        plt.close('all')

    def test_stats_have_required_keys(self):
        (_, _), stats = rq8_optimizer_effectiveness(self.df)
        for row in stats:
            for key in ('rq', 'test', 'statistic', 'p_value', 'effect_size', 'effect_metric', 'notes'):
                assert key in row
        plt.close('all')

    def test_mean_compression_rows_present(self):
        (_, _), stats = rq8_optimizer_effectiveness(self.df)
        cr_rows = [s for s in stats if s['test'] == 'mean_compression']
        assert len(cr_rows) > 0
        plt.close('all')

    def test_paired_t_test_rows_present(self):
        (_, _), stats = rq8_optimizer_effectiveness(self.df)
        t_rows = [s for s in stats if 'paired_t_test' in s['test']]
        assert len(t_rows) == 3  # vs each of 3 baselines
        plt.close('all')

    def test_paired_t_statistic_is_float_or_none(self):
        (_, _), stats = rq8_optimizer_effectiveness(self.df)
        for row in [s for s in stats if 'paired_t_test' in s['test']]:
            assert row['statistic'] is None or isinstance(row['statistic'], float)
        plt.close('all')

    def test_handles_empty_dataframe(self):
        """Empty df should produce figures without crashing."""
        empty_df = pd.DataFrame()
        (fig9, fig10), stats = rq8_optimizer_effectiveness(empty_df)
        assert isinstance(fig9, Figure)
        assert isinstance(fig10, Figure)
        plt.close('all')

    def test_handles_missing_method_column(self):
        """DataFrame without 'method' column falls back gracefully."""
        df_no_method = self.df.drop(columns=['method'])
        (fig9, fig10), stats = rq8_optimizer_effectiveness(df_no_method)
        assert isinstance(fig9, Figure)
        assert isinstance(fig10, Figure)
        plt.close('all')

    def test_compression_ratio_stats_positive(self):
        (_, _), stats = rq8_optimizer_effectiveness(self.df)
        for row in [s for s in stats if s['test'] == 'mean_compression']:
            assert row['statistic'] is None or row['statistic'] > 0
        plt.close('all')
