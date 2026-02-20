"""Tests for experiments/analysis.py."""

import json
import os
import tempfile
import pytest
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analysis import load_and_clean, REQUIRED_COLS, rq1_strategy_effect, rq2_token_efficiency, rq3_model_comparison, rq4_quality_tradeoff
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def make_synthetic_results(n_models=2, n_tasks=2, n_strategies=2, n_examples=2) -> list[dict]:
    """Build a minimal synthetic results list for testing."""
    models = ['llama-3.1-8b', 'gpt-4o-mini'][:n_models]
    tasks = ['qa', 'summarization'][:n_tasks]
    strategies = ['zero_shot', 'concise'][:n_strategies]
    records = []
    base_greenpes = 10.0
    for model in models:
        for task in tasks:
            for strategy in strategies:
                for ex in range(n_examples):
                    records.append({
                        'model': model,
                        'task': task,
                        'strategy': strategy,
                        'example_id': ex,
                        'greenpes': base_greenpes + ex,
                        'quality': 0.8 + ex * 0.05,
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
