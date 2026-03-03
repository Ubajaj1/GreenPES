import json
import os
import tempfile
import pytest


def test_benchmark_produces_correct_record_count():
    """Mock run: 1 model × 1 task × 7 levels × 2 examples = 14 records."""
    from experiments.saturation_benchmark import run_saturation_benchmark
    from greenprompt.llm import MockProvider

    provider = MockProvider(model='mock')
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        out = f.name

    try:
        run_saturation_benchmark(
            providers=[('mock', provider)],
            tasks=['qa'],
            examples_per_task=2,
            output_path=out,
            delay_between_calls=0,
        )
        with open(out) as f:
            results = json.load(f)
        # 1 model × 1 task × 7 levels × 2 examples = 14
        assert len(results) == 14
        required = {'model', 'task', 'level', 'example_id', 'prompt_tokens',
                    'output_tokens', 'quality', 'completed'}
        for r in results:
            assert required.issubset(r.keys()), f"Missing fields: {required - r.keys()}"
    finally:
        os.unlink(out)


def test_benchmark_resumes_skips_completed():
    """Resume skips records already in output file."""
    from experiments.saturation_benchmark import run_saturation_benchmark
    from greenprompt.llm import MockProvider

    provider = MockProvider(model='mock')
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
        existing = [
            {'model': 'mock', 'task': 'qa', 'level': lv, 'example_id': 0,
             'prompt_tokens': 10, 'output_tokens': 5, 'quality': 0.8, 'completed': True}
            for lv in range(1, 8)
        ]
        json.dump(existing, f)
        out = f.name

    try:
        run_saturation_benchmark(
            providers=[('mock', provider)],
            tasks=['qa'],
            examples_per_task=2,
            output_path=out,
            delay_between_calls=0,
            resume=True,
        )
        with open(out) as f:
            results = json.load(f)
        # 14 total - 7 already done = 7 new + 7 existing = 14
        assert len(results) == 14
    finally:
        os.unlink(out)


def test_record_levels_are_one_to_seven():
    """Levels in output are 1..7, not 0-indexed."""
    from experiments.saturation_benchmark import run_saturation_benchmark
    from greenprompt.llm import MockProvider

    provider = MockProvider(model='mock')
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        out = f.name

    try:
        run_saturation_benchmark(
            providers=[('mock', provider)],
            tasks=['classification'],
            examples_per_task=1,
            output_path=out,
            delay_between_calls=0,
        )
        with open(out) as f:
            results = json.load(f)
        levels = sorted({r['level'] for r in results})
        assert levels == [1, 2, 3, 4, 5, 6, 7]
    finally:
        os.unlink(out)
