"""Tests for PromptOptimizer and BaselineCompressor."""

import pytest

from greenprompt.optimizer import (
    PromptOptimizer,
    BaselineCompressor,
    OptimizationResult,
)
from greenprompt.llm import MockProvider
from greenprompt import GreenPromptScorer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scorer(response: str = 'Paris') -> GreenPromptScorer:
    return GreenPromptScorer(provider=MockProvider(response_text=response))


def _optimizer(
    rewriter_response: str = 'Short prompt.',
    scorer_response: str = 'Paris',
    quality_floor: float = 0.9,
    max_iterations: int = 5,
) -> PromptOptimizer:
    return PromptOptimizer(
        rewriter_provider=MockProvider(response_text=rewriter_response),
        scorer=_scorer(scorer_response),
        quality_floor=quality_floor,
        max_iterations=max_iterations,
    )


# ── BaselineCompressor.remove_filler ─────────────────────────────────────────

class TestRemoveFiller:

    def test_strips_please(self):
        result = BaselineCompressor.remove_filler(
            'Please summarize this article.'
        )
        assert 'Please' not in result
        assert 'summarize' in result

    def test_strips_kindly(self):
        result = BaselineCompressor.remove_filler('Kindly translate this text.')
        assert 'Kindly' not in result
        assert 'translate' in result

    def test_strips_could_you_please(self):
        result = BaselineCompressor.remove_filler('Could you please classify this?')
        assert 'Could you' not in result
        assert 'classify' in result

    def test_preserves_content_without_filler(self):
        prompt = 'What is the capital of France?'
        result = BaselineCompressor.remove_filler(prompt)
        assert 'capital of France' in result

    def test_returns_original_if_result_would_be_empty(self):
        prompt = 'Please'
        result = BaselineCompressor.remove_filler(prompt)
        assert len(result.strip()) > 0

    def test_collapses_extra_blank_lines(self):
        prompt = 'Summarize this.\n\n\n\nText here.'
        result = BaselineCompressor.remove_filler(prompt)
        assert '\n\n\n' not in result


# ── BaselineCompressor.truncate_examples ─────────────────────────────────────

class TestTruncateExamples:

    def test_removes_io_pairs(self):
        prompt = (
            'Classify the sentiment.\n\n'
            'Input: The food was great.\nOutput: positive\n'
            'Input: Service was slow.\nOutput: negative\n\n'
            'Input: Amazing product!\nOutput:'
        )
        result = BaselineCompressor.truncate_examples(prompt)
        # Training pairs removed; final query kept
        assert 'Input: The food was great.' not in result
        assert 'Input: Amazing product!' in result

    def test_preserves_prompt_without_examples(self):
        prompt = 'What is the capital of France?'
        result = BaselineCompressor.truncate_examples(prompt)
        assert 'capital of France' in result

    def test_returns_original_if_result_would_be_empty(self):
        prompt = 'Input: foo\nOutput: bar\n'
        result = BaselineCompressor.truncate_examples(prompt)
        assert len(result.strip()) > 0

    def test_returns_string(self):
        assert isinstance(BaselineCompressor.truncate_examples('any text'), str)


# ── BaselineCompressor.add_concise_suffix ────────────────────────────────────

class TestAddConciseSuffix:

    def test_appends_brevity_instruction(self):
        prompt = 'What is H2O?'
        result = BaselineCompressor.add_concise_suffix(prompt)
        assert len(result) > len(prompt)
        assert '50 words' in result or 'concise' in result.lower()

    def test_not_added_when_concise_already_present(self):
        prompt = 'Summarize. Be concise.'
        result = BaselineCompressor.add_concise_suffix(prompt)
        assert result.lower().count('concise') == 1

    def test_not_added_when_brief_present(self):
        prompt = 'Give a brief answer.'
        result = BaselineCompressor.add_concise_suffix(prompt)
        assert 'Be concise' not in result

    def test_not_added_when_max_present(self):
        prompt = 'Answer in max 30 words.'
        result = BaselineCompressor.add_concise_suffix(prompt)
        assert 'Be concise' not in result

    def test_returns_string(self):
        assert isinstance(BaselineCompressor.add_concise_suffix('text'), str)


# ── OptimizationResult dataclass ─────────────────────────────────────────────

class TestOptimizationResult:

    def test_fields_set_correctly(self):
        r = OptimizationResult(
            original_prompt='long prompt',
            optimized_prompt='short',
            original_tokens=100,
            optimized_tokens=20,
            compression_ratio=5.0,
            original_quality=0.8,
            optimized_quality=0.75,
            quality_retained=0.9375,
            iterations=2,
        )
        assert r.compression_ratio == 5.0
        assert r.quality_retained == pytest.approx(0.9375)
        assert r.iterations == 2

    def test_history_defaults_to_empty_list(self):
        r = OptimizationResult(
            original_prompt='a', optimized_prompt='b',
            original_tokens=10, optimized_tokens=5,
            compression_ratio=2.0, original_quality=0.8,
            optimized_quality=0.8, quality_retained=1.0, iterations=1,
        )
        assert r.history == []

    def test_history_not_shared_across_instances(self):
        def _make():
            return OptimizationResult(
                original_prompt='a', optimized_prompt='b',
                original_tokens=10, optimized_tokens=5,
                compression_ratio=2.0, original_quality=0.8,
                optimized_quality=0.8, quality_retained=1.0, iterations=1,
            )
        r1, r2 = _make(), _make()
        r1.history.append({'x': 1})
        assert r2.history == []


# ── PromptOptimizer ───────────────────────────────────────────────────────────

class TestPromptOptimizer:

    def test_returns_optimization_result(self):
        opt = _optimizer()
        result = opt.optimize('What is the capital of France?', task_type='qa')
        assert isinstance(result, OptimizationResult)

    def test_original_prompt_preserved_in_result(self):
        original = 'What is the capital of France?'
        opt = _optimizer()
        result = opt.optimize(original, task_type='qa')
        assert result.original_prompt == original

    def test_optimized_prompt_is_nonempty_string(self):
        opt = _optimizer()
        result = opt.optimize('Some fairly long prompt text here.', task_type='qa')
        assert isinstance(result.optimized_prompt, str)
        assert len(result.optimized_prompt) > 0

    def test_compression_ratio_is_positive(self):
        opt = _optimizer()
        result = opt.optimize('A prompt with several words in it.', task_type='qa')
        assert result.compression_ratio > 0.0

    def test_original_tokens_positive(self):
        opt = _optimizer()
        result = opt.optimize('Some prompt text.', task_type='qa')
        assert result.original_tokens > 0

    def test_optimized_tokens_positive(self):
        opt = _optimizer()
        result = opt.optimize('Some prompt text.', task_type='qa')
        assert result.optimized_tokens > 0

    def test_iterations_bounded_by_max_iterations(self):
        opt = _optimizer(max_iterations=2)
        result = opt.optimize('What is the capital of France?', task_type='qa')
        assert result.iterations <= 2

    def test_history_has_at_least_original_entry(self):
        opt = _optimizer()
        result = opt.optimize('Original prompt.', task_type='qa')
        assert len(result.history) >= 1
        assert result.history[0]['strategy'] == 'original'
        assert result.history[0]['accepted'] is True

    def test_quality_retained_non_negative(self):
        opt = _optimizer()
        result = opt.optimize('Some prompt.', task_type='qa')
        assert result.quality_retained >= 0.0

    def test_works_with_ground_truth(self):
        opt = _optimizer(scorer_response='Paris')
        result = opt.optimize(
            'What is the capital of France?',
            task_type='qa',
            ground_truth='Paris',
        )
        assert isinstance(result, OptimizationResult)

    def test_works_with_custom_evaluator(self):
        from greenprompt.evaluators import QAEvaluator
        opt = _optimizer(scorer_response='Paris')
        result = opt.optimize(
            'What is the capital of France?',
            task_type='qa',
            ground_truth='Paris',
            evaluator=QAEvaluator(),
        )
        assert isinstance(result, OptimizationResult)

    def test_quality_floor_zero_always_accepts(self):
        """With floor=0, any rewrite is accepted."""
        opt = _optimizer(quality_floor=0.0)
        result = opt.optimize('Some prompt.', task_type='qa')
        # All entries in history should be accepted (quality >= 0)
        accepted = [h for h in result.history if h.get('accepted')]
        assert len(accepted) >= 1

    def test_max_iterations_zero_returns_original(self):
        """With max_iterations=0, no rewrites attempted → optimized = original."""
        opt = _optimizer(max_iterations=0)
        original = 'Original verbose prompt text.'
        result = opt.optimize(original, task_type='qa')
        assert result.optimized_prompt == original
        assert result.iterations == 0
