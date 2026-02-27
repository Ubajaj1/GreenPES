"""
PromptOptimizer: Compress prompts while maintaining quality.

Provides:
  - PromptOptimizer: LLM-based iterative prompt rewriting with quality floor
  - BaselineCompressor: Rule-based (non-LLM) compression baselines
  - OptimizationResult: Result dataclass

Usage:
    from greenprompt.optimizer import PromptOptimizer, BaselineCompressor
    from greenprompt import GreenPromptScorer
    from greenprompt.llm import OpenAIProvider

    provider = OpenAIProvider(api_key="...")
    scorer = GreenPromptScorer(provider=provider)
    optimizer = PromptOptimizer(rewriter_provider=provider, scorer=scorer)
    result = optimizer.optimize(prompt, task_type='qa', ground_truth='Paris')
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from .llm import LLMProvider
from .scorer import GreenPromptScorer
from .evaluators import QualityEvaluator


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """Result of a single prompt optimization run."""
    original_prompt: str
    optimized_prompt: str
    original_tokens: int
    optimized_tokens: int
    compression_ratio: float          # original_tokens / optimized_tokens
    original_quality: float           # quality score of original prompt
    optimized_quality: float          # quality score of optimized prompt
    quality_retained: float           # optimized_quality / original_quality
    iterations: int                   # number of rewriting attempts
    history: list[dict] = field(default_factory=list)  # per-iteration records


# ── Baseline Compressors (no LLM) ─────────────────────────────────────────────

class BaselineCompressor:
    """
    Rule-based prompt compression baselines (no LLM calls).

    All methods are static: call as BaselineCompressor.remove_filler(prompt).
    """

    # Filler phrases commonly padded into prompts
    _FILLER_PATTERNS = [
        r'Please\s+',
        r'Kindly\s+',
        r'Could you\s+(?:please\s+)?',
        r'I would like you to\s+',
        r'I want you to\s+',
        r'You are (?:a |an )?(?:helpful |expert |professional |experienced )?'
        r'(?:AI |assistant |language model )?[^\n.]*[.\n]\s*',
        r'As (?:a |an )?(?:helpful |expert |professional |experienced )?'
        r'(?:AI |assistant |language model )[^\n.]*[.\n]\s*',
        r'Act as (?:a |an )?[^\n.]*[.\n]\s*',
    ]

    _FILLER_RE = re.compile(
        '|'.join(_FILLER_PATTERNS),
        flags=re.IGNORECASE,
    )

    # Input/Output few-shot pairs (e.g., "Input: ... Output: ...")
    _IO_PAIR_RE = re.compile(
        r'(?:Input|Question|Q)\s*:\s*[^\n]+\n(?:Output|Answer|A)\s*:\s*[^\n]+\n?',
        flags=re.IGNORECASE,
    )

    # Numbered / labeled example blocks (e.g., "Example 1: ...")
    _EXAMPLE_BLOCK_RE = re.compile(
        r'(?:Example|Ex\.?|Sample)\s*\d+\s*[:\-]\s*[^\n]+(?:\n[^\n]+)*\n?',
        flags=re.IGNORECASE,
    )

    @staticmethod
    def remove_filler(prompt: str) -> str:
        """Strip common filler phrases that add tokens without changing meaning."""
        result = BaselineCompressor._FILLER_RE.sub('', prompt)
        result = re.sub(r'\n{3,}', '\n\n', result).strip()
        return result if result.strip() else prompt

    @staticmethod
    def truncate_examples(prompt: str) -> str:
        """Remove few-shot example blocks (Input/Output pairs and labeled blocks)."""
        result = BaselineCompressor._IO_PAIR_RE.sub('', prompt)
        result = BaselineCompressor._EXAMPLE_BLOCK_RE.sub('', result)
        result = re.sub(r'\n{3,}', '\n\n', result).strip()
        return result if result.strip() else prompt

    @staticmethod
    def add_concise_suffix(prompt: str) -> str:
        """Append a brevity instruction if none already present."""
        lower = prompt.lower()
        if 'concise' in lower or 'brief' in lower or 'max ' in lower:
            return prompt
        return prompt.rstrip() + '\nBe concise (max 50 words).'


# ── Rewriting templates ───────────────────────────────────────────────────────

_REWRITE_TEMPLATES: dict[str, str] = {
    'compress': (
        'Rewrite the following prompt to be shorter while preserving its full meaning. '
        'Remove redundancy, filler words, and verbose phrasing. '
        'Return ONLY the rewritten prompt, nothing else.\n\n'
        'PROMPT:\n{prompt}'
    ),
    'extract_core': (
        'Extract only the core instruction from the following prompt. '
        'Remove all examples, verbose context, and role descriptions. '
        'Keep only what is essential to specify the task. '
        'Return ONLY the rewritten prompt, nothing else.\n\n'
        'PROMPT:\n{prompt}'
    ),
    'add_brevity': (
        'Add a conciseness constraint to the following prompt without removing any content. '
        'Append something like "Be concise." or "Answer in 1-2 sentences." '
        'Return ONLY the modified prompt, nothing else.\n\n'
        'PROMPT:\n{prompt}'
    ),
}

_REWRITING_ORDER = ['compress', 'extract_core', 'add_brevity']


# ── LLM-based Optimizer ───────────────────────────────────────────────────────

class PromptOptimizer:
    """
    Iteratively rewrite a prompt using an LLM to improve efficiency
    while maintaining a quality floor.

    Algorithm:
      1. Score original prompt → baseline quality
      2. Rewrite via LLM (compress → extract_core → add_brevity)
      3. Score rewrite
      4. Accept if quality ≥ floor × baseline_quality
      5. Cascade: feed accepted version to next rewriting strategy
      6. Return best accepted version (highest GreenPES)
    """

    def __init__(
        self,
        rewriter_provider: LLMProvider,
        scorer: GreenPromptScorer,
        quality_floor: float = 0.9,
        max_iterations: int = 5,
    ) -> None:
        """
        Args:
            rewriter_provider: LLM used to rewrite prompts.
            scorer:            GreenPromptScorer used to evaluate each candidate.
            quality_floor:     Accept rewrites whose quality ≥ floor × original
                               (default 0.9 → within 10% of original quality).
            max_iterations:    Maximum rewriting attempts (default 5).
        """
        self.rewriter_provider = rewriter_provider
        self.scorer = scorer
        self.quality_floor = quality_floor
        self.max_iterations = max_iterations

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _rewrite(self, prompt: str, strategy: str) -> str:
        """Call the rewriter LLM with the given strategy template."""
        template = _REWRITE_TEMPLATES.get(strategy, _REWRITE_TEMPLATES['compress'])
        rewrite_prompt = template.format(prompt=prompt)
        response = self.rewriter_provider.generate(rewrite_prompt, max_tokens=500)
        return response.text.strip()

    def _score(
        self,
        prompt: str,
        task_type: str,
        ground_truth: Optional[str],
        evaluator: Optional[QualityEvaluator],
    ) -> tuple[float, float, int]:
        """Return (quality, scaled_greenpes, total_tokens) for a prompt."""
        analysis = self.scorer.score_prompt(
            prompt=prompt,
            task_type=task_type,
            ground_truth=ground_truth,
            max_tokens=300,
            evaluator=evaluator,
        )
        return (
            analysis.score.quality,
            analysis.score.scaled_score,
            analysis.score.total_tokens,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def optimize(
        self,
        prompt: str,
        task_type: str = 'qa',
        ground_truth: Optional[str] = None,
        evaluator: Optional[QualityEvaluator] = None,
    ) -> OptimizationResult:
        """
        Optimize a prompt for efficiency while preserving quality.

        Args:
            prompt:       The original (verbose) prompt.
            task_type:    Task type for quality evaluation.
            ground_truth: Optional reference answer.
            evaluator:    Custom QualityEvaluator (uses task default if None).

        Returns:
            OptimizationResult with the best accepted compressed version.
        """
        # Step 1: baseline
        orig_quality, orig_greenpes, orig_tokens = self._score(
            prompt, task_type, ground_truth, evaluator
        )
        quality_threshold = self.quality_floor * orig_quality

        history: list[dict] = [{
            'iteration': 0,
            'strategy': 'original',
            'prompt': prompt,
            'quality': orig_quality,
            'greenpes': orig_greenpes,
            'tokens': orig_tokens,
            'accepted': True,
        }]

        best_prompt = prompt
        best_quality = orig_quality
        best_greenpes = orig_greenpes
        best_tokens = orig_tokens
        current_prompt = prompt
        iterations = 0

        # Steps 2-6: try rewriting strategies up to max_iterations
        for strategy in _REWRITING_ORDER:
            if iterations >= self.max_iterations:
                break
            iterations += 1

            try:
                candidate = self._rewrite(current_prompt, strategy)
            except Exception as e:
                history.append({
                    'iteration': iterations,
                    'strategy': strategy,
                    'prompt': current_prompt,
                    'quality': None,
                    'greenpes': None,
                    'tokens': None,
                    'accepted': False,
                    'error': str(e),
                })
                continue

            if not candidate or candidate == current_prompt:
                continue

            try:
                c_quality, c_greenpes, c_tokens = self._score(
                    candidate, task_type, ground_truth, evaluator
                )
            except Exception as e:
                history.append({
                    'iteration': iterations,
                    'strategy': strategy,
                    'prompt': candidate,
                    'quality': None,
                    'greenpes': None,
                    'tokens': None,
                    'accepted': False,
                    'error': str(e),
                })
                continue

            accepted = c_quality >= quality_threshold
            history.append({
                'iteration': iterations,
                'strategy': strategy,
                'prompt': candidate,
                'quality': c_quality,
                'greenpes': c_greenpes,
                'tokens': c_tokens,
                'accepted': accepted,
            })

            if accepted and c_greenpes > best_greenpes:
                best_prompt = candidate
                best_quality = c_quality
                best_greenpes = c_greenpes
                best_tokens = c_tokens
                current_prompt = candidate  # cascade: next strategy refines this

        compression_ratio = orig_tokens / best_tokens if best_tokens > 0 else 1.0
        quality_retained = best_quality / orig_quality if orig_quality > 0 else 1.0

        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=best_prompt,
            original_tokens=orig_tokens,
            optimized_tokens=best_tokens,
            compression_ratio=compression_ratio,
            original_quality=orig_quality,
            optimized_quality=best_quality,
            quality_retained=quality_retained,
            iterations=iterations,
            history=history,
        )
