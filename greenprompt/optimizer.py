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
#
# Three diverse, aggressive strategies used as candidates per iteration.
# All three are applied to the current-best prompt; the most-compressed
# candidate that passes the quality floor is accepted.

_CANDIDATE_TEMPLATES: dict[str, str] = {
    'aggressive_compress': (
        'You are a prompt compression expert. Your goal: make the prompt below as short as '
        'possible while keeping it fully answerable for a {task_type} task.\n'
        'Remove ALL of: role descriptions, filler words, politeness markers, redundant context, '
        'verbose phrasing, and any content not strictly needed to complete the task.\n'
        'Target at least 40% token reduction. Return ONLY the compressed prompt, no explanation.\n\n'
        'Original prompt:\n{prompt}'
    ),
    'minimal_form': (
        'Express the following prompt in the fewest words possible for a {task_type} task. '
        'Imagine you are sending it as a text message with a strict character limit. '
        'Keep only the essential question or instruction and any indispensable constraints. '
        'Return ONLY the minimal prompt, nothing else.\n\n'
        'Original prompt:\n{prompt}'
    ),
    'extract_core': (
        'Strip the following prompt down to its bare core instruction for a {task_type} task. '
        'Remove: all examples, role-play setup, verbose context, and decorative language. '
        'Keep: the task verb, the subject, and any hard constraints (e.g. format, length). '
        'Return ONLY the stripped prompt, nothing else.\n\n'
        'Original prompt:\n{prompt}'
    ),
}


# ── LLM-based Optimizer ───────────────────────────────────────────────────────

# Minimum fraction of tokens that must be saved for a rewrite to be considered.
_MIN_COMPRESSION = 0.05  # 5% token reduction required


class PromptOptimizer:
    """
    Iteratively compress a prompt using best-of-K LLM candidate selection.

    Algorithm per iteration:
      1. Generate K diverse rewrites of the current best prompt (in parallel
         calls to the rewriter LLM) using task-aware templates.
      2. Filter candidates that achieve at least MIN_COMPRESSION token
         reduction (estimated by word count — no extra API calls needed).
      3. Score only the most-compressed passing candidate with the full
         evaluator (target model + optional LLM judge).
      4. Accept if quality ≥ floor × original_quality.
      5. Early-stop when marginal compression gain < 1% over previous iteration
         or when no candidate passes the compression gate.
      6. Return the best accepted version.

    This is strictly stronger than single-candidate sequential rewriting:
    three diverse, task-aware strategies compete each round, and only the
    most aggressive compression that preserves quality survives.
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
            max_iterations:    Maximum rewriting rounds (default 5).
        """
        self.rewriter_provider = rewriter_provider
        self.scorer = scorer
        self.quality_floor = quality_floor
        self.max_iterations = max_iterations

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _rewrite(self, prompt: str, strategy: str, task_type: str) -> str:
        """Generate one rewrite candidate using the given strategy template."""
        template = _CANDIDATE_TEMPLATES[strategy]
        rewrite_prompt = template.format(prompt=prompt, task_type=task_type)
        response = self.rewriter_provider.generate(rewrite_prompt, max_tokens=500)
        return response.text.strip()

    def _word_count(self, text: str) -> int:
        """Fast proxy for token count used during candidate selection."""
        return len(text.split())

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
            prompt:       The original (verbose) prompt to compress.
            task_type:    Task type string; passed to task-aware rewrite templates.
            ground_truth: Optional reference answer for quality evaluation.
            evaluator:    Custom QualityEvaluator (uses task default if None).

        Returns:
            OptimizationResult with the best accepted compressed version.
        """
        # ── Baseline ──────────────────────────────────────────────────────────
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
        prev_compression = 1.0  # ratio at previous iteration (for early-stop)

        for iteration in range(1, self.max_iterations + 1):
            current_words = self._word_count(current_prompt)

            # ── Step 1: generate K candidates ─────────────────────────────────
            candidates: list[tuple[str, str, int]] = []  # (strategy, text, words)
            for strategy in _CANDIDATE_TEMPLATES:
                try:
                    c = self._rewrite(current_prompt, strategy, task_type)
                except Exception:
                    continue
                if c and c != current_prompt:
                    candidates.append((strategy, c, self._word_count(c)))

            # ── Step 2: filter by minimum compression gate ────────────────────
            min_words = int(current_words * (1 - _MIN_COMPRESSION))
            passing = [(s, c, w) for s, c, w in candidates if w <= min_words]

            if not passing:
                # No candidate achieved meaningful compression — stop early
                history.append({
                    'iteration': iteration,
                    'strategy': 'none',
                    'prompt': current_prompt,
                    'quality': None,
                    'greenpes': None,
                    'tokens': None,
                    'accepted': False,
                    'note': 'no candidate passed compression gate',
                })
                break

            # ── Step 3: score only the most-compressed candidate ──────────────
            best_strategy, best_candidate, _ = min(passing, key=lambda x: x[2])
            try:
                c_quality, c_greenpes, c_tokens = self._score(
                    best_candidate, task_type, ground_truth, evaluator
                )
            except Exception as e:
                history.append({
                    'iteration': iteration,
                    'strategy': best_strategy,
                    'prompt': best_candidate,
                    'quality': None,
                    'greenpes': None,
                    'tokens': None,
                    'accepted': False,
                    'error': str(e),
                })
                break

            accepted = c_quality >= quality_threshold

            history.append({
                'iteration': iteration,
                'strategy': best_strategy,
                'prompt': best_candidate,
                'quality': c_quality,
                'greenpes': c_greenpes,
                'tokens': c_tokens,
                'accepted': accepted,
                'n_candidates_generated': len(candidates),
                'n_candidates_passing': len(passing),
            })

            if accepted and c_greenpes > best_greenpes:
                best_prompt = best_candidate
                best_quality = c_quality
                best_greenpes = c_greenpes
                best_tokens = c_tokens
                current_prompt = best_candidate

                # ── Step 5: early-stop on marginal gain ───────────────────────
                current_ratio = orig_tokens / c_tokens if c_tokens > 0 else 1.0
                if current_ratio - prev_compression < 0.01:
                    break
                prev_compression = current_ratio

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
            iterations=len(history) - 1,
            history=history,
        )
