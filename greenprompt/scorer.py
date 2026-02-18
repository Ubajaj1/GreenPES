"""
GreenPromptScorer: Main API for measuring prompt efficiency.
"""

from dataclasses import dataclass
from typing import Optional

from .metrics import GreenPESCalculator, GreenPESScore, PromptResult
from .evaluators import QualityEvaluator, get_evaluator
from .llm import LLMProvider


@dataclass
class PromptAnalysis:
    """Complete analysis of a scored prompt."""
    prompt: str
    response: str
    score: GreenPESScore
    task_type: str
    model: str
    latency_ms: float
    quality_details: dict


class GreenPromptScorer:
    """
    Measure prompt efficiency using GreenPES.

    Usage:
        from greenprompt import GreenPromptScorer
        from greenprompt.llm import GeminiProvider

        scorer = GreenPromptScorer(provider=GeminiProvider(api_key="..."))
        result = scorer.score_prompt("What is the capital of France?")
        print(f"GreenPES: {result.score.scaled_score:.2f}")
    """

    def __init__(
        self,
        provider: LLMProvider,
        calculator: Optional[GreenPESCalculator] = None,
    ):
        """
        Args:
            provider: LLM provider to use for generation
            calculator: GreenPES calculator (uses default if not provided)
        """
        self.provider = provider
        self.calculator = calculator or GreenPESCalculator()

    def score_prompt(
        self,
        prompt: str,
        task_type: str = 'qa',
        ground_truth: Optional[str] = None,
        max_tokens: int = 500,
        evaluator: Optional[QualityEvaluator] = None,
    ) -> PromptAnalysis:
        """
        Score a single prompt for efficiency.

        Args:
            prompt: The prompt to evaluate
            task_type: Type of task ('qa', 'summarization')
            ground_truth: Optional expected answer for quality comparison
            max_tokens: Maximum tokens for LLM response
            evaluator: Custom evaluator (uses default for task_type if not provided)

        Returns:
            PromptAnalysis with score and details
        """
        # Get LLM response
        llm_response = self.provider.generate(prompt, max_tokens=max_tokens)

        # Evaluate quality
        if evaluator is None:
            evaluator = get_evaluator(task_type)

        quality, completed = evaluator.evaluate(llm_response.text, ground_truth)

        # Create PromptResult
        result = PromptResult(
            prompt=prompt,
            response=llm_response.text,
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            quality_score=quality,
            task_completed=completed,
            latency_ms=llm_response.latency_ms
        )

        # Calculate GreenPES
        score = self.calculator.calculate(result)

        return PromptAnalysis(
            prompt=prompt,
            response=llm_response.text,
            score=score,
            task_type=task_type,
            model=llm_response.model,
            latency_ms=llm_response.latency_ms,
            quality_details={
                'quality_score': quality,
                'task_completed': completed,
                'ground_truth': ground_truth,
            }
        )

    def compare_prompts(
        self,
        prompts: list[str],
        task_type: str = 'qa',
        ground_truth: Optional[str] = None,
        max_tokens: int = 500,
    ) -> list[PromptAnalysis]:
        """
        Compare multiple prompts for the same task.

        Returns list sorted by GreenPES score (highest first).
        """
        results = [
            self.score_prompt(p, task_type, ground_truth, max_tokens)
            for p in prompts
        ]
        return sorted(results, key=lambda x: x.score.scaled_score, reverse=True)
