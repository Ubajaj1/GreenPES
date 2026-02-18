"""
GreenPES: Green Prompt Efficiency Score

Core metric calculation for measuring prompt efficiency.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptResult:
    """Result from running a prompt through an LLM."""
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    quality_score: float
    task_completed: bool
    latency_ms: Optional[float] = None


@dataclass
class GreenPESScore:
    """Computed GreenPES efficiency score."""
    raw_score: float
    scaled_score: float
    quality: float
    efficiency: float
    input_tokens: int
    output_tokens: int
    total_tokens: int


class GreenPESCalculator:
    """
    Calculate Green Prompt Efficiency Score.

    GreenPES = (Quality × Task_Completion) / (Input_Tokens + α × Output_Tokens)

    Higher scores indicate more efficient prompts.
    """

    def __init__(self, alpha: float = 1.5, scale_factor: float = 1000):
        """
        Args:
            alpha: Weight for output tokens (default 1.5, outputs cost more)
            scale_factor: Multiplier for readable scores (default 1000)
        """
        self.alpha = alpha
        self.scale_factor = scale_factor

    def calculate(self, result: PromptResult) -> GreenPESScore:
        """Calculate GreenPES for a single prompt-response pair."""
        token_cost = result.input_tokens + self.alpha * result.output_tokens

        quality_component = result.quality_score * (1.0 if result.task_completed else 0.5)

        raw_score = quality_component / token_cost if token_cost > 0 else 0
        scaled_score = raw_score * self.scale_factor

        return GreenPESScore(
            raw_score=raw_score,
            scaled_score=scaled_score,
            quality=result.quality_score,
            efficiency=1 / token_cost if token_cost > 0 else 0,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens
        )

    def compare(self, results: list[PromptResult]) -> list[tuple[PromptResult, GreenPESScore]]:
        """Compare multiple prompts, returning sorted by efficiency (best first)."""
        scored = [(r, self.calculate(r)) for r in results]
        return sorted(scored, key=lambda x: x[1].scaled_score, reverse=True)
