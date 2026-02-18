"""
GreenPES: Green Prompt Efficiency Score

A framework for measuring and optimizing LLM prompt efficiency.
"""

from .metrics import GreenPESCalculator, GreenPESScore, PromptResult
from .scorer import GreenPromptScorer, PromptAnalysis
from .evaluators import QualityEvaluator, QAEvaluator, SummarizationEvaluator, get_evaluator
from .llm import (
    LLMProvider, GeminiProvider, GroqProvider, OpenAIProvider,
    AnthropicProvider, TogetherProvider, MockProvider, LLMResponse
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "GreenPromptScorer",
    "PromptAnalysis",
    # Metrics
    "GreenPESCalculator",
    "GreenPESScore",
    "PromptResult",
    # Evaluators
    "QualityEvaluator",
    "QAEvaluator",
    "SummarizationEvaluator",
    "get_evaluator",
    # LLM Providers
    "LLMProvider",
    "GeminiProvider",
    "GroqProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "TogetherProvider",
    "MockProvider",
    "LLMResponse",
]
