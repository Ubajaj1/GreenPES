"""
Quality evaluators for GreenPES.

Task-specific quality assessment for prompt responses.
"""

import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from typing import Optional


class QualityEvaluator(ABC):
    """Base class for task-specific quality evaluation."""

    @abstractmethod
    def evaluate(self, response: str, ground_truth: Optional[str] = None) -> tuple[float, bool]:
        """
        Evaluate response quality.

        Returns:
            quality_score: float in [0, 1]
            task_completed: bool indicating if task was completed
        """
        pass


class QAEvaluator(QualityEvaluator):
    """Evaluate question-answering quality."""

    def evaluate(self, response: str, ground_truth: Optional[str] = None) -> tuple[float, bool]:
        response = response.strip()

        if not response:
            return 0.0, False

        if ground_truth is None:
            # Without ground truth, use heuristics
            has_content = len(response) > 5
            not_hedging = not any(
                response.lower().startswith(phrase)
                for phrase in ("i'm not sure", "i don't know", "i cannot", "i can't")
            )
            quality = 0.7 if (has_content and not_hedging) else 0.3
            return quality, has_content

        # With ground truth, use similarity
        response_lower = response.lower()
        truth_lower = ground_truth.lower()

        # Check for exact containment
        if truth_lower in response_lower:
            return 1.0, True

        # Use sequence matching for partial credit
        similarity = SequenceMatcher(None, response_lower, truth_lower).ratio()

        # Boost score if key answer appears
        words_in_truth = set(truth_lower.split())
        words_in_response = set(response_lower.split())
        overlap = len(words_in_truth & words_in_response) / len(words_in_truth) if words_in_truth else 0

        quality = max(similarity, overlap)
        completed = quality > 0.3

        return quality, completed


class SummarizationEvaluator(QualityEvaluator):
    """Evaluate summarization quality."""

    def __init__(self, target_length: int = 100, tolerance: float = 0.5):
        """
        Args:
            target_length: Target word count for summary
            tolerance: Acceptable deviation from target (0.5 = 50%)
        """
        self.target_length = target_length
        self.tolerance = tolerance

    def evaluate(self, response: str, ground_truth: Optional[str] = None) -> tuple[float, bool]:
        response = response.strip()

        if not response:
            return 0.0, False

        words = response.split()
        word_count = len(words)

        # Length appropriateness score
        if word_count == 0:
            return 0.0, False

        length_ratio = word_count / self.target_length
        # Score is 1.0 at target, decreases as we deviate
        if length_ratio <= 1:
            length_score = length_ratio
        else:
            # Penalize being too long more heavily
            length_score = max(0, 1 - (length_ratio - 1) * 0.5)

        # Coherence heuristics
        has_sentences = len(re.findall(r'[.!?]', response)) >= 1
        has_structure = word_count >= 10

        # Combine scores
        coherence_score = 0.0
        if has_sentences:
            coherence_score += 0.5
        if has_structure:
            coherence_score += 0.5

        quality = length_score * 0.6 + coherence_score * 0.4
        completed = word_count >= 10 and has_sentences

        return min(quality, 1.0), completed


def get_evaluator(task_type: str) -> QualityEvaluator:
    """Get the appropriate evaluator for a task type."""
    evaluators = {
        'qa': QAEvaluator(),
        'question_answering': QAEvaluator(),
        'summarization': SummarizationEvaluator(),
        'summary': SummarizationEvaluator(),
    }
    return evaluators.get(task_type.lower(), QAEvaluator())
