"""
Quality evaluators for GreenPES.

Task-specific quality assessment for prompt responses.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .llm import LLMProvider


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
    """Evaluate question-answering quality via word-level overlap."""

    def evaluate(self, response: str, ground_truth: Optional[str] = None) -> tuple[float, bool]:
        response = response.strip()

        if not response:
            return 0.0, False

        if ground_truth is None:
            has_content = len(response.split()) > 3
            not_hedging = not any(
                phrase in response.lower()
                for phrase in ("i'm not sure", "i don't know", "i cannot", "i can't")
            )
            quality = 0.7 if (has_content and not_hedging) else 0.3
            return quality, has_content and not_hedging

        response_lower = response.lower()
        truth_lower = ground_truth.lower()

        # Negation check: "not [answer]" in response means incorrect
        if f"not {truth_lower}" in response_lower:
            return 0.0, False

        # Word-level overlap (ROUGE-1 recall against ground truth)
        # Strip punctuation so "Paris." matches "Paris"
        _strip = re.compile(r'[^\w]')
        words_in_truth = {_strip.sub('', w) for w in truth_lower.split()}
        words_in_response = {_strip.sub('', w) for w in response_lower.split()}

        if not words_in_truth:
            return 0.0, False

        overlap = len(words_in_truth & words_in_response) / len(words_in_truth)
        completed = overlap > 0.5

        return overlap, completed


class SummarizationEvaluator(QualityEvaluator):
    """Evaluate summarization quality using length, coherence, and optional ROUGE-1."""

    def __init__(self, target_length: int = 100):
        self.target_length = target_length

    def evaluate(self, response: str, ground_truth: Optional[str] = None) -> tuple[float, bool]:
        response = response.strip()

        if not response:
            return 0.0, False

        words = response.split()
        word_count = len(words)

        if word_count == 0:
            return 0.0, False

        # Length score: 1.0 at target, linear below, penalised above
        length_ratio = word_count / self.target_length
        if length_ratio <= 1:
            length_score = length_ratio
        else:
            length_score = max(0.0, 1.0 - (length_ratio - 1) * 0.5)

        # Coherence heuristics
        has_sentences = len(re.findall(r'[.!?]', response)) >= 1
        has_structure = word_count >= 10
        coherence_score = (0.5 if has_sentences else 0.0) + (0.5 if has_structure else 0.0)

        if ground_truth is not None:
            # ROUGE-1 recall: fraction of ground-truth words present in summary
            gt_words = set(ground_truth.lower().split())
            resp_words = set(response.lower().split())
            rouge1 = len(gt_words & resp_words) / len(gt_words) if gt_words else 0.0
            quality = length_score * 0.3 + coherence_score * 0.2 + rouge1 * 0.5
        else:
            quality = length_score * 0.6 + coherence_score * 0.4

        completed = word_count >= 10 and has_sentences
        return min(quality, 1.0), completed


class ClassificationEvaluator(QualityEvaluator):
    """Evaluate classification quality by checking if the correct label appears in the response."""

    def evaluate(self, response: str, ground_truth: Optional[str] = None) -> tuple[float, bool]:
        response = response.strip()

        if not response:
            return 0.0, False

        if ground_truth is None:
            return 0.7, True

        response_lower = response.lower()
        truth_lower = ground_truth.lower()

        # Negation check: "not [label]" means the model rejected the correct label
        if f"not {truth_lower}" in response_lower:
            return 0.0, False

        if truth_lower in response_lower:
            return 1.0, True

        return 0.0, False


# Constraint checkers for InstructionFollowingEvaluator
_CONSTRAINT_CHECKS: dict[str, Callable[[str], bool]] = {
    "bullet_points": lambda r: bool(re.search(r'^[-*•]\s', r, re.MULTILINE)),
    "numbered_list": lambda r: bool(re.search(r'^\d+[.)]\s', r, re.MULTILINE)),
    "single_word": lambda r: len(r.strip().split()) == 1,
}


class InstructionFollowingEvaluator(QualityEvaluator):
    """
    Evaluate whether a response follows structural constraints.

    Supported constraints: 'bullet_points', 'numbered_list', 'single_word'.
    Quality is the fraction of constraints satisfied; task is completed when > 50%.
    """

    def __init__(self, constraints: Optional[list[str]] = None):
        self.constraints = constraints or []

    def evaluate(self, response: str, ground_truth: Optional[str] = None) -> tuple[float, bool]:
        response = response.strip()

        if not response:
            return 0.0, False

        if not self.constraints:
            return 0.7, True

        scores = [
            1.0 if _CONSTRAINT_CHECKS[c](response) else 0.0
            for c in self.constraints
            if c in _CONSTRAINT_CHECKS
        ]

        if not scores:
            return 0.7, True

        quality = sum(scores) / len(scores)
        completed = quality > 0.5
        return quality, completed


def get_evaluator(task_type: str) -> QualityEvaluator:
    """Get the appropriate evaluator for a task type."""
    evaluators = {
        'qa': QAEvaluator(),
        'question_answering': QAEvaluator(),
        'summarization': SummarizationEvaluator(),
        'summary': SummarizationEvaluator(),
        'classification': ClassificationEvaluator(),
        'instruction_following': InstructionFollowingEvaluator(),
    }
    return evaluators.get(task_type.lower(), QAEvaluator())


class LLMJudgeEvaluator(QualityEvaluator):
    """
    LLM-as-judge evaluator using any LLMProvider (default: gpt-4o-mini).

    Scores responses on 4 dimensions (1–5 each):
      - correctness, completeness, reasoning, conciseness
    Returns normalized mean (sum / 20) in [0, 1].
    Retries once on JSON parse failure; falls back to heuristic evaluator.
    """

    _RUBRICS: dict[str, str] = {
        'qa': (
            "Focus heavily on Correctness (does the answer match the reference?). "
            "Completeness checks if the answer is fully given. "
            "Reasoning Quality assesses logical justification. "
            "Conciseness penalizes unnecessary verbosity."
        ),
        'summarization': (
            "Focus on Completeness (coverage of key points from the reference). "
            "Correctness checks factual accuracy. "
            "Conciseness rewards brevity over length. "
            "Reasoning Quality assesses coherence and structure."
        ),
        'classification': (
            "Focus on Correctness (does the classification match the reference label?). "
            "Conciseness rewards direct labeling without padding. "
            "Reasoning Quality assesses any justification given. "
            "Completeness checks if the classification is unambiguous."
        ),
        'instruction_following': (
            "Focus on Completeness (are all stated constraints satisfied?). "
            "Correctness checks format and structural adherence. "
            "Conciseness rewards appropriate brevity. "
            "Reasoning Quality assesses overall response clarity."
        ),
    }

    _JUDGE_PROMPT = """\
You are an expert judge evaluating an AI assistant's response.

Task type: {task_type}
Evaluation guidance: {rubric}
Reference answer (if available): {ground_truth}

AI response to evaluate:
{response}

Rate the response on these 4 dimensions, each from 1 (very poor) to 5 (excellent):
- correctness: Does the response accurately answer the task?
- completeness: Is the response complete and thorough?
- reasoning: Is the reasoning or logic sound and clear?
- conciseness: Is the response appropriately concise?

Respond ONLY with valid JSON in this exact format:
{{"correctness": <1-5>, "completeness": <1-5>, "reasoning": <1-5>, "conciseness": <1-5>}}"""

    def __init__(self, judge_provider: 'LLMProvider', task_type: str):
        self.judge_provider = judge_provider
        self.task_type = task_type
        self._fallback: QualityEvaluator = get_evaluator(task_type)
        self.last_scores: Optional[dict] = None  # set after each evaluate() call

    def _parse_scores(self, text: str) -> Optional[dict]:
        """Extract and validate JSON scores from judge response."""
        match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if not match:
            return None
        try:
            scores = json.loads(match.group())
        except json.JSONDecodeError:
            return None
        required = {'correctness', 'completeness', 'reasoning', 'conciseness'}
        if not required.issubset(scores.keys()):
            return None
        for k in required:
            v = scores[k]
            if not isinstance(v, (int, float)) or not (1 <= v <= 5):
                return None
        return scores

    def _call_judge(self, prompt: str) -> Optional[dict]:
        """Call judge provider and parse scores. Returns None on any failure."""
        try:
            resp = self.judge_provider.generate(prompt, max_tokens=100)
            return self._parse_scores(resp.text)
        except Exception:
            return None

    def evaluate(self, response: str, ground_truth: Optional[str] = None) -> tuple[float, bool]:
        response = response.strip()
        if not response:
            return 0.0, False

        rubric = self._RUBRICS.get(self.task_type, self._RUBRICS['qa'])
        prompt = self._JUDGE_PROMPT.format(
            task_type=self.task_type,
            rubric=rubric,
            ground_truth=ground_truth if ground_truth else "Not provided",
            response=response,
        )

        scores = self._call_judge(prompt)
        if scores is None:
            scores = self._call_judge(prompt)  # one retry

        if scores is None:
            self.last_scores = None
            return self._fallback.evaluate(response, ground_truth)

        self.last_scores = scores
        total = sum(scores[k] for k in ('correctness', 'completeness', 'reasoning', 'conciseness'))
        quality = total / 20.0
        return quality, quality >= 0.5


def get_judge_evaluator(task_type: str, judge_provider: 'LLMProvider') -> LLMJudgeEvaluator:
    """Get an LLM judge evaluator for the given task type."""
    return LLMJudgeEvaluator(judge_provider=judge_provider, task_type=task_type)
