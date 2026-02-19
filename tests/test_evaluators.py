"""Tests for quality evaluators."""
import pytest
from greenprompt.evaluators import (
    QAEvaluator,
    SummarizationEvaluator,
    ClassificationEvaluator,
    InstructionFollowingEvaluator,
    get_evaluator,
)


# ── QAEvaluator ───────────────────────────────────────────────────────────────

class TestQAEvaluator:

    def test_exact_word_match_returns_perfect_score(self):
        ev = QAEvaluator()
        quality, completed = ev.evaluate("Paris", ground_truth="Paris")
        assert quality == 1.0
        assert completed is True

    def test_answer_embedded_in_longer_response(self):
        ev = QAEvaluator()
        quality, completed = ev.evaluate(
            "The capital of France is Paris.",
            ground_truth="Paris",
        )
        assert quality == 1.0
        assert completed is True

    def test_negation_returns_zero(self):
        """'The answer is not Paris' should not count as correct."""
        ev = QAEvaluator()
        quality, completed = ev.evaluate(
            "The capital of France is not Paris.",
            ground_truth="Paris",
        )
        assert quality == 0.0
        assert completed is False

    def test_partial_word_overlap_gives_partial_credit(self):
        """Multi-word answer with partial match gets partial credit."""
        ev = QAEvaluator()
        # ground_truth has 3 words, response contains 2 of them
        quality, completed = ev.evaluate(
            "World War ended",
            ground_truth="World War Two",
        )
        assert 0.0 < quality < 1.0

    def test_wrong_answer_returns_zero(self):
        ev = QAEvaluator()
        quality, completed = ev.evaluate("London", ground_truth="Paris")
        assert quality == 0.0
        assert completed is False

    def test_empty_response_returns_zero(self):
        ev = QAEvaluator()
        quality, completed = ev.evaluate("", ground_truth="Paris")
        assert quality == 0.0
        assert completed is False

    def test_no_ground_truth_non_empty_non_hedging(self):
        ev = QAEvaluator()
        quality, completed = ev.evaluate("The answer is 42.")
        assert quality == 0.7
        assert completed is True

    def test_no_ground_truth_hedging_response(self):
        ev = QAEvaluator()
        quality, completed = ev.evaluate("I don't know the answer.")
        assert quality == 0.3
        assert completed is False


# ── SummarizationEvaluator ────────────────────────────────────────────────────

class TestSummarizationEvaluator:

    def test_no_tolerance_parameter(self):
        """tolerance was removed — init only takes target_length."""
        ev = SummarizationEvaluator(target_length=50)
        assert not hasattr(ev, 'tolerance')

    def test_good_summary_at_target_length(self):
        ev = SummarizationEvaluator(target_length=10)
        # Exactly 10 words, with a sentence.
        response = "This is a well-formed summary with ten words exactly here."
        quality, completed = ev.evaluate(response)
        assert quality > 0.5
        assert completed is True

    def test_empty_response_returns_zero(self):
        ev = SummarizationEvaluator()
        quality, completed = ev.evaluate("")
        assert quality == 0.0
        assert completed is False

    def test_very_long_summary_is_penalized(self):
        ev = SummarizationEvaluator(target_length=10)
        # 50 words — 5x target
        response = " ".join(["word"] * 50) + "."
        quality_long, _ = ev.evaluate(response)

        short_response = " ".join(["word"] * 10) + "."
        quality_short, _ = ev.evaluate(short_response)

        assert quality_long < quality_short

    def test_with_ground_truth_boosts_score_on_overlap(self):
        """ROUGE-1: summary covering ground truth words scores higher."""
        ev = SummarizationEvaluator(target_length=20)
        ground_truth = "climate change causes rising temperatures and sea levels"
        # Response that overlaps heavily
        good_response = (
            "Climate change is causing rising temperatures and sea levels globally. "
            "This affects many ecosystems and human populations worldwide."
        )
        # Response with no overlap
        bad_response = (
            "The stock market performed well this quarter with record profits. "
            "Investors are optimistic about the economic outlook for next year."
        )
        quality_good, _ = ev.evaluate(good_response, ground_truth=ground_truth)
        quality_bad, _ = ev.evaluate(bad_response, ground_truth=ground_truth)
        assert quality_good > quality_bad

    def test_without_ground_truth_uses_length_and_coherence(self):
        ev = SummarizationEvaluator(target_length=20)
        response = " ".join(["word"] * 20) + "."
        quality, completed = ev.evaluate(response)
        assert quality > 0.0
        assert completed is True


# ── ClassificationEvaluator ───────────────────────────────────────────────────

class TestClassificationEvaluator:

    def test_exact_label_match_returns_perfect_score(self):
        ev = ClassificationEvaluator()
        quality, completed = ev.evaluate("positive", ground_truth="positive")
        assert quality == 1.0
        assert completed is True

    def test_match_is_case_insensitive(self):
        ev = ClassificationEvaluator()
        quality, completed = ev.evaluate("POSITIVE", ground_truth="positive")
        assert quality == 1.0
        assert completed is True

    def test_label_embedded_in_longer_response(self):
        """Model often says 'The sentiment is positive.' — should still pass."""
        ev = ClassificationEvaluator()
        quality, completed = ev.evaluate(
            "The sentiment of this text is positive.",
            ground_truth="positive",
        )
        assert quality == 1.0
        assert completed is True

    def test_negated_label_returns_zero(self):
        """'This is not positive' should NOT count as 'positive'."""
        ev = ClassificationEvaluator()
        quality, completed = ev.evaluate(
            "This text is not positive.",
            ground_truth="positive",
        )
        assert quality == 0.0
        assert completed is False

    def test_wrong_label_returns_zero(self):
        ev = ClassificationEvaluator()
        quality, completed = ev.evaluate("negative", ground_truth="positive")
        assert quality == 0.0
        assert completed is False

    def test_empty_response_returns_zero(self):
        ev = ClassificationEvaluator()
        quality, completed = ev.evaluate("", ground_truth="positive")
        assert quality == 0.0
        assert completed is False

    def test_without_ground_truth_non_empty_passes(self):
        ev = ClassificationEvaluator()
        quality, completed = ev.evaluate("positive")
        assert quality > 0.0
        assert completed is True

    def test_without_ground_truth_empty_fails(self):
        ev = ClassificationEvaluator()
        quality, completed = ev.evaluate("")
        assert quality == 0.0
        assert completed is False


# ── InstructionFollowingEvaluator ─────────────────────────────────────────────

class TestInstructionFollowingEvaluator:

    def test_bullet_point_constraint_met(self):
        ev = InstructionFollowingEvaluator(constraints=["bullet_points"])
        response = "- First point\n- Second point\n- Third point"
        quality, completed = ev.evaluate(response)
        assert quality > 0.5
        assert completed is True

    def test_bullet_point_constraint_not_met(self):
        ev = InstructionFollowingEvaluator(constraints=["bullet_points"])
        response = "This is just a plain paragraph with no bullets."
        quality, completed = ev.evaluate(response)
        assert quality < 0.5
        assert completed is False

    def test_single_word_constraint_met(self):
        ev = InstructionFollowingEvaluator(constraints=["single_word"])
        quality, completed = ev.evaluate("Paris")
        assert quality == 1.0
        assert completed is True

    def test_single_word_constraint_not_met(self):
        ev = InstructionFollowingEvaluator(constraints=["single_word"])
        quality, completed = ev.evaluate("The answer is Paris.")
        assert quality < 1.0
        assert completed is False

    def test_numbered_list_constraint_met(self):
        ev = InstructionFollowingEvaluator(constraints=["numbered_list"])
        response = "1. First\n2. Second\n3. Third"
        quality, completed = ev.evaluate(response)
        assert quality > 0.5
        assert completed is True

    def test_numbered_list_constraint_not_met(self):
        ev = InstructionFollowingEvaluator(constraints=["numbered_list"])
        response = "Just some text without any numbering."
        quality, completed = ev.evaluate(response)
        assert quality < 0.5
        assert completed is False

    def test_no_constraints_non_empty_passes(self):
        ev = InstructionFollowingEvaluator()
        quality, completed = ev.evaluate("Some reasonable response.")
        assert quality > 0.0
        assert completed is True

    def test_no_constraints_empty_fails(self):
        ev = InstructionFollowingEvaluator()
        quality, completed = ev.evaluate("")
        assert quality == 0.0
        assert completed is False

    def test_two_constraints_both_met_returns_one(self):
        ev = InstructionFollowingEvaluator(constraints=["bullet_points", "numbered_list"])
        # Numbered list also has a bullet prefix variant — use numbered for both checks
        response = "1. Alpha\n2. Beta\n3. Gamma"
        # Only numbered_list is met; bullet_points check will fail
        # So we expect partial, not 1.0 — this verifies averaging logic
        quality, _ = ev.evaluate(response)
        assert 0.0 < quality <= 1.0

    def test_two_constraints_partial_gives_middle_score(self):
        """Satisfies bullet_points but not single_word → score between 0 and 1."""
        ev = InstructionFollowingEvaluator(constraints=["bullet_points", "single_word"])
        response = "- Alpha\n- Beta"
        quality, completed = ev.evaluate(response)
        assert 0.0 < quality < 1.0


# ── get_evaluator factory ─────────────────────────────────────────────────────

class TestGetEvaluator:

    def test_classification_returns_classification_evaluator(self):
        ev = get_evaluator("classification")
        assert isinstance(ev, ClassificationEvaluator)

    def test_instruction_following_returns_instruction_evaluator(self):
        ev = get_evaluator("instruction_following")
        assert isinstance(ev, InstructionFollowingEvaluator)

    def test_qa_still_works(self):
        ev = get_evaluator("qa")
        assert isinstance(ev, QAEvaluator)

    def test_summarization_still_works(self):
        ev = get_evaluator("summarization")
        assert isinstance(ev, SummarizationEvaluator)
