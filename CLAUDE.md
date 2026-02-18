# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GreenPES (Green Prompt Efficiency Score) is a Python library and CLI tool for measuring and optimizing LLM prompt efficiency. It provides a standardized metric balancing quality, token efficiency, and output conciseness.

**Core Metric Formula:**
```
GreenPES = (Quality × Task_Completion) / (Input_Tokens + α × Output_Tokens)
```
Where α = 1.5 (output tokens weighted higher due to cost). Scores are multiplied by 1000 for readability.

**Status:** Skeleton structure exists; implementation follows the detailed plan in `greenpes_implementation_plan.md`.

## Commands

```bash
# Installation (once setup.py is implemented)
pip install -e .

# CLI usage (once implemented)
greenprompt score "Your prompt here" --task summarization
greenprompt optimize "Your verbose prompt" --task code

# Testing (once tests are implemented)
pytest tests/
pytest tests/test_metrics.py -v

# Run experiments
python experiments/benchmark.py
python experiments/analysis.py
```

## Architecture

### Core Components (`greenprompt/`)

- **metrics.py** - `GreenPESCalculator` computes efficiency scores from `PromptResult` data (tokens, quality, completion)
- **evaluators.py** - Task-specific quality evaluators (`SummarizationEvaluator`, `QAEvaluator`, `CodeGenerationEvaluator`, `ClassificationEvaluator`) inheriting from abstract `QualityEvaluator`
- **llm.py** - Provider abstraction (`OpenAIProvider`, `GeminiProvider`, `GroqProvider`) with unified `LLMResponse` format
- **scorer.py** - Main API `GreenPromptScorer` orchestrates LLM calls, evaluation, and scoring
- **optimizer.py** - `PromptOptimizer` applies transformation rules (remove fluff, add brevity constraints, simplify instructions) while maintaining 90% quality floor
- **cli.py** - Click-based CLI with `score` and `optimize` commands
- **tasks.py** - Benchmark task definitions (summarization, QA, code generation, classification)

### Experiments (`experiments/`)

- **prompting_strategies.py** - 5 strategies: zero-shot, zero-shot-verbose, few-shot, chain-of-thought, concise
- **benchmark.py** - Runner for models × tasks × strategies matrix
- **analysis.py** - Statistical analysis (t-tests, confidence intervals) and visualizations
- **cross_task.py** - Cross-task strategy comparison

### Data Flow

```
User Prompt → GreenPromptScorer.score_prompt()
    → LLMProvider.generate()
    → QualityEvaluator.evaluate()
    → GreenPESCalculator.calculate()
    → PromptAnalysis result
```

## API Keys

Set via environment variables:
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY` (for Gemini)
- `GROQ_API_KEY`

Free tier APIs (Gemini 60 RPM, Groq 30 RPM) are sufficient for development and benchmarking.

## Implementation Notes

- Quality evaluators are pluggable—extend `QualityEvaluator` base class for new task types
- Token counting uses `tiktoken` for OpenAI-compatible tokenization
- Optimizer rules in `optimizer.py` are applied sequentially; quality is re-checked after each transformation
- Benchmark targets ~100 data points: 2 models × 4 tasks × 5 strategies × 3-5 examples each
