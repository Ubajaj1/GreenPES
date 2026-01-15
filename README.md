# GreenPES: Green Prompt Efficiency Score

A standardized metric and optimizer for LLM prompt efficiency, enabling sustainable and cost-effective LLM deployment.

## Project Structure

```
GreenPES/
├── greenprompt/              # Main library package
│   ├── __init__.py          # Package initialization
│   ├── metrics.py           # GreenPES metric implementation
│   ├── evaluators.py        # Task-specific quality evaluators
│   ├── llm.py              # LLM API wrappers (OpenAI, Gemini, Groq)
│   ├── scorer.py           # Main GreenPromptScorer class
│   ├── optimizer.py        # Prompt optimization logic
│   ├── cli.py              # Command-line interface
│   └── tasks.py            # Benchmark task definitions
│
├── experiments/             # Benchmarking and analysis scripts
│   ├── __init__.py
│   ├── prompting_strategies.py  # Different prompting approaches
│   ├── benchmark.py        # Main benchmark runner
│   ├── analysis.py         # Statistical analysis
│   └── cross_task.py       # Cross-task analysis
│
├── tests/                   # Unit and integration tests
│   ├── __init__.py
│   ├── test_metrics.py
│   ├── test_evaluators.py
│   ├── test_optimizer.py
│   └── test_integration.py
│
├── results/                 # Experiment results (gitignored)
│   ├── benchmark_results.json
│   ├── figures/
│   └── analysis/
│
├── data/                    # Datasets and examples (gitignored)
│   ├── raw/
│   └── processed/
│
├── docs/                    # Documentation
│   ├── guidelines.md       # Green Prompt Engineering Guidelines
│   ├── api_reference.md    # API documentation
│   └── examples/           # Usage examples
│
├── paper/                   # Research paper
│   ├── main.tex
│   ├── figures/
│   └── references.bib
│
├── .github/                 # GitHub configuration
│   └── workflows/          # CI/CD workflows
│
├── .gitignore              # Git ignore rules
├── setup.py                # Package installation script
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── greenpes_implementation_plan.md  # Detailed implementation plan
```

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from greenprompt import GreenPromptScorer
from greenprompt.llm import GeminiProvider

scorer = GreenPromptScorer(provider=GeminiProvider(api_key="..."))
result = scorer.score_prompt("What is the capital of France?")
print(f"GreenPES: {result.score.scaled_score}")
```

## CLI Usage

```bash
# Score a prompt
greenprompt score "What is the capital of France?"

# Optimize a prompt
greenprompt optimize "Could you please tell me the capital of France?"
```


