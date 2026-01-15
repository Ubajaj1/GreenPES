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

## Development Status

🚧 **In Development** - Following the 100-hour implementation plan

### Phase 1: Metric Design & Core Implementation (Week 1)
- [ ] Define GreenPES metric
- [ ] Implement quality evaluators
- [ ] Create LLM API wrappers
- [ ] Build main scorer class
- [ ] Define evaluation tasks

### Phase 2: Benchmarking Experiments (Week 2)
- [ ] Define prompting strategies
- [ ] Run benchmark experiments
- [ ] Statistical analysis
- [ ] Cross-task analysis

### Phase 3: Optimizer Tool (Week 3)
- [ ] Build prompt optimizer
- [ ] Create CLI tool
- [ ] Write guidelines document
- [ ] Package library

### Phase 4: Paper & Release (Week 4)
- [ ] Write research paper
- [ ] Create visualizations
- [ ] Documentation & release
- [ ] Submission prep

## Target Venue

- **Primary:** SustainNLP @ ACL/EMNLP 2026
- **Backup:** Green AI Workshop @ NeurIPS 2026

## License

MIT License (to be added)

## Citation

```bibtex
@inproceedings{greenpes2026,
  title={GreenPES: Green Prompt Efficiency Score for Sustainable LLM Deployment},
  author={TBD},
  booktitle={SustainNLP Workshop},
  year={2026}
}
```
