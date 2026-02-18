# GreenPES: Green Prompt Efficiency Score

A standardized metric and optimizer for LLM prompt efficiency, enabling sustainable and cost-effective LLM deployment.

## Status

> **🚧 In Progress**

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


