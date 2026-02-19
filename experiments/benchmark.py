"""
Benchmark runner for GreenPES experiments.

Runs experiments across 7 models × 4 tasks × 5 strategies.
Target: 560 experiments (7 × 4 × 5 × 4 examples per task).

Usage:
    python experiments/benchmark.py                          # all models, all tasks, 4 examples
    python experiments/benchmark.py --models gpt-4o-mini    # single model
    python experiments/benchmark.py --quick                  # 1 example, zero_shot only
    python experiments/benchmark.py --mock                   # no API calls (testing)
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greenprompt import GreenPromptScorer
from greenprompt.evaluators import InstructionFollowingEvaluator
from greenprompt.llm import (
    LLMProvider, OpenAIProvider, AnthropicProvider,
    GeminiProvider, GroqProvider, MockProvider,
)
from experiments.prompting_strategies import generate_prompt, BENCHMARK_EXAMPLES


# ── Constants ─────────────────────────────────────────────────────────────────

STRATEGIES = ['zero_shot', 'zero_shot_verbose', 'few_shot', 'cot', 'concise']
TASKS = ['qa', 'summarization', 'classification', 'instruction_following']

# All 7 models for the benchmark
MODEL_CONFIGS: dict[str, dict] = {
    'llama-3.1-8b': {
        'provider_cls': GroqProvider,
        'model':        'llama-3.1-8b-instant',
        'env_key':      'GROQ_API_KEY',
    },
    'llama-3.3-70b': {
        'provider_cls': GroqProvider,
        'model':        'llama-3.3-70b-versatile',
        'env_key':      'GROQ_API_KEY',
    },
    'qwen3-32b': {
        'provider_cls': GroqProvider,
        'model':        'qwen/qwen3-32b',
        'env_key':      'GROQ_API_KEY',
    },
    'kimi-k2': {
        'provider_cls': GroqProvider,
        'model':        'moonshotai/kimi-k2-instruct',
        'env_key':      'GROQ_API_KEY',
    },
    'gpt-4o-mini': {
        'provider_cls': OpenAIProvider,
        'model':        'gpt-4o-mini',
        'env_key':      'OPENAI_API_KEY',
    },
    'claude-haiku': {
        'provider_cls': AnthropicProvider,
        'model':        'claude-haiku-4-5-20251001',
        'env_key':      'ANTHROPIC_API_KEY',
    },
    'gemini-flash': {
        'provider_cls': GeminiProvider,
        'model':        'gemini-2.0-flash',
        'env_key':      'GOOGLE_API_KEY',
    },
}


# ── Provider factory ──────────────────────────────────────────────────────────

def get_provider(model_name: str) -> tuple[str, LLMProvider]:
    """
    Instantiate the provider for a model name.

    Reads the API key from the environment (loaded from .env).
    Raises ValueError if the required key is missing.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid options: {', '.join(MODEL_CONFIGS)}"
        )
    cfg = MODEL_CONFIGS[model_name]
    api_key = os.environ.get(cfg['env_key'])
    if not api_key:
        raise ValueError(
            f"Missing environment variable '{cfg['env_key']}' "
            f"required for model '{model_name}'"
        )
    provider = cfg['provider_cls'](api_key=api_key, model=cfg['model'])
    return model_name, provider


# ── Core benchmark loop ───────────────────────────────────────────────────────

def run_benchmark(
    providers: list[tuple[str, LLMProvider]],
    tasks: list[str] = TASKS,
    strategies: list[str] = STRATEGIES,
    examples_per_task: int = 4,
    output_path: str = 'results/benchmark_results.json',
    delay_between_calls: float = 1.0,
    verbose: bool = True,
) -> list[dict]:
    """
    Run the full benchmark across models × tasks × strategies × examples.

    Args:
        providers:            List of (name, provider) tuples.
        tasks:                Task types to run (default: all 4).
        strategies:           Prompting strategies to test (default: all 5).
        examples_per_task:    How many examples per task to use (default: 4 → 560 total).
        output_path:          JSON file for results.
        delay_between_calls:  Seconds between API calls for rate limiting.
        verbose:              Print per-run progress.

    Returns:
        List of result dicts (one per LLM call).
    """
    results = []
    total_runs = len(providers) * len(tasks) * len(strategies) * examples_per_task
    current_run = 0

    for provider_name, provider in providers:
        scorer = GreenPromptScorer(provider=provider)

        for task in tasks:
            examples = BENCHMARK_EXAMPLES.get(task, [])[:examples_per_task]

            for strategy in strategies:
                for i, example in enumerate(examples):
                    current_run += 1

                    if verbose:
                        print(
                            f"[{current_run}/{total_runs}] "
                            f"{provider_name} | {task} | {strategy} | ex {i+1}"
                        )

                    try:
                        prompt = generate_prompt(strategy, task, example)

                        # instruction_following: build evaluator with per-example constraints
                        if task == 'instruction_following':
                            evaluator = InstructionFollowingEvaluator(
                                constraints=example.get('constraints', [])
                            )
                        else:
                            evaluator = None  # scorer calls get_evaluator(task)

                        analysis = scorer.score_prompt(
                            prompt=prompt,
                            task_type=task,
                            ground_truth=example.get('ground_truth'),
                            max_tokens=300,
                            evaluator=evaluator,
                        )

                        result = {
                            'model':           provider_name,
                            'task':            task,
                            'strategy':        strategy,
                            'example_id':      i,
                            'greenpes':        analysis.score.scaled_score,
                            'quality':         analysis.score.quality,
                            'input_tokens':    analysis.score.input_tokens,
                            'output_tokens':   analysis.score.output_tokens,
                            'total_tokens':    analysis.score.total_tokens,
                            'latency_ms':      analysis.latency_ms,
                            'task_completed':  analysis.quality_details['task_completed'],
                            'prompt_length':   len(prompt),
                            'response_length': len(analysis.response),
                            'timestamp':       datetime.now().isoformat(),
                            'prompt':          prompt,
                            'response':        analysis.response,
                            'ground_truth':    example.get('ground_truth'),
                            'constraints':     example.get('constraints'),
                        }
                        results.append(result)

                        if verbose:
                            print(
                                f"    GreenPES: {analysis.score.scaled_score:.2f} | "
                                f"Quality: {analysis.score.quality:.2f} | "
                                f"Tokens: {analysis.score.total_tokens}"
                            )

                    except Exception as e:
                        print(f"    ERROR: {e}")
                        results.append({
                            'model':     provider_name,
                            'task':      task,
                            'strategy':  strategy,
                            'example_id': i,
                            'error':     str(e),
                            'timestamp': datetime.now().isoformat(),
                        })

                    if delay_between_calls > 0:
                        time.sleep(delay_between_calls)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        successful = [r for r in results if 'error' not in r]
        print(f"\nResults saved to {output_path}")
        print(f"Successful: {len(successful)}/{total_runs}")

    return results


# ── Quick sanity check ────────────────────────────────────────────────────────

def run_quick_test(provider: LLMProvider, provider_name: str = "test") -> list[dict]:
    """One example per task, zero_shot only — verifies full pipeline end-to-end."""
    print(f"Running quick test ({len(TASKS)} tasks × 1 example × zero_shot)...")
    return run_benchmark(
        providers=[(provider_name, provider)],
        tasks=TASKS,
        strategies=['zero_shot'],
        examples_per_task=1,
        output_path='results/quick_test.json',
        delay_between_calls=0.5,
    )


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    import statistics

    successful = [r for r in results if 'error' not in r]
    if not successful:
        print("No successful runs to summarise.")
        return

    scores = [r['greenpes'] for r in successful]
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Successful runs: {len(successful)}/{len(results)}")
    print(f"Mean GreenPES:   {statistics.mean(scores):.2f}")
    if len(scores) > 1:
        print(f"Std  GreenPES:   {statistics.stdev(scores):.2f}")

    for label, key in [("Model", "model"), ("Task", "task"), ("Strategy", "strategy")]:
        print(f"\nBy {label}:")
        groups = sorted(set(r[key] for r in successful))
        for g in groups:
            g_scores = [r['greenpes'] for r in successful if r[key] == g]
            print(f"  {g:<30} {statistics.mean(g_scores):.2f}  (n={len(g_scores)})")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='GreenPES Benchmark — 7 models × 4 tasks × 5 strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available models: {', '.join(MODEL_CONFIGS)}",
    )
    parser.add_argument(
        '--models', default='all',
        help='Comma-separated model names, or "all" (default: all)',
    )
    parser.add_argument(
        '--tasks', default='all',
        help=f'Comma-separated task names, or "all" (default: all). Options: {", ".join(TASKS)}',
    )
    parser.add_argument(
        '--examples', type=int, default=4,
        help='Examples per task (default: 4 → 560 total experiments)',
    )
    parser.add_argument(
        '--delay', type=float, default=1.0,
        help='Seconds between API calls for rate limiting (default: 1.0; Groq free tier = 30 RPM)',
    )
    parser.add_argument(
        '--output', default='results/benchmark_results.json',
        help='Output JSON file path (default: results/benchmark_results.json)',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test: 1 example × zero_shot × first available model',
    )
    parser.add_argument(
        '--mock', action='store_true',
        help='Use MockProvider (no API calls, for testing pipeline)',
    )
    args = parser.parse_args()

    # Resolve models
    if args.mock:
        providers = [('mock', MockProvider())]
    elif args.models == 'all':
        providers = []
        missing = []
        for name in MODEL_CONFIGS:
            try:
                providers.append(get_provider(name))
            except ValueError as e:
                missing.append(str(e))
        if missing:
            print("Skipping models with missing API keys:")
            for m in missing:
                print(f"  {m}")
        if not providers:
            print("\nNo providers available. Set API keys in .env or use --mock.")
            sys.exit(1)
    else:
        providers = []
        for name in args.models.split(','):
            name = name.strip()
            try:
                providers.append(get_provider(name))
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)

    # Resolve tasks
    tasks = TASKS if args.tasks == 'all' else [t.strip() for t in args.tasks.split(',')]
    for t in tasks:
        if t not in TASKS:
            print(f"Unknown task '{t}'. Valid tasks: {', '.join(TASKS)}")
            sys.exit(1)

    print(f"Models:   {[p[0] for p in providers]}")
    print(f"Tasks:    {tasks}")
    print(f"Examples: {args.examples} per task")
    total = len(providers) * len(tasks) * len(STRATEGIES) * args.examples
    print(f"Total experiments: {total}")
    print()

    if args.quick:
        results = run_quick_test(providers[0][1], providers[0][0])
    else:
        results = run_benchmark(
            providers=providers,
            tasks=tasks,
            strategies=STRATEGIES,
            examples_per_task=args.examples,
            output_path=args.output,
            delay_between_calls=args.delay,
        )

    print_summary(results)
