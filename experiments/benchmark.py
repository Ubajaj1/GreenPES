"""
Benchmark runner for GreenPES experiments.

Runs experiments across models, tasks, and prompting strategies.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Load .env file if it exists
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greenprompt import GreenPromptScorer
from greenprompt.llm import LLMProvider, GeminiProvider, GroqProvider, MockProvider
from experiments.prompting_strategies import generate_prompt, BENCHMARK_EXAMPLES


STRATEGIES = ['zero_shot', 'zero_shot_verbose', 'few_shot', 'cot', 'concise']
TASKS = ['qa', 'summarization']


def run_benchmark(
    providers: list[tuple[str, LLMProvider]],
    tasks: list[str] = TASKS,
    strategies: list[str] = STRATEGIES,
    examples_per_task: int = 5,
    output_path: str = 'results/benchmark_results.json',
    delay_between_calls: float = 1.0,
    verbose: bool = True,
) -> list[dict]:
    """
    Run full benchmark across models, tasks, and strategies.

    Args:
        providers: List of (name, provider) tuples
        tasks: List of task types to benchmark
        strategies: List of prompting strategies to test
        examples_per_task: Number of examples per task
        output_path: Path to save results
        delay_between_calls: Seconds to wait between API calls (rate limiting)
        verbose: Print progress

    Returns:
        List of result dictionaries
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
                        print(f"[{current_run}/{total_runs}] {provider_name} | {task} | {strategy} | example {i+1}")

                    try:
                        # Generate prompt using strategy
                        prompt = generate_prompt(strategy, task, example)

                        # Score the prompt
                        analysis = scorer.score_prompt(
                            prompt=prompt,
                            task_type=task,
                            ground_truth=example.get('ground_truth'),
                            max_tokens=300,
                        )

                        result = {
                            'model': provider_name,
                            'task': task,
                            'strategy': strategy,
                            'example_id': i,
                            'greenpes': analysis.score.scaled_score,
                            'quality': analysis.score.quality,
                            'input_tokens': analysis.score.input_tokens,
                            'output_tokens': analysis.score.output_tokens,
                            'total_tokens': analysis.score.total_tokens,
                            'latency_ms': analysis.latency_ms,
                            'task_completed': analysis.quality_details['task_completed'],
                            'prompt_length': len(prompt),
                            'response_length': len(analysis.response),
                            'timestamp': datetime.now().isoformat(),
                            'prompt': prompt,
                            'response': analysis.response,
                            'ground_truth': example.get('ground_truth'),
                        }

                        results.append(result)

                        if verbose:
                            print(f"    GreenPES: {analysis.score.scaled_score:.2f} | "
                                  f"Quality: {analysis.score.quality:.2f} | "
                                  f"Tokens: {analysis.score.total_tokens}")

                    except Exception as e:
                        print(f"    ERROR: {e}")
                        results.append({
                            'model': provider_name,
                            'task': task,
                            'strategy': strategy,
                            'example_id': i,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat(),
                        })

                    # Rate limiting delay
                    if delay_between_calls > 0:
                        time.sleep(delay_between_calls)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\nResults saved to {output_path}")
        print(f"Total successful runs: {len([r for r in results if 'error' not in r])}/{total_runs}")

    return results


def run_quick_test(provider: LLMProvider, provider_name: str = "test") -> list[dict]:
    """Run a quick test with minimal examples to verify setup."""
    print("Running quick test (2 examples)...")
    return run_benchmark(
        providers=[(provider_name, provider)],
        tasks=['qa'],
        strategies=['zero_shot', 'concise'],
        examples_per_task=1,
        output_path='results/quick_test.json',
        delay_between_calls=0.5,
    )


if __name__ == '__main__':
    # Example usage - replace with your API keys
    import argparse

    parser = argparse.ArgumentParser(description='Run GreenPES benchmark')
    parser.add_argument('--gemini-key', type=str, help='Gemini API key')
    parser.add_argument('--groq-key', type=str, help='Groq API key')
    parser.add_argument('--provider', type=str, choices=['gemini', 'groq', 'all'],
                        default='all', help='Which provider to use (default: all)')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    parser.add_argument('--mock', action='store_true', help='Use mock provider (no API calls)')
    args = parser.parse_args()

    # Get API keys from args or environment
    gemini_key = args.gemini_key or os.environ.get('GEMINI_API_KEY')
    groq_key = args.groq_key or os.environ.get('GROQ_API_KEY')

    if args.mock:
        # Test without API calls
        print("Using mock provider...")
        providers = [('mock', MockProvider())]
    else:
        providers = []
        if groq_key and args.provider in ('groq', 'all'):
            providers.append(('llama-3.1-8b', GroqProvider(api_key=groq_key)))
        if gemini_key and args.provider in ('gemini', 'all'):
            providers.append(('gemini-flash', GeminiProvider(api_key=gemini_key)))

        if not providers:
            print("No API keys provided. Use --mock for testing, or provide keys:")
            print("  --gemini-key YOUR_KEY")
            print("  --groq-key YOUR_KEY")
            print("  Or set GEMINI_API_KEY / GROQ_API_KEY environment variables")
            print("  Use --provider groq/gemini to pick one")
            exit(1)

    if args.quick:
        # Quick test with first provider
        results = run_quick_test(providers[0][1], providers[0][0])
    else:
        # Full benchmark
        results = run_benchmark(providers)

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    successful = [r for r in results if 'error' not in r]
    if successful:
        import statistics
        scores = [r['greenpes'] for r in successful]
        print(f"Total runs: {len(successful)}")
        print(f"Mean GreenPES: {statistics.mean(scores):.2f}")
        print(f"Std GreenPES: {statistics.stdev(scores):.2f}" if len(scores) > 1 else "")

        # By strategy
        print("\nBy Strategy:")
        strategies_seen = set(r['strategy'] for r in successful)
        for strategy in strategies_seen:
            strat_scores = [r['greenpes'] for r in successful if r['strategy'] == strategy]
            print(f"  {strategy}: {statistics.mean(strat_scores):.2f} (n={len(strat_scores)})")
