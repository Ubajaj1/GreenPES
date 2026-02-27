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
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greenprompt import GreenPromptScorer
from greenprompt.evaluators import (
    InstructionFollowingEvaluator,
    LLMJudgeEvaluator,
    get_evaluator,
    get_judge_evaluator,
)
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
        'env_key':      'GEMINI_API_KEY',
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

def _build_evaluator(
    task: str,
    example: dict,
    evaluator_type: str,
    judge_provider: Optional[LLMProvider],
) -> tuple:
    """
    Build the appropriate evaluator for a single example.

    Returns (evaluator, judge_scores_placeholder) where judge_scores_placeholder
    is None for heuristic mode.
    """
    if evaluator_type == 'llm_judge' and judge_provider is not None:
        ev = LLMJudgeEvaluator(judge_provider=judge_provider, task_type=task)
        # Wrap to capture per-dimension scores
        return ev, True
    elif task == 'instruction_following':
        return InstructionFollowingEvaluator(
            constraints=example.get('constraints', [])
        ), False
    else:
        return None, False  # scorer calls get_evaluator(task)


def _load_existing(output_path: str) -> tuple[list[dict], set[tuple]]:
    """
    Load any existing results from output_path.

    Returns (results_list, completed_keys) where completed_keys is a set of
    (model, task, strategy, example_id) tuples already finished.
    """
    if not os.path.exists(output_path):
        return [], set()
    try:
        with open(output_path) as f:
            existing = json.load(f)
        completed = {
            (r['model'], r['task'], r['strategy'], r['example_id'])
            for r in existing
            if 'model' in r and 'error' not in r
        }
        return existing, completed
    except Exception:
        return [], set()


def _save_incremental(results: list[dict], output_path: str) -> None:
    """Atomically overwrite output_path with current results list."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    tmp = output_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, output_path)


def run_benchmark(
    providers: list[tuple[str, LLMProvider]],
    tasks: list[str] = TASKS,
    strategies: list[str] = STRATEGIES,
    examples_per_task: int = 4,
    output_path: str = 'results/benchmark_results.json',
    delay_between_calls: float = 1.0,
    verbose: bool = True,
    evaluator_type: str = 'heuristic',
    judge_provider: 'Optional[LLMProvider]' = None,
    difficulty_filter: str = 'all',
    resume: bool = False,
) -> list[dict]:
    """
    Run the full benchmark across models × tasks × strategies × examples.

    Args:
        providers:            List of (name, provider) tuples.
        tasks:                Task types to run (default: all 4).
        strategies:           Prompting strategies to test (default: all 5).
        examples_per_task:    How many examples per task to use (default: 4).
        output_path:          JSON file for results (saved after every run).
        delay_between_calls:  Seconds between API calls for rate limiting.
        verbose:              Print per-run progress.
        evaluator_type:       'heuristic' or 'llm_judge'.
        judge_provider:       LLMProvider to use as judge (required if evaluator_type='llm_judge').
        difficulty_filter:    'all', 'easy', or 'hard' — filter examples by difficulty field.
        resume:               If True, skip already-completed (model, task, strategy, example_id)
                              combinations found in output_path.

    Returns:
        List of result dicts (one per LLM call).
    """
    if resume:
        results, completed = _load_existing(output_path)
        if completed:
            print(f"Resuming: {len(completed)} experiments already done, "
                  f"{len(results)} records loaded from {output_path}")
    else:
        results, completed = [], set()

    for provider_name, provider in providers:
        scorer = GreenPromptScorer(provider=provider)

        for task in tasks:
            all_examples = BENCHMARK_EXAMPLES.get(task, [])

            # Apply difficulty filter
            if difficulty_filter != 'all':
                all_examples = [
                    e for e in all_examples
                    if e.get('difficulty', 'easy') == difficulty_filter
                ]

            examples = all_examples[:examples_per_task]

            for strategy in strategies:
                for i, example in enumerate(examples):
                    # Skip if already completed (resume mode)
                    if (provider_name, task, strategy, i) in completed:
                        if verbose:
                            print(f"[{provider_name} | {task} | {strategy} | ex {i+1}] SKIP")
                        continue

                    if verbose:
                        print(
                            f"[{provider_name} | {task} | {strategy} | ex {i+1}]"
                        )

                    try:
                        prompt = generate_prompt(strategy, task, example)

                        evaluator, _is_judge = _build_evaluator(
                            task, example, evaluator_type, judge_provider
                        )

                        analysis = scorer.score_prompt(
                            prompt=prompt,
                            task_type=task,
                            ground_truth=example.get('ground_truth'),
                            max_tokens=300,
                            evaluator=evaluator,
                        )

                        judge_scores = (
                            evaluator.last_scores
                            if hasattr(evaluator, 'last_scores')
                            else None
                        )
                        result = {
                            'model':           provider_name,
                            'task':            task,
                            'strategy':        strategy,
                            'example_id':      i,
                            'difficulty':      example.get('difficulty', 'easy'),
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
                            'evaluator_type':  evaluator_type,
                            'judge_scores':    judge_scores,
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
                            'model':          provider_name,
                            'task':           task,
                            'strategy':       strategy,
                            'example_id':     i,
                            'difficulty':     example.get('difficulty', 'easy'),
                            'evaluator_type': evaluator_type,
                            'error':          str(e),
                            'timestamp':      datetime.now().isoformat(),
                        })

                    # Save after every experiment so nothing is lost on interruption
                    _save_incremental(results, output_path)

                    if delay_between_calls > 0:
                        time.sleep(delay_between_calls)

    if verbose:
        successful = [r for r in results if 'error' not in r]
        print(f"\nResults saved to {output_path}")
        print(f"Successful: {len(successful)}/{len(results)}")

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
    parser.add_argument(
        '--evaluator', default='heuristic', choices=['heuristic', 'llm_judge'],
        help='Evaluator type: heuristic (default) or llm_judge (uses judge model for scoring)',
    )
    parser.add_argument(
        '--judge-model', default='gpt-4o-mini',
        help='Model to use as LLM judge when --evaluator llm_judge (default: gpt-4o-mini)',
    )
    parser.add_argument(
        '--strategies', default='all', choices=['all', 'scaling'],
        help='Strategy set: all=original 5 strategies, scaling=parameterized scaling variants',
    )
    parser.add_argument(
        '--difficulty', default='all', choices=['all', 'easy', 'hard'],
        help='Filter examples by difficulty: all (default), easy, or hard',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from existing output file, skipping already-completed experiments',
    )
    args = parser.parse_args()

    # Resolve strategies
    if args.strategies == 'all':
        active_strategies = STRATEGIES
    else:
        # 'scaling' strategies are handled by Task 4 (added later)
        from experiments.prompting_strategies import SCALING_STRATEGIES  # type: ignore[attr-defined]
        active_strategies = SCALING_STRATEGIES

    # Resolve judge provider if llm_judge mode
    judge_provider: Optional[LLMProvider] = None
    if args.evaluator == 'llm_judge':
        if args.mock:
            # In mock mode, use a MockProvider that returns valid JSON scores
            judge_provider = MockProvider(
                response_text='{"correctness": 3, "completeness": 3, "reasoning": 3, "conciseness": 3}'
            )
            print("Judge model: mock (--mock flag set)")
        else:
            judge_model = args.judge_model
            if judge_model in MODEL_CONFIGS:
                _, judge_provider = get_provider(judge_model)
            else:
                openai_key = os.environ.get('OPENAI_API_KEY')
                if not openai_key:
                    print("Error: --evaluator llm_judge requires OPENAI_API_KEY for judge model.")
                    sys.exit(1)
                judge_provider = OpenAIProvider(api_key=openai_key, model=judge_model)
            print(f"Judge model: {judge_model}")

    # Resolve models
    if args.mock:
        providers: list[tuple[str, LLMProvider]] = [('mock', MockProvider())]
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

    print(f"Models:    {[p[0] for p in providers]}")
    print(f"Tasks:     {tasks}")
    print(f"Examples:  {args.examples} per task")
    print(f"Evaluator: {args.evaluator}")
    print(f"Difficulty:{args.difficulty}")
    total = len(providers) * len(tasks) * len(active_strategies) * args.examples
    print(f"Total experiments: {total}")
    print()

    if args.quick:
        results = run_quick_test(providers[0][1], providers[0][0])
    else:
        results = run_benchmark(
            providers=providers,
            tasks=tasks,
            strategies=active_strategies,
            examples_per_task=args.examples,
            output_path=args.output,
            delay_between_calls=args.delay,
            evaluator_type=args.evaluator,
            judge_provider=judge_provider,
            difficulty_filter=args.difficulty,
            resume=args.resume,
        )

    print_summary(results)
