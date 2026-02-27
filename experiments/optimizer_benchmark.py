"""
Optimizer benchmark runner for GreenPES experiments.

For each (model, task, verbose_strategy):
  1. Generate prompts for N examples
  2. Score original (baseline)
  3. Run through PromptOptimizer (LLM rewriting)
  4. Run through each BaselineCompressor method
  5. Score all versions with LLM judge
  6. Record compression ratio vs. quality retention

7 models × 4 tasks × 3 verbose_strategies × 10 examples = up to 840 base runs
Each base run produces 4 records (original + 3 compressors) + 1 optimizer record.

Usage:
    python experiments/optimizer_benchmark.py                          # all models, llm_judge scoring
    python experiments/optimizer_benchmark.py --mock                   # no API calls
    python experiments/optimizer_benchmark.py --models gpt-4o-mini    # single model
    python experiments/optimizer_benchmark.py --examples 2             # quick test
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
from greenprompt.evaluators import LLMJudgeEvaluator, InstructionFollowingEvaluator
from greenprompt.llm import LLMProvider, OpenAIProvider, MockProvider
from greenprompt.optimizer import PromptOptimizer, BaselineCompressor
from experiments.prompting_strategies import generate_prompt, BENCHMARK_EXAMPLES
from experiments.benchmark import MODEL_CONFIGS, get_provider


# ── Constants ─────────────────────────────────────────────────────────────────

VERBOSE_STRATEGIES = ['zero_shot_verbose', 'few_shot', 'cot']
TASKS = ['qa', 'summarization', 'classification', 'instruction_following']

BASELINE_METHODS = ['remove_filler', 'truncate_examples', 'add_concise_suffix']


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_evaluator(
    task: str,
    judge_provider: Optional[LLMProvider],
) -> 'LLMJudgeEvaluator | None':
    """Return an LLMJudgeEvaluator if judge_provider given, else None."""
    if judge_provider is not None:
        return LLMJudgeEvaluator(judge_provider=judge_provider, task_type=task)
    return None


def _score_prompt(
    scorer: GreenPromptScorer,
    prompt: str,
    task: str,
    example: dict,
    evaluator,
) -> dict:
    """Score a single prompt. Returns a flat metrics dict (or error dict)."""
    try:
        analysis = scorer.score_prompt(
            prompt=prompt,
            task_type=task,
            ground_truth=example.get('ground_truth'),
            max_tokens=300,
            evaluator=evaluator,
        )
        return {
            'greenpes':        analysis.score.scaled_score,
            'quality':         analysis.score.quality,
            'input_tokens':    analysis.score.input_tokens,
            'output_tokens':   analysis.score.output_tokens,
            'total_tokens':    analysis.score.total_tokens,
            'latency_ms':      analysis.latency_ms,
            'task_completed':  analysis.quality_details['task_completed'],
            'response_length': len(analysis.response),
        }
    except Exception as e:
        return {'error': str(e)}


# ── Core benchmark loop ───────────────────────────────────────────────────────

def _load_existing_opt(output_path: str) -> tuple[list[dict], set[tuple]]:
    """Load existing optimizer results and return completed (model, task, strategy, example_id) set."""
    p = Path(output_path)
    if not p.exists():
        return [], set()
    try:
        with open(p) as f:
            records = json.load(f)
        # A combo is complete when ALL methods (original + baselines + optimizer) are present
        # We track completed combos as those that have 'original' method record without error
        completed: set[tuple] = set()
        for r in records:
            if 'error' not in r and r.get('method') == 'original':
                completed.add((r['model'], r['task'], r['strategy'], r['example_id']))
        print(f"  Loaded {len(records)} existing records, {len(completed)} completed combos")
        return records, completed
    except Exception as e:
        print(f"  Warning: could not load {output_path}: {e}")
        return [], set()


def _save_incremental_opt(results: list[dict], output_path: str) -> None:
    """Atomically write results to output_path via a temp file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, str(p))


def run_optimizer_benchmark(
    providers: list[tuple[str, LLMProvider]],
    tasks: list[str] = TASKS,
    verbose_strategies: list[str] = VERBOSE_STRATEGIES,
    examples_per_task: int = 10,
    output_path: str = 'results/optimizer_results.json',
    delay_between_calls: float = 1.0,
    verbose: bool = True,
    judge_provider: Optional[LLMProvider] = None,
    optimizer_provider: Optional[LLMProvider] = None,
    difficulty_filter: str = 'all',
    resume: bool = False,
) -> list[dict]:
    """
    Run the optimizer benchmark across models × tasks × verbose_strategies × examples.

    For each (model, task, strategy, example):
      - Scores the original verbose prompt
      - Runs each BaselineCompressor method and scores result
      - Runs PromptOptimizer (if optimizer_provider given) and scores result

    Args:
        providers:            List of (name, provider) tuples for evaluation.
        tasks:                Task types to run.
        verbose_strategies:   Verbose prompting strategies to compress.
        examples_per_task:    Examples per task (default 10).
        output_path:          Output JSON file.
        delay_between_calls:  Seconds between API calls.
        verbose:              Print progress.
        judge_provider:       LLMProvider for LLM-judge evaluation.
        optimizer_provider:   LLMProvider for PromptOptimizer rewriting.
                              If None, optimizer step is skipped.
        difficulty_filter:    'all', 'easy', or 'hard'.
        resume:               If True, skip already-completed combos from output_path.

    Returns:
        List of result dicts (one per method per (model, task, strategy, example)).
    """
    if resume:
        results, completed = _load_existing_opt(output_path)
    else:
        results, completed = [], set()

    for provider_name, provider in providers:
        scorer = GreenPromptScorer(provider=provider)

        # Build optimizer for this provider (uses optimizer_provider for rewriting)
        optimizer: Optional[PromptOptimizer] = None
        if optimizer_provider is not None:
            optimizer = PromptOptimizer(
                rewriter_provider=optimizer_provider,
                scorer=scorer,
                quality_floor=0.9,
                max_iterations=3,
            )

        for task in tasks:
            all_examples = BENCHMARK_EXAMPLES.get(task, [])
            if difficulty_filter != 'all':
                all_examples = [
                    e for e in all_examples
                    if e.get('difficulty', 'easy') == difficulty_filter
                ]
            examples = all_examples[:examples_per_task]

            for strategy in verbose_strategies:
                for i, example in enumerate(examples):
                    combo = (provider_name, task, strategy, i)
                    if resume and combo in completed:
                        if verbose:
                            print(
                                f'[{provider_name} | {task} | {strategy} | ex {i+1}] SKIP (already done)'
                            )
                        continue

                    if verbose:
                        print(
                            f'[{provider_name} | {task} | {strategy} | ex {i+1}]'
                        )

                    try:
                        original_prompt = generate_prompt(strategy, task, example)
                    except Exception as e:
                        print(f'    ERROR generating prompt: {e}')
                        continue

                    # Build evaluator (LLM judge or heuristic)
                    if task == 'instruction_following' and judge_provider is None:
                        evaluator = InstructionFollowingEvaluator(
                            constraints=example.get('constraints', [])
                        )
                    else:
                        evaluator = _build_evaluator(task, judge_provider)

                    base_record = {
                        'model':            provider_name,
                        'task':             task,
                        'strategy':         strategy,
                        'example_id':       i,
                        'difficulty':       example.get('difficulty', 'easy'),
                        'prompt_length':    len(original_prompt),
                        'ground_truth':     example.get('ground_truth'),
                        'constraints':      example.get('constraints'),
                        'evaluator_type':   'llm_judge' if judge_provider else 'heuristic',
                        'timestamp':        datetime.now().isoformat(),
                    }

                    # ── Original prompt (baseline) ────────────────────────
                    orig_metrics = _score_prompt(
                        scorer, original_prompt, task, example, evaluator
                    )
                    results.append({
                        **base_record,
                        'method':              'original',
                        'compressed_prompt':   original_prompt,
                        'original_tokens':     orig_metrics.get('total_tokens'),
                        'compressed_tokens':   orig_metrics.get('total_tokens'),
                        'compression_ratio':   1.0,
                        'quality_retained':    1.0,
                        **orig_metrics,
                    })
                    if verbose and 'error' not in orig_metrics:
                        print(
                            f'    [original] tokens={orig_metrics.get("total_tokens")} '
                            f'quality={orig_metrics.get("quality", 0):.2f}'
                        )

                    orig_tokens = orig_metrics.get('total_tokens', 1) or 1
                    orig_quality = orig_metrics.get('quality', 0.0) or 0.0

                    if delay_between_calls > 0:
                        time.sleep(delay_between_calls)

                    # ── Baseline compressors ──────────────────────────────
                    for method_name in BASELINE_METHODS:
                        try:
                            compress_fn = getattr(BaselineCompressor, method_name)
                            compressed = compress_fn(original_prompt)
                        except Exception as e:
                            results.append({
                                **base_record,
                                'method': method_name,
                                'error': str(e),
                            })
                            continue

                        comp_metrics = _score_prompt(
                            scorer, compressed, task, example, evaluator
                        )
                        comp_tokens = comp_metrics.get('total_tokens', orig_tokens)
                        comp_quality = comp_metrics.get('quality', 0.0)

                        compression_ratio = orig_tokens / comp_tokens if comp_tokens > 0 else 1.0
                        quality_retained = comp_quality / orig_quality if orig_quality > 0 else 1.0

                        results.append({
                            **base_record,
                            'method':              method_name,
                            'compressed_prompt':   compressed,
                            'original_tokens':     orig_tokens,
                            'compressed_tokens':   comp_tokens,
                            'compression_ratio':   round(compression_ratio, 4),
                            'quality_retained':    round(quality_retained, 4),
                            **comp_metrics,
                        })
                        if verbose and 'error' not in comp_metrics:
                            print(
                                f'    [{method_name}] ratio={compression_ratio:.2f} '
                                f'quality_retained={quality_retained:.2f}'
                            )

                        if delay_between_calls > 0:
                            time.sleep(delay_between_calls)

                    # ── LLM optimizer ─────────────────────────────────────
                    if optimizer is not None:
                        try:
                            opt_result = optimizer.optimize(
                                prompt=original_prompt,
                                task_type=task,
                                ground_truth=example.get('ground_truth'),
                                evaluator=evaluator,
                            )
                            opt_metrics = _score_prompt(
                                scorer, opt_result.optimized_prompt, task, example, evaluator
                            )
                            results.append({
                                **base_record,
                                'method':              'llm_optimizer',
                                'compressed_prompt':   opt_result.optimized_prompt,
                                'original_tokens':     opt_result.original_tokens,
                                'compressed_tokens':   opt_result.optimized_tokens,
                                'compression_ratio':   round(opt_result.compression_ratio, 4),
                                'quality_retained':    round(opt_result.quality_retained, 4),
                                'opt_iterations':      opt_result.iterations,
                                **opt_metrics,
                            })
                            if verbose and 'error' not in opt_metrics:
                                print(
                                    f'    [llm_optimizer] ratio={opt_result.compression_ratio:.2f} '
                                    f'quality_retained={opt_result.quality_retained:.2f}'
                                )
                        except Exception as e:
                            print(f'    ERROR in optimizer: {e}')
                            results.append({
                                **base_record,
                                'method': 'llm_optimizer',
                                'error': str(e),
                                'timestamp': datetime.now().isoformat(),
                            })

                        if delay_between_calls > 0:
                            time.sleep(delay_between_calls)

                    # Save after every combo (all methods for this example)
                    _save_incremental_opt(results, output_path)

    # Final save (no-op if already saved, but ensures file exists)
    _save_incremental_opt(results, output_path)

    if verbose:
        successful = [r for r in results if 'error' not in r]
        print(f'\nResults saved to {output_path}')
        print(f'Records: {len(results)} total, {len(successful)} successful')

    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='GreenPES Optimizer Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--models', default='all',
        help='Comma-separated model names, or "all" (default: all)',
    )
    parser.add_argument(
        '--tasks', default='all',
        help=f'Comma-separated task names, or "all". Options: {", ".join(TASKS)}',
    )
    parser.add_argument(
        '--examples', type=int, default=10,
        help='Examples per task (default: 10)',
    )
    parser.add_argument(
        '--delay', type=float, default=1.0,
        help='Seconds between API calls (default: 1.0)',
    )
    parser.add_argument(
        '--output', default='results/optimizer_results.json',
        help='Output JSON file (default: results/optimizer_results.json)',
    )
    parser.add_argument(
        '--mock', action='store_true',
        help='Use MockProvider for all calls (no API keys required)',
    )
    parser.add_argument(
        '--judge-model', default='gpt-4o-mini',
        help='Model to use as LLM judge (default: gpt-4o-mini)',
    )
    parser.add_argument(
        '--optimizer-model', default='gpt-4o-mini',
        help='Model to use as rewriter in PromptOptimizer (default: gpt-4o-mini)',
    )
    parser.add_argument(
        '--no-optimizer', action='store_true',
        help='Skip LLM optimizer; run baseline compressors only',
    )
    parser.add_argument(
        '--difficulty', default='all', choices=['all', 'easy', 'hard'],
        help='Filter examples by difficulty (default: all)',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from existing output file, skipping completed combos',
    )
    args = parser.parse_args()

    # ── Resolve judge provider ────────────────────────────────────────────────
    judge_provider: Optional[LLMProvider] = None
    if args.mock:
        judge_provider = MockProvider(
            response_text='{"correctness": 3, "completeness": 3, "reasoning": 3, "conciseness": 3}'
        )
    else:
        openai_key = os.environ.get('OPENAI_API_KEY')
        if not openai_key:
            print('Warning: OPENAI_API_KEY not set — using heuristic evaluator.')
        else:
            if args.judge_model in MODEL_CONFIGS:
                _, judge_provider = get_provider(args.judge_model)
            else:
                judge_provider = OpenAIProvider(api_key=openai_key, model=args.judge_model)
        print(f'Judge: {args.judge_model if judge_provider else "heuristic"}')

    # ── Resolve optimizer provider ────────────────────────────────────────────
    optimizer_provider: Optional[LLMProvider] = None
    if not args.no_optimizer:
        if args.mock:
            optimizer_provider = MockProvider(response_text='Rewritten shorter prompt.')
        else:
            openai_key = os.environ.get('OPENAI_API_KEY')
            if openai_key:
                if args.optimizer_model in MODEL_CONFIGS:
                    _, optimizer_provider = get_provider(args.optimizer_model)
                else:
                    optimizer_provider = OpenAIProvider(
                        api_key=openai_key, model=args.optimizer_model
                    )
                print(f'Optimizer rewriter: {args.optimizer_model}')
            else:
                print('Warning: OPENAI_API_KEY not set — skipping LLM optimizer.')

    # ── Resolve target providers ──────────────────────────────────────────────
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
            print('Skipping models with missing API keys:')
            for m in missing:
                print(f'  {m}')
        if not providers:
            print('No providers available. Set API keys in .env or use --mock.')
            sys.exit(1)
    else:
        providers = []
        for name in args.models.split(','):
            name = name.strip()
            try:
                providers.append(get_provider(name))
            except ValueError as e:
                print(f'Error: {e}')
                sys.exit(1)

    # ── Resolve tasks ─────────────────────────────────────────────────────────
    tasks = TASKS if args.tasks == 'all' else [t.strip() for t in args.tasks.split(',')]
    for t in tasks:
        if t not in TASKS:
            print(f"Unknown task '{t}'. Valid tasks: {', '.join(TASKS)}")
            sys.exit(1)

    total = len(providers) * len(tasks) * len(VERBOSE_STRATEGIES) * args.examples
    methods = 1 + len(BASELINE_METHODS) + (1 if optimizer_provider else 0)
    print(f'Models:      {[p[0] for p in providers]}')
    print(f'Tasks:       {tasks}')
    print(f'Examples:    {args.examples} per task')
    print(f'Methods:     {methods} (original + {len(BASELINE_METHODS)} baselines'
          f'{" + llm_optimizer" if optimizer_provider else ""})')
    print(f'Total runs:  ~{total * methods}')
    print()

    results = run_optimizer_benchmark(
        providers=providers,
        tasks=tasks,
        verbose_strategies=VERBOSE_STRATEGIES,
        examples_per_task=args.examples,
        output_path=args.output,
        delay_between_calls=args.delay,
        judge_provider=judge_provider,
        optimizer_provider=optimizer_provider,
        difficulty_filter=args.difficulty,
        resume=args.resume,
    )

    # Summary
    successful = [r for r in results if 'error' not in r]
    by_method: dict[str, list[float]] = {}
    for r in successful:
        m = r.get('method', 'unknown')
        by_method.setdefault(m, []).append(r.get('compression_ratio', 1.0))

    print(f'\n{"="*55}')
    print('OPTIMIZER BENCHMARK SUMMARY')
    print(f'{"="*55}')
    print(f'Successful: {len(successful)}/{len(results)}')
    print('\nMean compression ratio by method:')
    import statistics
    for method, ratios in sorted(by_method.items()):
        print(f'  {method:<25} {statistics.mean(ratios):.3f}  (n={len(ratios)})')
