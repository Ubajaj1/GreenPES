"""
Saturation benchmark runner.

Tests how quality varies across 7 prompt-length levels for each task,
using heuristic evaluation. Supports --resume for incremental runs.

Usage:
    # Day 1: non-Groq models
    python experiments/saturation_benchmark.py \
        --models gpt-4o-mini claude-haiku gemini-flash \
        --output results/saturation_results.json --delay 1.5

    # Day 2-5: one Groq model per day
    python experiments/saturation_benchmark.py \
        --models llama-3.1-8b \
        --output results/saturation_results.json --delay 2.5 --resume
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greenprompt.llm import (
    LLMProvider, OpenAIProvider, AnthropicProvider,
    GeminiProvider, GroqProvider, MockProvider,
)
from greenprompt.evaluators import get_evaluator, InstructionFollowingEvaluator
from experiments.saturation_prompts import (
    SATURATION_TEMPLATES, format_prompt, NUM_LEVELS,
)
from experiments.prompting_strategies import BENCHMARK_EXAMPLES


TASKS = ['qa', 'summarization', 'classification', 'instruction_following']

MODEL_CONFIGS: dict[str, dict] = {
    'llama-3.1-8b':  {'provider_cls': GroqProvider,      'model': 'llama-3.1-8b-instant',          'env_key': 'GROQ_API_KEY'},
    'llama-3.3-70b': {'provider_cls': GroqProvider,      'model': 'llama-3.3-70b-versatile',        'env_key': 'GROQ_API_KEY'},
    'qwen3-32b':     {'provider_cls': GroqProvider,      'model': 'qwen/qwen3-32b',                 'env_key': 'GROQ_API_KEY'},
    'kimi-k2':       {'provider_cls': GroqProvider,      'model': 'moonshotai/kimi-k2-instruct',    'env_key': 'GROQ_API_KEY'},
    'gpt-4o-mini':   {'provider_cls': OpenAIProvider,    'model': 'gpt-4o-mini',                    'env_key': 'OPENAI_API_KEY'},
    'claude-haiku':  {'provider_cls': AnthropicProvider, 'model': 'claude-haiku-4-5-20251001',      'env_key': 'ANTHROPIC_API_KEY'},
    'gemini-flash':  {'provider_cls': GeminiProvider,    'model': 'gemini-2.0-flash',               'env_key': 'GEMINI_API_KEY'},
    'mock':          {'provider_cls': MockProvider,      'model': 'mock',                           'env_key': None},
}


def get_provider(model_name: str) -> tuple[str, LLMProvider]:
    cfg = MODEL_CONFIGS[model_name]
    if cfg['env_key']:
        api_key = os.environ.get(cfg['env_key'])
        if not api_key:
            raise ValueError(f"Missing env var '{cfg['env_key']}' for model '{model_name}'")
        provider = cfg['provider_cls'](api_key=api_key, model=cfg['model'])
    else:
        provider = cfg['provider_cls'](model=cfg['model'])
    return model_name, provider


def _load_existing(output_path: str, resume: bool) -> tuple[list[dict], set[tuple]]:
    if not resume or not os.path.exists(output_path):
        return [], set()
    try:
        with open(output_path) as f:
            existing = json.load(f)
        done = {
            (r['model'], r['task'], r['level'], r['example_id'])
            for r in existing
            if 'error' not in r
        }
        return existing, done
    except Exception:
        return [], set()


def _save(results: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    tmp = output_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, output_path)


def run_saturation_benchmark(
    providers: list[tuple[str, LLMProvider]],
    tasks: list[str] = TASKS,
    examples_per_task: int = 20,
    output_path: str = 'results/saturation_results.json',
    delay_between_calls: float = 2.0,
    resume: bool = False,
) -> list[dict]:

    results, done = _load_existing(output_path, resume)
    print(f"Resuming: {len(done)} experiments already done" if resume and done
          else "Starting fresh run")

    for model_name, provider in providers:
        for task in tasks:
            examples = BENCHMARK_EXAMPLES[task][:examples_per_task]
            templates = SATURATION_TEMPLATES[task]

            for level_idx, template in enumerate(templates):
                level = level_idx + 1  # 1-indexed

                for ex_idx, example in enumerate(examples):
                    key = (model_name, task, level, ex_idx)
                    if key in done:
                        print(f"[{model_name} | {task} | level {level} | ex {ex_idx+1}] SKIP")
                        continue

                    prompt = format_prompt(template, task, example)
                    label = f"[{model_name} | {task} | level {level} | ex {ex_idx+1}]"

                    try:
                        response = provider.generate(prompt, max_tokens=512)

                        if task == 'instruction_following':
                            evaluator = InstructionFollowingEvaluator(
                                constraints=example.get('constraints', [])
                            )
                        else:
                            evaluator = get_evaluator(task)

                        quality, completed = evaluator.evaluate(
                            response.text,
                            example.get('ground_truth'),
                        )

                        record = {
                            'model':         model_name,
                            'task':          task,
                            'level':         level,
                            'example_id':    ex_idx,
                            'prompt_tokens': response.input_tokens,
                            'output_tokens': response.output_tokens,
                            'quality':       quality,
                            'completed':     completed,
                            'timestamp':     datetime.now().isoformat(),
                        }
                        print(f"{label} quality={quality:.3f} tokens={response.input_tokens}")

                    except Exception as e:
                        record = {
                            'model':      model_name,
                            'task':       task,
                            'level':      level,
                            'example_id': ex_idx,
                            'error':      str(e),
                            'timestamp':  datetime.now().isoformat(),
                        }
                        print(f"{label} ERROR: {e}")

                    results.append(record)
                    if 'error' not in record:
                        done.add(key)
                    _save(results, output_path)

                    if delay_between_calls > 0:
                        time.sleep(delay_between_calls)

    successful = len([r for r in results if 'error' not in r])
    print(f"\nDone. {successful} successful records.")
    return results


def main():
    parser = argparse.ArgumentParser(description='Saturation benchmark runner')
    parser.add_argument('--models',   nargs='+', default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--tasks',    nargs='+', default=TASKS, choices=TASKS)
    parser.add_argument('--examples', type=int,  default=20)
    parser.add_argument('--output',   default='results/saturation_results.json')
    parser.add_argument('--delay',    type=float, default=2.0)
    parser.add_argument('--resume',   action='store_true')
    args = parser.parse_args()

    providers = [get_provider(m) for m in args.models]
    total = len(args.models) * len(args.tasks) * NUM_LEVELS * args.examples
    print(f"Models: {args.models}")
    print(f"Tasks:  {args.tasks}")
    print(f"Levels: {NUM_LEVELS} per task")
    print(f"Total:  {total} experiments\n")

    run_saturation_benchmark(
        providers=providers,
        tasks=args.tasks,
        examples_per_task=args.examples,
        output_path=args.output,
        delay_between_calls=args.delay,
        resume=args.resume,
    )


if __name__ == '__main__':
    main()
