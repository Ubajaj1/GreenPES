import pytest
from experiments.saturation_prompts import (
    SATURATION_TEMPLATES, TASK_INPUT_KEY, format_prompt, NUM_LEVELS
)

TASKS = ['qa', 'summarization', 'classification', 'instruction_following']

def test_all_tasks_have_seven_levels():
    for task in TASKS:
        assert len(SATURATION_TEMPLATES[task]) == NUM_LEVELS, \
            f"{task} has {len(SATURATION_TEMPLATES[task])} levels, expected {NUM_LEVELS}"

def test_all_templates_contain_placeholder():
    for task in TASKS:
        placeholder = '{' + TASK_INPUT_KEY[task] + '}'
        for i, tmpl in enumerate(SATURATION_TEMPLATES[task]):
            assert placeholder in tmpl, \
                f"{task} level {i+1} missing placeholder '{placeholder}'"

def test_format_prompt_substitutes_input():
    example = {'input': 'What is 2+2?', 'ground_truth': '4'}
    prompt = format_prompt(SATURATION_TEMPLATES['qa'][0], 'qa', example)
    assert 'What is 2+2?' in prompt
    assert '{question}' not in prompt

def test_format_prompt_all_tasks():
    examples = {
        'qa': {'input': 'What color is the sky?', 'ground_truth': 'blue'},
        'summarization': {'input': 'The sky is blue.', 'ground_truth': 'Sky is blue.'},
        'classification': {'input': 'I love this!', 'ground_truth': 'positive'},
        'instruction_following': {'input': 'Write 3 words.', 'ground_truth': 'any 3 words'},
    }
    for task, ex in examples.items():
        for tmpl in SATURATION_TEMPLATES[task]:
            prompt = format_prompt(tmpl, task, ex)
            assert ex['input'] in prompt
            placeholder = '{' + TASK_INPUT_KEY[task] + '}'
            assert placeholder not in prompt, f"Unfilled placeholder in {task} template"

def test_token_counts_increase_by_level():
    """Each level should have more tokens than the previous (roughly)."""
    import tiktoken
    enc = tiktoken.get_encoding('cl100k_base')
    example = {'input': 'placeholder input text for token counting purposes here', 'ground_truth': 'x'}
    for task in TASKS:
        prev_tokens = 0
        for i, tmpl in enumerate(SATURATION_TEMPLATES[task]):
            prompt = format_prompt(tmpl, task, example)
            tokens = len(enc.encode(prompt))
            assert tokens > prev_tokens, \
                f"{task} level {i+1} ({tokens} tokens) not > level {i} ({prev_tokens} tokens)"
            prev_tokens = tokens
