"""
Prompting strategies for GreenPES benchmarking.

Five strategies to compare efficiency across different prompting approaches.
"""

from typing import Optional


class PromptingStrategy:
    """Generate prompts using different strategies."""

    @staticmethod
    def zero_shot(task_instruction: str, input_text: str) -> str:
        """Minimal prompt, no examples. Most token-efficient."""
        return f"{task_instruction}: {input_text}"

    @staticmethod
    def zero_shot_verbose(task_instruction: str, input_text: str) -> str:
        """Detailed instructions, no examples. Higher input tokens."""
        return f"""You are an expert assistant. Your task is to {task_instruction.lower()}.
Please provide a clear, accurate, and helpful response.

Input: {input_text}

Response:"""

    @staticmethod
    def few_shot(task_instruction: str, input_text: str, examples: list[dict]) -> str:
        """Include examples before the task. Very high input tokens."""
        example_str = "\n\n".join([
            f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
            for i, ex in enumerate(examples[:3])
        ])
        return f"""{task_instruction}

{example_str}

Now complete the following:
Input: {input_text}
Output:"""

    @staticmethod
    def chain_of_thought(task_instruction: str, input_text: str) -> str:
        """Encourage step-by-step reasoning. High output tokens."""
        return f"""{task_instruction}: {input_text}

Let's think step by step:
1."""

    @staticmethod
    def concise(task_instruction: str, input_text: str, word_limit: int = 50) -> str:
        """Explicitly request brevity. Low output tokens."""
        return f"{task_instruction}. Be concise (max {word_limit} words): {input_text}"


# Predefined task configurations
TASK_CONFIGS = {
    'qa': {
        'instruction': 'Answer the following question',
        'examples': [
            {'input': 'What is the largest planet in our solar system?', 'output': 'Jupiter'},
            {'input': 'Who painted the Mona Lisa?', 'output': 'Leonardo da Vinci'},
            {'input': 'What is the chemical symbol for gold?', 'output': 'Au'},
        ]
    },
    'summarization': {
        'instruction': 'Summarize the following text',
        'examples': [
            {
                'input': 'The Amazon rainforest is the largest tropical rainforest in the world, covering over 5.5 million square kilometers. It is home to millions of species of plants and animals.',
                'output': 'The Amazon is the world\'s largest tropical rainforest, spanning 5.5 million sq km and hosting millions of species.'
            },
            {
                'input': 'Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities have been the main driver since the 1800s.',
                'output': 'Climate change involves long-term temperature and weather shifts, primarily driven by human activities since the 1800s.'
            },
        ]
    }
}


# Test examples for benchmarking
BENCHMARK_EXAMPLES = {
    'qa': [
        {'input': 'What is the capital of France?', 'ground_truth': 'Paris'},
        {'input': 'Who wrote Romeo and Juliet?', 'ground_truth': 'William Shakespeare'},
        {'input': 'What is the speed of light in km/s?', 'ground_truth': '299792'},
        {'input': 'What year did World War II end?', 'ground_truth': '1945'},
        {'input': 'What is the smallest prime number?', 'ground_truth': '2'},
    ],
    'summarization': [
        {
            'input': '''Artificial intelligence has transformed numerous industries over the past decade.
From healthcare diagnostics to autonomous vehicles, AI systems now perform tasks that once required
human expertise. Machine learning algorithms can analyze vast datasets to identify patterns invisible
to human observers, while natural language processing enables computers to understand and generate
human-like text.''',
            'ground_truth': None  # No single correct summary
        },
        {
            'input': '''The Great Barrier Reef, located off the coast of Australia, is the world's largest
coral reef system. Composed of over 2,900 individual reefs and 900 islands, it stretches for over
2,300 kilometers. The reef is home to a remarkable diversity of life, including 1,500 species of fish,
400 types of coral, and many endangered species.''',
            'ground_truth': None
        },
        {
            'input': '''Remote work has become increasingly common since 2020. Studies show that many
employees report higher productivity when working from home, though some struggle with isolation.
Companies are now adopting hybrid models that combine office and remote work to balance flexibility
with collaboration.''',
            'ground_truth': None
        },
        {
            'input': '''Renewable energy sources like solar and wind power have seen dramatic cost
reductions over the past decade. Solar panel costs have fallen by 90% since 2010, making clean
energy increasingly competitive with fossil fuels. Many countries now generate significant portions
of their electricity from renewable sources.''',
            'ground_truth': None
        },
        {
            'input': '''Sleep plays a crucial role in memory consolidation and overall health. Adults
typically need 7-9 hours of sleep per night. Chronic sleep deprivation has been linked to increased
risk of heart disease, obesity, and cognitive decline. Good sleep hygiene includes maintaining a
consistent schedule and limiting screen time before bed.''',
            'ground_truth': None
        },
    ]
}


def generate_prompt(strategy: str, task_type: str, example: dict) -> str:
    """
    Generate a prompt using the specified strategy.

    Args:
        strategy: One of 'zero_shot', 'zero_shot_verbose', 'few_shot', 'cot', 'concise'
        task_type: 'qa' or 'summarization'
        example: Dict with 'input' key (and optionally 'ground_truth')

    Returns:
        Formatted prompt string
    """
    config = TASK_CONFIGS[task_type]
    instruction = config['instruction']
    input_text = example['input']

    if strategy == 'zero_shot':
        return PromptingStrategy.zero_shot(instruction, input_text)
    elif strategy == 'zero_shot_verbose':
        return PromptingStrategy.zero_shot_verbose(instruction, input_text)
    elif strategy == 'few_shot':
        return PromptingStrategy.few_shot(instruction, input_text, config['examples'])
    elif strategy == 'cot' or strategy == 'chain_of_thought':
        return PromptingStrategy.chain_of_thought(instruction, input_text)
    elif strategy == 'concise':
        return PromptingStrategy.concise(instruction, input_text)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
