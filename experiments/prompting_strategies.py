"""
Prompting strategies for GreenPES benchmarking.

Five strategies to compare efficiency across different prompting approaches.
"""

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


# ── TASK CONFIGURATIONS ───────────────────────────────────────────────────────
# instruction: used in every strategy's prompt template
# examples: 3 few-shot demonstrations (input/output pairs)

TASK_CONFIGS = {
    'qa': {
        'instruction': 'Answer the following question',
        'examples': [
            {'input': 'What is the largest planet in our solar system?', 'output': 'Jupiter'},
            {'input': 'Who painted the Mona Lisa?', 'output': 'Leonardo da Vinci'},
            {'input': 'What is the chemical symbol for gold?', 'output': 'Au'},
        ],
    },
    'summarization': {
        'instruction': 'Summarize the following text in 2-3 sentences',
        'examples': [
            {
                'input': (
                    'The Amazon rainforest is the largest tropical rainforest in the world, '
                    'covering over 5.5 million square kilometers. It is home to millions of '
                    'species of plants and animals, many of which have not yet been discovered.'
                ),
                'output': (
                    "The Amazon is the world's largest tropical rainforest, spanning 5.5 million "
                    "sq km and hosting millions of species, many still undiscovered."
                ),
            },
            {
                'input': (
                    'Climate change refers to long-term shifts in global temperatures and weather '
                    'patterns. Human activities have been the main driver since the 1800s, primarily '
                    'through burning fossil fuels.'
                ),
                'output': (
                    'Climate change involves long-term temperature and weather shifts, primarily '
                    'driven by human fossil fuel use since the 1800s.'
                ),
            },
            {
                'input': (
                    'The Great Wall of China was built over many centuries by successive dynasties '
                    'to protect Chinese states from northern invasions. It stretches thousands of '
                    'miles across northern China and is one of the most iconic structures in the world.'
                ),
                'output': (
                    'The Great Wall of China, built over centuries to defend against invasion, '
                    'spans thousands of miles and is one of the world\'s most iconic landmarks.'
                ),
            },
        ],
    },
    'classification': {
        'instruction': 'Classify the sentiment of the following text as positive, negative, or neutral',
        'examples': [
            {
                'input': 'This product is absolutely incredible. Best purchase I have made all year!',
                'output': 'positive',
            },
            {
                'input': 'Terrible quality. Broke after two days. Complete waste of money.',
                'output': 'negative',
            },
            {
                'input': 'It arrived on time and works as described. Nothing special, but does the job.',
                'output': 'neutral',
            },
        ],
    },
    'instruction_following': {
        'instruction': 'Answer the following using the exact format specified',
        'examples': [
            {
                'input': 'List 3 benefits of drinking water using bullet points.',
                'output': '- Keeps you hydrated\n- Improves skin health\n- Aids digestion',
            },
            {
                'input': 'In one word, what is the opposite of hot?',
                'output': 'Cold',
            },
            {
                'input': 'Give 3 steps to make tea using a numbered list.',
                'output': '1. Boil water\n2. Steep the tea bag for 3-5 minutes\n3. Remove the bag and serve',
            },
        ],
    },
}


# ── BENCHMARK EXAMPLES ────────────────────────────────────────────────────────
# 20 examples per task. Real-world inputs people actually write to LLMs.
#
# ground_truth:
#   qa           — short answer for word-overlap scoring
#   summarization — reference summary for ROUGE-1 scoring
#   classification — label string ('positive' / 'negative' / 'neutral')
#   instruction_following — None (constraint field drives the evaluator)
#
# constraints (instruction_following only):
#   list of constraint names passed to InstructionFollowingEvaluator

BENCHMARK_EXAMPLES = {

    # ── QA ────────────────────────────────────────────────────────────────────
    # Mix of general knowledge, tech, and science — the kind people actually ask.
    'qa': [
        {'input': 'What is the capital of France?',                     'ground_truth': 'Paris'},
        {'input': 'Who wrote Romeo and Juliet?',                        'ground_truth': 'Shakespeare'},
        {'input': 'What is the speed of light in km/s?',               'ground_truth': '299792'},
        {'input': 'What year did World War II end?',                    'ground_truth': '1945'},
        {'input': 'What is the smallest prime number?',                 'ground_truth': '2'},
        {'input': 'What does HTTP stand for?',                          'ground_truth': 'Hypertext Transfer Protocol'},
        {'input': 'How many days are in a leap year?',                  'ground_truth': '366'},
        {'input': 'What is the boiling point of water in Celsius?',     'ground_truth': '100'},
        {'input': 'What does API stand for?',                           'ground_truth': 'Application Programming Interface'},
        {'input': 'What is the default port for HTTPS?',               'ground_truth': '443'},
        {'input': 'Who invented the telephone?',                        'ground_truth': 'Alexander Graham Bell'},
        {'input': 'What year was Python first released?',               'ground_truth': '1991'},
        {'input': 'What is the time complexity of binary search?',      'ground_truth': 'O(log n)'},
        {'input': 'What does SQL stand for?',                           'ground_truth': 'Structured Query Language'},
        {'input': 'What is the freezing point of water in Fahrenheit?', 'ground_truth': '32'},
        {'input': 'What is 2 to the power of 10?',                     'ground_truth': '1024'},
        {'input': 'Who painted the Sistine Chapel ceiling?',            'ground_truth': 'Michelangelo'},
        {'input': 'What is the capital of Japan?',                      'ground_truth': 'Tokyo'},
        {'input': 'How many bytes are in a kilobyte?',                  'ground_truth': '1024'},
        {'input': 'What does CPU stand for?',                           'ground_truth': 'Central Processing Unit'},
    ],

    # ── SUMMARIZATION ─────────────────────────────────────────────────────────
    # Real-world passages people paste into ChatGPT/Claude to summarize.
    # ground_truth is a reference summary — activates ROUGE-1 scoring.
    'summarization': [
        {
            'input': (
                'Artificial intelligence has transformed numerous industries over the past decade. '
                'From healthcare diagnostics to autonomous vehicles, AI systems now perform tasks '
                'that once required human expertise. Machine learning algorithms can analyze vast '
                'datasets to identify patterns invisible to human observers, while natural language '
                'processing enables computers to understand and generate human-like text.'
            ),
            'ground_truth': (
                'AI has transformed industries like healthcare and transport by performing expert-level '
                'tasks through machine learning and natural language processing.'
            ),
        },
        {
            'input': (
                "The Great Barrier Reef, located off the coast of Australia, is the world's largest "
                'coral reef system. Composed of over 2,900 individual reefs and 900 islands, it '
                'stretches for over 2,300 kilometers. The reef is home to a remarkable diversity of '
                'life, including 1,500 species of fish, 400 types of coral, and many endangered species.'
            ),
            'ground_truth': (
                "Australia's Great Barrier Reef spans 2,300 km and hosts over 1,500 fish species, "
                '400 coral types, and many endangered animals, making it the world\'s largest coral system.'
            ),
        },
        {
            'input': (
                'Remote work has become increasingly common since 2020. Studies show that many '
                'employees report higher productivity when working from home, though some struggle '
                'with isolation. Companies are now adopting hybrid models that combine office and '
                'remote work to balance flexibility with collaboration.'
            ),
            'ground_truth': (
                'Remote work surged after 2020, boosting productivity for many but causing isolation '
                'for some, prompting companies to adopt hybrid office-remote models.'
            ),
        },
        {
            'input': (
                'Renewable energy sources like solar and wind power have seen dramatic cost reductions '
                'over the past decade. Solar panel costs have fallen by 90% since 2010, making clean '
                'energy increasingly competitive with fossil fuels. Many countries now generate '
                'significant portions of their electricity from renewable sources.'
            ),
            'ground_truth': (
                'Solar and wind energy costs have dropped sharply—solar by 90% since 2010—making '
                'renewables competitive with fossil fuels and widely adopted globally.'
            ),
        },
        {
            'input': (
                'Sleep plays a crucial role in memory consolidation and overall health. Adults '
                'typically need 7-9 hours of sleep per night. Chronic sleep deprivation has been '
                'linked to increased risk of heart disease, obesity, and cognitive decline. Good '
                'sleep hygiene includes maintaining a consistent schedule and limiting screen time '
                'before bed.'
            ),
            'ground_truth': (
                'Adults need 7-9 hours of sleep for memory and health; chronic deprivation raises '
                'risks of heart disease, obesity, and cognitive decline.'
            ),
        },
        {
            'input': (
                'The Python programming language was created by Guido van Rossum and first released '
                'in 1991. It emphasizes code readability and simplicity, using indentation to define '
                'code blocks rather than curly braces. Python has become the dominant language for '
                'data science and machine learning, backed by a vast ecosystem of libraries including '
                'NumPy, Pandas, and TensorFlow.'
            ),
            'ground_truth': (
                'Python, created by Guido van Rossum in 1991, prioritizes readability and has become '
                'the leading language for data science and machine learning.'
            ),
        },
        {
            'input': (
                'Intermittent fasting cycles between periods of eating and fasting. Popular methods '
                'include the 16:8 approach—fasting for 16 hours and eating within an 8-hour window—'
                'and the 5:2 diet, which involves eating normally for five days and restricting '
                'calories on two days. Research suggests benefits include weight loss, improved '
                'insulin sensitivity, and reduced inflammation.'
            ),
            'ground_truth': (
                'Intermittent fasting alternates eating and fasting periods; popular methods like '
                '16:8 and 5:2 are linked to weight loss and improved insulin sensitivity.'
            ),
        },
        {
            'input': (
                'Blockchain is a distributed ledger where data is stored in cryptographically linked '
                'blocks. Each block contains a timestamp and transaction data; once recorded, the '
                'data is extremely difficult to alter. Originally developed for Bitcoin, blockchain '
                'is now applied to supply chain management, voting systems, and digital contracts.'
            ),
            'ground_truth': (
                'Blockchain is a tamper-resistant distributed ledger built for Bitcoin that is now '
                'used in supply chains, voting systems, and digital contracts.'
            ),
        },
        {
            'input': (
                'The James Webb Space Telescope, launched in December 2021, is the most powerful '
                'space observatory ever built. It observes the universe in infrared, allowing it '
                'to peer through dust clouds and study the earliest galaxies formed after the Big Bang. '
                'Its first images, released in 2022, revealed galaxies billions of light-years away '
                'in unprecedented detail.'
            ),
            'ground_truth': (
                'Launched in 2021, the James Webb Space Telescope uses infrared to observe early '
                'galaxies and distant cosmic structures with unprecedented clarity.'
            ),
        },
        {
            'input': (
                'Credit scores in the US range from 300 to 850 and are calculated from payment '
                'history, amounts owed, length of credit history, new credit inquiries, and credit '
                'mix. A score above 700 is generally considered good, while above 800 is excellent. '
                'Lenders use credit scores to set interest rates and decide whether to approve loans, '
                'credit cards, or mortgages.'
            ),
            'ground_truth': (
                'US credit scores (300-850) factor in payment history, debt levels, and credit history '
                'length; scores above 700 are good and affect loan approvals and interest rates.'
            ),
        },
        {
            'input': (
                'The Mediterranean diet is based on the traditional eating habits of countries '
                'bordering the Mediterranean Sea. It emphasizes fruits, vegetables, whole grains, '
                'legumes, nuts, and olive oil, with moderate amounts of fish and poultry and minimal '
                'red meat. Numerous studies have linked it to reduced risk of heart disease, stroke, '
                'and type 2 diabetes.'
            ),
            'ground_truth': (
                'The Mediterranean diet, centered on plants, whole grains, and olive oil, is '
                'consistently linked to lower risks of heart disease, stroke, and diabetes.'
            ),
        },
        {
            'input': (
                'Docker allows developers to package applications and their dependencies into '
                'containers—lightweight, portable units that run consistently across different '
                'environments. Unlike virtual machines, containers share the host operating system, '
                'making them faster and more resource-efficient. Docker has become a cornerstone '
                'of modern DevOps and microservices architecture.'
            ),
            'ground_truth': (
                'Docker packages apps into lightweight containers that run consistently across '
                'environments, making it faster than VMs and central to DevOps and microservices.'
            ),
        },
        {
            'input': (
                'Elon Musk acquired Twitter in October 2022 for approximately $44 billion and '
                'subsequently rebranded it as X. He laid off roughly half the workforce and '
                'replaced the free verified badge system with paid Twitter Blue subscriptions. '
                'The changes triggered significant advertiser departures and user migration to '
                'alternatives like Bluesky and Mastodon.'
            ),
            'ground_truth': (
                'Musk bought Twitter for $44B in 2022, rebranded it to X, cut half the staff, '
                'shifted to paid verification, and drove away advertisers and users.'
            ),
        },
        {
            'input': (
                'Large language models like GPT-4 are trained on massive text corpora using '
                'self-supervised learning, where the model learns to predict the next word in a '
                'sequence. After pre-training, they are refined with reinforcement learning from '
                'human feedback (RLHF), which teaches the model to produce responses that humans '
                'rate as helpful, harmless, and honest.'
            ),
            'ground_truth': (
                'LLMs are pre-trained by predicting next words on large text datasets, then '
                'fine-tuned with RLHF to be more helpful and aligned with human preferences.'
            ),
        },
        {
            'input': (
                'The gut microbiome consists of trillions of bacteria, fungi, and viruses living '
                'in the digestive tract. Research increasingly links gut health to immune function, '
                'mental wellbeing, and neurological conditions. Diet plays a key role: fiber, '
                'fermented foods, and diverse plant intake support a healthy microbiome, while '
                'ultra-processed foods and antibiotics can disrupt it.'
            ),
            'ground_truth': (
                'The gut microbiome influences immunity, mental health, and neurology; a diverse, '
                'fiber-rich diet supports it while processed foods and antibiotics disrupt it.'
            ),
        },
        {
            'input': (
                'TypeScript is a statically typed superset of JavaScript developed by Microsoft. '
                'Types catch bugs at compile time rather than at runtime, making large codebases '
                'easier to maintain and refactor. TypeScript compiles to plain JavaScript and runs '
                'anywhere JavaScript does, and has become the default choice for large-scale web '
                'applications at companies like Google, Airbnb, and Slack.'
            ),
            'ground_truth': (
                "TypeScript adds static typing to JavaScript, catching bugs earlier and improving "
                "maintainability; it compiles to JavaScript and is widely adopted in large-scale web development."
            ),
        },
        {
            'input': (
                'Inflation erodes purchasing power as the general price level rises over time. '
                'Central banks like the US Federal Reserve combat it by raising interest rates, '
                'which makes borrowing more expensive and slows spending and investment. Most '
                'developed economies target an annual inflation rate of around 2%, balancing '
                'growth with price stability.'
            ),
            'ground_truth': (
                'Inflation erodes purchasing power; central banks raise interest rates to slow '
                'spending and bring inflation back toward the 2% target.'
            ),
        },
        {
            'input': (
                'The Stanford prison experiment was conducted by Philip Zimbardo in 1971. '
                'Participants were randomly assigned roles of guards or prisoners in a simulated '
                'jail. The study was terminated after just six days when mock guards began '
                'psychologically abusing prisoners. It is widely cited as evidence of how '
                'situational factors shape behavior, though its methodology has been heavily '
                'criticized in recent decades.'
            ),
            'ground_truth': (
                "Zimbardo's 1971 Stanford prison experiment was halted after 6 days when fake guards "
                "abused prisoners, illustrating situational influences on behavior—though its methodology remains contested."
            ),
        },
        {
            'input': (
                'Attention deficit hyperactivity disorder (ADHD) is a neurodevelopmental condition '
                'characterized by persistent inattention, hyperactivity, and impulsivity that '
                'interferes with daily functioning. It affects approximately 5-7% of children '
                'and often persists into adulthood. Treatment typically combines behavioral '
                'therapy with medication such as stimulants like methylphenidate or amphetamines.'
            ),
            'ground_truth': (
                'ADHD is a neurodevelopmental disorder affecting 5-7% of children, marked by '
                'inattention and hyperactivity, typically treated with behavioral therapy and stimulant medication.'
            ),
        },
        {
            'input': (
                'Quantum computing uses the principles of quantum mechanics—superposition and '
                'entanglement—to perform computations that would be infeasible for classical '
                'computers. Rather than classical bits that are either 0 or 1, quantum bits '
                '(qubits) can exist in multiple states simultaneously. Applications include '
                'drug discovery, cryptography, and optimization problems, though practical '
                'large-scale quantum computers remain years away.'
            ),
            'ground_truth': (
                'Quantum computing uses qubits in superposition to solve problems infeasible for '
                'classical computers, with potential applications in cryptography, drug discovery, '
                'and optimization—though large-scale systems are still years away.'
            ),
        },
    ],

    # ── CLASSIFICATION ────────────────────────────────────────────────────────
    # Authentic review and comment text. Labels: 'positive', 'negative', 'neutral'.
    # Representative of product reviews, restaurant feedback, app stores, travel.
    'classification': [
        # ── positive (7) ──
        {
            'input': (
                'Absolutely love this laptop. Fast, lightweight, and the battery lasts all day. '
                'Worth every penny.'
            ),
            'ground_truth': 'positive',
        },
        {
            'input': (
                "Just finished this book and I couldn't put it down. The plot twists kept me "
                'guessing right until the last page.'
            ),
            'ground_truth': 'positive',
        },
        {
            'input': (
                'The customer service rep was incredibly helpful and resolved my issue in under '
                '5 minutes. Rare to see this level of care these days.'
            ),
            'ground_truth': 'positive',
        },
        {
            'input': (
                'Best ramen I have had outside of Japan. The broth is incredibly rich, noodles '
                'are perfectly chewy, and the portions are generous. Already planning my return.'
            ),
            'ground_truth': 'positive',
        },
        {
            'input': (
                'This online course completely changed how I approach machine learning. '
                'Clear explanations, great exercises, and the instructor actually responds to questions.'
            ),
            'ground_truth': 'positive',
        },
        {
            'input': (
                'Delivered two days early and packaged exceptionally well. The product matches '
                'the description exactly. This seller has earned a loyal customer.'
            ),
            'ground_truth': 'positive',
        },
        {
            'input': (
                'Hotel exceeded every expectation. Room was spotless, staff were genuinely '
                'friendly, and the breakfast buffet had something for everyone. Will definitely stay again.'
            ),
            'ground_truth': 'positive',
        },
        # ── negative (7) ──
        {
            'input': (
                'Do not buy this. Stopped working after a week and the customer support team '
                'was completely useless. Total waste of money.'
            ),
            'ground_truth': 'negative',
        },
        {
            'input': (
                'Waited 45 minutes for a table even with a reservation, food arrived cold, '
                'and the waiter barely acknowledged us. Never coming back.'
            ),
            'ground_truth': 'negative',
        },
        {
            'input': (
                'The app crashes every single time I try to open it on my iPhone 15. '
                "Reported the bug twice and got no response. Uninstalled."
            ),
            'ground_truth': 'negative',
        },
        {
            'input': (
                'Misleading product photos — what arrived looked nothing like what was advertised. '
                'Returning immediately and disputing the charge.'
            ),
            'ground_truth': 'negative',
        },
        {
            'input': (
                'Overpriced and deeply underwhelming. Paid $85 for a meal that tasted like '
                'frozen food microwaved and served on a fancy plate.'
            ),
            'ground_truth': 'negative',
        },
        {
            'input': (
                'Flight delayed 4 hours with zero communication from the airline. Missed my '
                'connection, had to rebook at my own expense. Absolutely infuriating.'
            ),
            'ground_truth': 'negative',
        },
        {
            'input': (
                'The build quality is shocking for this price point. Plastic feels cheap, '
                'buttons stick, and the screen has dead pixels straight out of the box.'
            ),
            'ground_truth': 'negative',
        },
        # ── neutral (6) ──
        {
            'input': (
                'Product arrived as described. Packaging was intact and delivery was on schedule. '
                'Does what it says on the box.'
            ),
            'ground_truth': 'neutral',
        },
        {
            'input': (
                'Decent coffee shop. Nothing remarkable but nothing to complain about either. '
                'I would return if I happened to be nearby.'
            ),
            'ground_truth': 'neutral',
        },
        {
            'input': (
                'The software has a steep learning curve, but once you get used to it, it works '
                'reliably enough for basic tasks. Documentation could be better.'
            ),
            'ground_truth': 'neutral',
        },
        {
            'input': (
                'Flight was on time, seats were standard economy, and the in-flight food was '
                'mediocre. An entirely average flying experience.'
            ),
            'ground_truth': 'neutral',
        },
        {
            'input': (
                'Book had some interesting ideas but felt repetitive by the halfway point. '
                'Finished it but would not read it again.'
            ),
            'ground_truth': 'neutral',
        },
        {
            'input': (
                'Gym is clean and has the equipment I need. Gets crowded on weekday evenings '
                'but is fine on weekends. Standard pricing for the area.'
            ),
            'ground_truth': 'neutral',
        },

    ],

    # ── INSTRUCTION FOLLOWING ─────────────────────────────────────────────────
    # Each input embeds the format constraint naturally, as people actually phrase it.
    # 'constraints' tells the evaluator what structural check to apply.
    # Mix: bullet_points (7), numbered_list (7), single_word (6).
    'instruction_following': [
        # ── bullet_points (7) ──
        {
            'input': 'Using bullet points, list 4 reasons why people choose Python for data science.',
            'constraints': ['bullet_points'],
            'ground_truth': None,
        },
        {
            'input': 'List the main causes of the 2008 financial crisis using bullet points.',
            'constraints': ['bullet_points'],
            'ground_truth': None,
        },
        {
            'input': 'Name 5 popular JavaScript frameworks using bullet points.',
            'constraints': ['bullet_points'],
            'ground_truth': None,
        },
        {
            'input': 'Using bullet points, describe 3 key differences between supervised and unsupervised learning.',
            'constraints': ['bullet_points'],
            'ground_truth': None,
        },
        {
            'input': 'List the core principles of agile software development using bullet points.',
            'constraints': ['bullet_points'],
            'ground_truth': None,
        },
        {
            'input': 'Using bullet points, give 5 practical tips for improving sleep quality.',
            'constraints': ['bullet_points'],
            'ground_truth': None,
        },
        {
            'input': 'List 4 pros and 4 cons of remote work using bullet points.',
            'constraints': ['bullet_points'],
            'ground_truth': None,
        },
        # ── numbered_list (7) ──
        {
            'input': 'Give me step-by-step instructions for setting up a Python virtual environment. Use a numbered list.',
            'constraints': ['numbered_list'],
            'ground_truth': None,
        },
        {
            'input': 'List the steps to resolve a merge conflict in git using a numbered list.',
            'constraints': ['numbered_list'],
            'ground_truth': None,
        },
        {
            'input': 'Using a numbered list, walk me through how to prepare for a job interview.',
            'constraints': ['numbered_list'],
            'ground_truth': None,
        },
        {
            'input': 'List the steps of the scientific method in order. Use a numbered list.',
            'constraints': ['numbered_list'],
            'ground_truth': None,
        },
        {
            'input': 'Give me a numbered list of steps to troubleshoot a slow internet connection.',
            'constraints': ['numbered_list'],
            'ground_truth': None,
        },
        {
            'input': 'Using a numbered list, describe the stages of the software development lifecycle (SDLC).',
            'constraints': ['numbered_list'],
            'ground_truth': None,
        },
        {
            'input': 'List 5 steps for writing a strong cover letter. Use a numbered list.',
            'constraints': ['numbered_list'],
            'ground_truth': None,
        },
        # ── single_word (6) ──
        {
            'input': 'In one word, what programming language is primarily used for iOS app development?',
            'constraints': ['single_word'],
            'ground_truth': None,
        },
        {
            'input': 'Answer in a single word: what is the opposite of encryption?',
            'constraints': ['single_word'],
            'ground_truth': None,
        },
        {
            'input': 'In one word, what element has the chemical symbol Fe?',
            'constraints': ['single_word'],
            'ground_truth': None,
        },
        {
            'input': 'Reply with a single word: what do you call a function defined inside a class in Python?',
            'constraints': ['single_word'],
            'ground_truth': None,
        },
        {
            'input': 'In one word, what is the biological process by which plants produce food using sunlight?',
            'constraints': ['single_word'],
            'ground_truth': None,
        },
        {
            'input': 'Answer with a single word: what is the most widely spoken native language in the world?',
            'constraints': ['single_word'],
            'ground_truth': None,
        },
    ],
}


def generate_prompt(strategy: str, task_type: str, example: dict) -> str:
    """
    Generate a prompt using the specified strategy.

    Args:
        strategy: One of 'zero_shot', 'zero_shot_verbose', 'few_shot', 'cot', 'concise'
        task_type: 'qa', 'summarization', 'classification', or 'instruction_following'
        example: Dict with 'input' key (and optionally 'ground_truth', 'constraints')

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
