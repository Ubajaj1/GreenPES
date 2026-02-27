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

    @staticmethod
    def cot_stepped(task_instruction: str, input_text: str, steps: int = 3) -> str:
        """Chain-of-thought with a fixed number of step starters shown."""
        step_starters = "\n".join(f"{i+1}." for i in range(steps))
        return f"""{task_instruction}: {input_text}

Let's think step by step:
{step_starters}"""

    @staticmethod
    def verbose_detailed(task_instruction: str, input_text: str) -> str:
        """Verbose prompt with additional role context and output guidance."""
        return f"""You are a knowledgeable and precise expert assistant.
Your task is to {task_instruction.lower()}.
Please reason carefully before answering, and provide a well-structured response.
If the question is ambiguous, state your interpretation clearly.

Input: {input_text}

Your response:"""


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


# ── Tag existing examples as 'easy' ──────────────────────────────────────────
for _task_examples in BENCHMARK_EXAMPLES.values():
    for _ex in _task_examples:
        _ex.setdefault('difficulty', 'easy')


# ── HARD EXAMPLES ─────────────────────────────────────────────────────────────
# 10 per task; difficulty='hard'. Merged into BENCHMARK_EXAMPLES below.
# QA:   multi-hop, math word problems, ambiguous
# Summ: 300+ word passages, competing viewpoints, technical content
# Cls:  sarcasm, irony, double negatives, mixed sentiment
# IF:   multi-constraint, specific counts, exclusion rules

_HARD_EXAMPLES: dict[str, list[dict]] = {

    # ── QA hard ───────────────────────────────────────────────────────────────
    'qa': [
        {
            # Multi-hop: largest land area → Russia → capital
            'input': 'What is the capital of the country with the largest land area?',
            'ground_truth': 'Moscow',
            'difficulty': 'hard',
        },
        {
            # Multi-hop: inventor of WWW → Tim Berners-Lee → knighthood year
            'input': 'In what year did the inventor of the World Wide Web receive his knighthood?',
            'ground_truth': '2004',
            'difficulty': 'hard',
        },
        {
            # Math word problem
            'input': 'A train travels 120 km in 1.5 hours. At the same speed, how many km will it cover in 4 hours?',
            'ground_truth': '320',
            'difficulty': 'hard',
        },
        {
            # Math: area → perimeter
            'input': 'A square has an area of 144 square meters. What is its perimeter in meters?',
            'ground_truth': '48',
            'difficulty': 'hard',
        },
        {
            # Multi-step percentage
            'input': 'If 15% of a number is 75, what is 40% of that same number?',
            'ground_truth': '200',
            'difficulty': 'hard',
        },
        {
            # Multi-hop: second largest planet → Saturn → moon count
            'input': 'Which planet has more moons: the largest or the second-largest planet in the solar system?',
            'ground_truth': 'Saturn',
            'difficulty': 'hard',
        },
        {
            # Multi-hop: element in 78% of atmosphere → nitrogen → boiling point
            'input': 'What is the boiling point in Celsius of the element that makes up about 78% of Earth\'s atmosphere?',
            'ground_truth': '-196',
            'difficulty': 'hard',
        },
        {
            # Cooking math
            'input': 'A recipe needs 3 cups of flour for 24 cookies. How many cups are needed for 36 cookies?',
            'ground_truth': '4.5',
            'difficulty': 'hard',
        },
        {
            # Ambiguous "father of" question
            'input': 'Who is widely considered the father of modern computer science?',
            'ground_truth': 'Alan Turing',
            'difficulty': 'hard',
        },
        {
            # Multi-hop: most abundant metal in crust → aluminum → symbol
            'input': 'What is the chemical symbol for the most abundant metal in Earth\'s crust?',
            'ground_truth': 'Al',
            'difficulty': 'hard',
        },
    ],

    # ── Summarization hard ────────────────────────────────────────────────────
    'summarization': [
        {
            'input': (
                'The debate over nuclear energy has intensified as governments seek low-carbon electricity sources. '
                'Proponents argue that modern reactor designs, including small modular reactors (SMRs), are far '
                'safer than their predecessors and can provide reliable baseload power that solar and wind cannot '
                'match without expensive battery storage. They cite France, where nuclear supplies nearly 70% of '
                'electricity with among the lowest grid carbon intensity in Europe. Critics counter that uranium '
                'mining is environmentally damaging, construction costs have repeatedly overrun budgets—with '
                'projects like the UK\'s Hinkley Point C now estimated at over £30 billion—and that the unsolved '
                'problem of long-term radioactive waste storage poses intergenerational risks. Environmental groups '
                'remain deeply divided, with some prominent figures like Stewart Brand and James Lovelock embracing '
                'nuclear as a climate necessity, while Greenpeace and others maintain their historic opposition.'
            ),
            'ground_truth': (
                'Nuclear energy divides opinion: proponents highlight its low-carbon baseload reliability and France\'s '
                'success, while critics cite high construction costs, uranium mining damage, and unresolved nuclear waste '
                'storage as fundamental barriers.'
            ),
            'difficulty': 'hard',
        },
        {
            'input': (
                'Transformer neural networks, introduced by Vaswani et al. in the 2017 paper "Attention Is All You '
                'Need," replaced recurrent architectures for sequence modeling. The core innovation is the self-attention '
                'mechanism, which allows each token in a sequence to attend to every other token simultaneously, enabling '
                'parallelization during training. A transformer block consists of multi-head self-attention, layer '
                'normalization, and position-wise feed-forward networks. Positional encodings are added to token '
                'embeddings because, unlike RNNs, transformers have no inherent notion of sequence order. Scaling these '
                'architectures to billions of parameters—enabled by attention\'s parallelism—produced large language '
                'models that exhibit emergent capabilities such as few-shot learning and complex reasoning that smaller '
                'models lack. The compute cost scales quadratically with sequence length, which has motivated research '
                'into efficient attention variants like sparse attention, linear attention, and sliding-window approaches.'
            ),
            'ground_truth': (
                'Transformers use self-attention to process all tokens simultaneously, enabling parallelization and '
                'scaling to billions of parameters; this unlocks emergent capabilities but scales quadratically with '
                'sequence length, driving research into efficient attention variants.'
            ),
            'difficulty': 'hard',
        },
        {
            'input': (
                'The 2008 global financial crisis stemmed from a confluence of factors spanning two decades. '
                'Financial deregulation in the 1980s and 1990s allowed banks to engage in riskier activities. '
                'The U.S. housing market boomed through the early 2000s, fueled by low interest rates and '
                'aggressive mortgage lending to borrowers with poor creditworthiness—so-called subprime mortgages. '
                'These loans were bundled into complex instruments called collateralized debt obligations (CDOs), '
                'which were then rated as investment-grade by agencies that failed to model correlated default risk. '
                'When housing prices peaked and began falling in 2006, default rates surged. The failure of Bear '
                'Stearns in March 2008 and Lehman Brothers in September 2008 triggered a global credit freeze. '
                'Governments intervened with unprecedented bailouts—TARP in the U.S. spent $700 billion—and '
                'coordinated central bank action, yet the subsequent recession cost millions their jobs and homes.'
            ),
            'ground_truth': (
                'The 2008 crisis arose from decades of deregulation, subprime mortgage lending bundled into '
                'mispriced CDOs, and rating failures; Lehman Brothers\' collapse triggered a global credit freeze '
                'requiring massive government bailouts.'
            ),
            'difficulty': 'hard',
        },
        {
            'input': (
                'CRISPR-Cas9 gene editing allows scientists to make precise cuts in DNA at targeted locations. '
                'The system uses a guide RNA to direct the Cas9 protein—a molecular scissor—to a specific genomic '
                'sequence. After cutting, the cell\'s natural repair mechanisms can be harnessed: if a donor template '
                'is provided, the cell performs homology-directed repair, inserting or replacing genetic material with '
                'high precision. Without a template, the error-prone non-homologous end joining pathway introduces '
                'insertions or deletions that typically disrupt gene function. Clinical applications include sickle '
                'cell disease, where CRISPR reactivates fetal hemoglobin; inherited blindness caused by CEP290 '
                'mutations; and certain forms of cancer. Off-target edits—cuts at unintended genomic sites—remain '
                'a safety concern, prompting development of high-fidelity Cas9 variants and base editors that '
                'chemically modify individual bases without creating double-strand breaks.'
            ),
            'ground_truth': (
                'CRISPR-Cas9 uses guide RNA to direct precise DNA cuts, enabling gene correction via repair '
                'pathways; it has shown clinical promise for genetic diseases but requires continued improvement '
                'of off-target editing accuracy.'
            ),
            'difficulty': 'hard',
        },
        {
            'input': (
                'Modern central bank policy faces a tension between two mandates: price stability and full '
                'employment. Traditional models assumed a stable Phillips curve—a predictable tradeoff where '
                'lower unemployment leads to higher inflation. Decades of data suggest this relationship has '
                'flattened, possibly due to globalization anchoring goods prices and better-anchored inflation '
                'expectations. The 2021–2023 inflation surge challenged this view: supply chain disruptions '
                'from COVID-19, energy price spikes from the Ukraine conflict, and pandemic-era fiscal stimulus '
                'drove inflation to 40-year highs in major economies. Central banks responded with rapid rate '
                'hikes—the Federal Reserve raised its benchmark rate from near-zero to over 5% in 18 months. '
                'Critics argue the hiking cycle risked triggering unnecessary recession, while others contend '
                'that delayed action allowed inflation to become entrenched. Whether the Fed achieved a "soft '
                'landing"—lowering inflation without deep recession—remains contested among economists.'
            ),
            'ground_truth': (
                'Central banks face tension between inflation control and employment; the 2021-2023 inflation '
                'surge driven by supply shocks and fiscal stimulus prompted rapid rate hikes, with debate '
                'ongoing over whether the response risked unnecessary recession or succeeded in a soft landing.'
            ),
            'difficulty': 'hard',
        },
        {
            'input': (
                'Urban heat islands (UHIs) occur when cities experience significantly higher temperatures than '
                'surrounding rural areas—often 1 to 7 degrees Celsius warmer on average, with peaks exceeding '
                '10°C on calm, clear nights. Dark impervious surfaces like asphalt and rooftops absorb and '
                'retain solar radiation; vegetation is replaced by materials with low albedo; and waste heat '
                'from vehicles, air conditioning, and industry adds directly to urban air. The feedback loop '
                'is vicious: higher temperatures increase air conditioning demand, which releases more heat '
                'outdoors, further warming the city. Mitigation strategies include green roofs, urban tree '
                'canopy expansion, reflective "cool pavement," and district cooling systems. Equity concerns '
                'are acute: low-income urban neighborhoods typically have less tree cover, older housing '
                'stock without insulation, and higher proportions of elderly and outdoor workers vulnerable '
                'to heat illness. As climate change raises baseline temperatures, effective UHI mitigation '
                'is becoming a public health imperative.'
            ),
            'ground_truth': (
                'Urban heat islands arise from heat-absorbing surfaces, lost vegetation, and waste heat, '
                'raising city temperatures up to 10°C above rural areas; mitigation like green roofs and '
                'cool pavements is urgent but equity challenges persist, especially for low-income neighborhoods.'
            ),
            'difficulty': 'hard',
        },
        {
            'input': (
                'The effective altruism (EA) movement, popularized by philosopher Peter Singer\'s "drowning '
                'child" thought experiment, argues that we have a strong moral obligation to help those in '
                'need when we can do so at little cost to ourselves—and that we should choose causes and '
                'interventions based on evidence of impact rather than emotional proximity. EA prioritizes '
                'global health, animal welfare, and existential risk reduction from artificial intelligence '
                'or pandemics. Critics, including philosophers like Samuel Scheffler, argue that EA\'s '
                'impartialist demands are psychologically unrealistic and ignore the special obligations we '
                'have to particular people and communities. The 2022 collapse of FTX and revelations about '
                'Sam Bankman-Fried—who publicly espoused EA while allegedly committing large-scale fraud—'
                'raised questions about whether consequentialist reasoning can rationalize unethical behavior '
                'and whether the movement prioritizes abstract future risks over concrete present-day harms.'
            ),
            'ground_truth': (
                'Effective altruism promotes evidence-based, impartial philanthropy prioritizing global health '
                'and existential risk; critics question its psychological demands and special-obligation neglect, '
                'while FTX\'s collapse raised concerns about consequentialist reasoning enabling rationalized '
                'misconduct.'
            ),
            'difficulty': 'hard',
        },
        {
            'input': (
                'Neuroplasticity refers to the brain\'s ability to reorganize itself by forming new neural '
                'connections throughout life. Once thought to occur only during early development, research '
                'now shows significant plasticity persists into adulthood. The hippocampus, critical for '
                'memory formation, continues to generate new neurons—a process called adult neurogenesis—'
                'though the extent and functional significance in humans remains debated. Experience-dependent '
                'plasticity drives skill acquisition: repeated practice strengthens synaptic connections '
                'through long-term potentiation (LTP), while unused pathways weaken via long-term depression '
                '(LTD). Stroke rehabilitation exploits plasticity: undamaged brain regions can partially '
                'assume functions of damaged areas. However, plasticity is bidirectional—it also underlies '
                'maladaptive processes like addiction, chronic pain, and anxiety disorders, where repeated '
                'signals strengthen pathways that would better remain weak.'
            ),
            'ground_truth': (
                'Neuroplasticity—the brain\'s lifelong ability to rewire itself—underlies learning, stroke '
                'recovery, and skill development via LTP and LTD, but also drives maladaptive conditions '
                'like addiction and chronic pain.'
            ),
            'difficulty': 'hard',
        },
        {
            'input': (
                'The geopolitics of rare earth elements (REEs) has emerged as a significant source of '
                'international tension. Seventeen metallic elements classified as rare earths are critical '
                'inputs for electric vehicle motors, wind turbines, smartphones, and precision-guided '
                'military systems. Despite the name, most REEs are not geologically rare—they are, however, '
                'concentrated in economically viable deposits primarily in China, which controls roughly '
                '60% of global production and an even higher share of processing capacity. China used '
                'REE export restrictions during its 2010 dispute with Japan over the Senkaku Islands, '
                'demonstrating willingness to weaponize this supply chain advantage. Western governments '
                'have since invested in domestic extraction, allied-country diversification (notably '
                'Australia and Canada), and recycling programs, but building new processing infrastructure '
                'takes a decade and involves significant environmental costs from toxic by-products.'
            ),
            'ground_truth': (
                'China dominates rare earth production and processing, creating strategic vulnerabilities '
                'it has demonstrated willingness to exploit; Western efforts to diversify supply face '
                'decade-long timelines and environmental challenges.'
            ),
            'difficulty': 'hard',
        },
        {
            'input': (
                'Large-scale language model pretraining follows a scaling hypothesis: more parameters, '
                'more data, and more compute reliably improve performance across tasks. The Chinchilla '
                'paper (Hoffmann et al., 2022) challenged prevailing practice by showing that many models '
                'of the era were undertrained relative to their parameter count—optimal compute allocation '
                'requires scaling data and model size equally. Chinchilla (70B parameters, 1.4T tokens) '
                'outperformed GPT-3 (175B parameters) trained on 300B tokens despite being smaller. '
                'Subsequent models like LLaMA-2 and Mistral adopted data-rich training regimes accordingly. '
                'However, the scaling hypothesis has limits: "emergent" capabilities appear discontinuously '
                'as models cross certain scales, and performance on reasoning benchmarks can plateau or '
                'regress with further scaling without architectural or training changes. Whether scaling '
                'alone can produce general intelligence remains deeply contested.'
            ),
            'ground_truth': (
                'The Chinchilla paper showed that optimal LLM training requires scaling data and model size '
                'equally; it outperformed larger undertrained models, reshaping training practices, though '
                'whether scaling alone produces general intelligence remains contested.'
            ),
            'difficulty': 'hard',
        },
    ],

    # ── Classification hard ───────────────────────────────────────────────────
    'classification': [
        {
            # Sarcasm → negative
            'input': (
                'Oh great, another software update that fixes bugs I never had and introduces ones I now have '
                'to live with. Truly the gift that keeps on giving.'
            ),
            'ground_truth': 'negative',
            'difficulty': 'hard',
        },
        {
            # Mixed: praise food, criticize service
            'input': (
                'The food was genuinely outstanding — best risotto I have had in years. But the service was '
                'so slow and dismissive that it overshadowed everything else.'
            ),
            'ground_truth': 'negative',
            'difficulty': 'hard',
        },
        {
            # Irony → negative
            'input': (
                'Nothing like spending three hours on hold with customer support to really put your day in '
                'perspective. Absolutely loved every second of it.'
            ),
            'ground_truth': 'negative',
            'difficulty': 'hard',
        },
        {
            # Hedged, mildly negative leaning neutral
            'input': (
                "Wouldn't say it's my favorite hotel chain, but I've definitely stayed in worse places. "
                'It does the job if you just need somewhere to sleep.'
            ),
            'ground_truth': 'neutral',
            'difficulty': 'hard',
        },
        {
            # Mixed: packaging positive, product negative → neutral
            'input': (
                'Absolutely gorgeous packaging — they clearly put thought into presentation. '
                'The product inside, however, was deeply underwhelming and not worth the premium price.'
            ),
            'ground_truth': 'neutral',
            'difficulty': 'hard',
        },
        {
            # Sarcasm with exclamation → negative
            'input': (
                'An 8-hour flight with a 4-hour delay and two broken entertainment screens — '
                "what a seamless travel experience! Highly recommend if you enjoy existential suffering."
            ),
            'ground_truth': 'negative',
            'difficulty': 'hard',
        },
        {
            # Double negative → positive
            'input': (
                "I can't say I wasn't impressed. The interface is cleaner than I expected, "
                'and I have no complaints about the performance.'
            ),
            'ground_truth': 'positive',
            'difficulty': 'hard',
        },
        {
            # Conditional positive
            'input': (
                'If you can overlook the steep price and the occasional lag on startup, '
                "it's actually a pretty solid piece of software once you get into the workflow."
            ),
            'ground_truth': 'positive',
            'difficulty': 'hard',
        },
        {
            # Narrative arc (negative → positive) → positive
            'input': (
                'Started off rocky — confusing UI and a frustrating onboarding process. '
                'But once I got past the learning curve, it became genuinely indispensable to my workflow.'
            ),
            'ground_truth': 'positive',
            'difficulty': 'hard',
        },
        {
            # Idiomatic negative
            'input': (
                "Classic case of overpromising and underdelivering. The marketing made it sound "
                "revolutionary; the reality was mediocre at best."
            ),
            'ground_truth': 'negative',
            'difficulty': 'hard',
        },
    ],

    # ── Instruction following hard ────────────────────────────────────────────
    'instruction_following': [
        {
            # Multi-constraint: bullet points with a specific count
            'input': 'In exactly 4 bullet points, explain the four key properties of a relational database.',
            'constraints': ['bullet_points'],
            'ground_truth': None,
            'difficulty': 'hard',
        },
        {
            # Numbered list with embedded format requirement per item
            'input': (
                'List 5 sorting algorithms in a numbered list. '
                'After each algorithm name, include its average-case time complexity in parentheses.'
            ),
            'constraints': ['numbered_list'],
            'ground_truth': None,
            'difficulty': 'hard',
        },
        {
            # Single word but requires reasoning to identify the word
            'input': 'In one word only, what is the term for a function that calls itself in programming?',
            'constraints': ['single_word'],
            'ground_truth': None,
            'difficulty': 'hard',
        },
        {
            # Bullet points with exclusion rule
            'input': (
                'Using bullet points, name 5 machine learning algorithms. '
                'Do NOT include any neural network or deep learning methods.'
            ),
            'constraints': ['bullet_points'],
            'ground_truth': None,
            'difficulty': 'hard',
        },
        {
            # Numbered list with per-step constraint
            'input': (
                'Give step-by-step instructions for setting up SSH key authentication on a Linux server. '
                'Use a numbered list and keep each step to one sentence maximum.'
            ),
            'constraints': ['numbered_list'],
            'ground_truth': None,
            'difficulty': 'hard',
        },
        {
            # Single word — technical Python term
            'input': (
                'In a single word, what is the name for a special Python method that starts and ends '
                'with double underscores (e.g., __init__)?'
            ),
            'constraints': ['single_word'],
            'ground_truth': None,
            'difficulty': 'hard',
        },
        {
            # Bullet points with action-verb requirement (complex format instruction)
            'input': (
                'Using exactly 3 bullet points, each starting with an action verb, '
                'describe how the HTTPS handshake works.'
            ),
            'constraints': ['bullet_points'],
            'ground_truth': None,
            'difficulty': 'hard',
        },
        {
            # Numbered list — complex domain
            'input': (
                'Using a numbered list, describe the 6 stages of the software development lifecycle (SDLC) '
                'in order. Include only the stage names and a one-sentence description of each.'
            ),
            'constraints': ['numbered_list'],
            'ground_truth': None,
            'difficulty': 'hard',
        },
        {
            # Single word — output of an expression
            'input': (
                'Answer with a single word: in Python, what data type is returned by the expression '
                'type(3.14).__name__?'
            ),
            'constraints': ['single_word'],
            'ground_truth': None,
            'difficulty': 'hard',
        },
        {
            # Bullet list with comparison (structured but complex)
            'input': (
                'In exactly 4 bullet points, list 4 key differences between REST APIs and GraphQL. '
                'Start each bullet with the aspect being compared.'
            ),
            'constraints': ['bullet_points'],
            'ground_truth': None,
            'difficulty': 'hard',
        },
    ],
}

# Merge hard examples into BENCHMARK_EXAMPLES
for _task, _hard_list in _HARD_EXAMPLES.items():
    BENCHMARK_EXAMPLES[_task].extend(_hard_list)


# ── SCALING STRATEGIES ────────────────────────────────────────────────────────
# 12 parameterized variants for Direction 2 (token efficiency scaling laws).
# Names encode the parameter: concise_10w, few_shot_2, cot_3step, verbose_1.

SCALING_STRATEGIES: list[str] = [
    'concise_10w',
    'concise_25w',
    'concise_50w',
    'concise_100w',
    'concise_200w',
    'few_shot_1',
    'few_shot_2',
    'few_shot_3',
    'cot_1step',
    'cot_3step',
    'cot_5step',
    'verbose_1',
]


def generate_prompt(strategy: str, task_type: str, example: dict) -> str:
    """
    Generate a prompt using the specified strategy.

    Core strategies: 'zero_shot', 'zero_shot_verbose', 'few_shot', 'cot', 'concise'

    Scaling strategy variants (Direction 2):
      - concise_Nw   → concise prompt with word limit N (e.g. concise_10w → max 10 words)
      - few_shot_N   → few-shot with N examples (1, 2, or 3)
      - cot_Nstep    → chain-of-thought with N step starters shown
      - verbose_1    → verbose prompt with extra role context

    Args:
        strategy:  Strategy name (core or scaling variant)
        task_type: 'qa', 'summarization', 'classification', or 'instruction_following'
        example:   Dict with 'input' key (and optionally 'ground_truth', 'constraints')

    Returns:
        Formatted prompt string
    """
    config = TASK_CONFIGS[task_type]
    instruction = config['instruction']
    input_text = example['input']

    # ── Core strategies ───────────────────────────────────────────────────────
    if strategy == 'zero_shot':
        return PromptingStrategy.zero_shot(instruction, input_text)
    elif strategy == 'zero_shot_verbose':
        return PromptingStrategy.zero_shot_verbose(instruction, input_text)
    elif strategy == 'few_shot':
        return PromptingStrategy.few_shot(instruction, input_text, config['examples'])
    elif strategy in ('cot', 'chain_of_thought'):
        return PromptingStrategy.chain_of_thought(instruction, input_text)
    elif strategy == 'concise':
        return PromptingStrategy.concise(instruction, input_text)

    # ── Scaling variants ──────────────────────────────────────────────────────
    elif strategy.startswith('concise_') and strategy.endswith('w'):
        # concise_Nw → word limit N
        word_limit = int(strategy[len('concise_'):-1])
        return PromptingStrategy.concise(instruction, input_text, word_limit=word_limit)

    elif strategy.startswith('few_shot_') and strategy[len('few_shot_'):].isdigit():
        # few_shot_N → N examples
        n = int(strategy[len('few_shot_'):])
        return PromptingStrategy.few_shot(instruction, input_text, config['examples'][:n])

    elif strategy.startswith('cot_') and strategy.endswith('step'):
        # cot_Nstep → N step starters
        steps = int(strategy[len('cot_'):-len('step')])
        return PromptingStrategy.cot_stepped(instruction, input_text, steps=steps)

    elif strategy == 'verbose_1':
        return PromptingStrategy.verbose_detailed(instruction, input_text)

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")
