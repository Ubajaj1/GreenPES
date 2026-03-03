"""
7-level prompt templates for the saturation experiment.

Each task has 7 templates of increasing length (additive components).
Level 1 is minimal; each subsequent level adds one meaningful component.
"""

NUM_LEVELS = 7

# Which key in the template maps to example['input']
TASK_INPUT_KEY = {
    'qa':                     'question',
    'summarization':          'text',
    'classification':         'text',
    'instruction_following':  'instruction',
}

SATURATION_TEMPLATES: dict[str, list[str]] = {

    # ── QA ────────────────────────────────────────────────────────────────────
    'qa': [
        # Level 1: bare question
        "{question}",

        # Level 2: + task label
        "Answer this question: {question}",

        # Level 3: + accuracy instruction
        "Answer this question accurately and concisely.\n\nQuestion: {question}",

        # Level 4: + format + uncertainty handling
        (
            "Answer this question accurately and concisely. "
            "Respond in 1-2 sentences. "
            "If you don't know the answer, say 'I don't know'.\n\n"
            "Question: {question}"
        ),

        # Level 5: + role persona
        (
            "You are a knowledgeable and precise assistant. "
            "Answer the following question accurately and concisely. "
            "Respond in 1-2 sentences. "
            "If you don't know the answer, say 'I don't know'. "
            "Stick to facts only.\n\n"
            "Question: {question}"
        ),

        # Level 6: + detailed guidelines
        (
            "You are a knowledgeable and precise assistant. "
            "Your task is to answer questions accurately and concisely.\n\n"
            "Guidelines:\n"
            "- Respond in 1-2 sentences only\n"
            "- Stick strictly to verifiable facts\n"
            "- Do not add opinions, caveats, or tangential information\n"
            "- If you are uncertain, say 'I don't know' rather than guessing\n"
            "- Do not repeat the question in your answer\n"
            "- Use simple, clear language\n\n"
            "Question: {question}"
        ),

        # Level 7: + 1 worked example
        (
            "You are a knowledgeable and precise assistant. "
            "Your task is to answer questions accurately and concisely.\n\n"
            "Guidelines:\n"
            "- Respond in 1-2 sentences only\n"
            "- Stick strictly to verifiable facts\n"
            "- Do not add opinions, caveats, or tangential information\n"
            "- If you are uncertain, say 'I don't know' rather than guessing\n"
            "- Do not repeat the question in your answer\n"
            "- Use simple, clear language\n\n"
            "Example:\n"
            "Question: What is the capital of France?\n"
            "Answer: Paris is the capital of France.\n\n"
            "Now answer:\n"
            "Question: {question}"
        ),
    ],

    # ── CLASSIFICATION ────────────────────────────────────────────────────────
    'classification': [
        # Level 1: bare
        "Classify: {text}",

        # Level 2: + class names
        "Classify sentiment as positive, negative, or neutral: {text}",

        # Level 3: + output format
        "Classify sentiment as positive, negative, or neutral. Respond with only the label: {text}",

        # Level 4: + label definitions
        (
            "Classify the sentiment of the following text as positive, negative, or neutral. "
            "Respond with only the label.\n\n"
            "Definitions:\n"
            "- positive: overall favorable or optimistic tone\n"
            "- negative: overall unfavorable or critical tone\n"
            "- neutral: balanced, factual, or no clear sentiment\n\n"
            "Text: {text}"
        ),

        # Level 5: + edge case handling
        (
            "Classify the sentiment of the following text as positive, negative, or neutral. "
            "Respond with only the label.\n\n"
            "Definitions:\n"
            "- positive: overall favorable or optimistic tone\n"
            "- negative: overall unfavorable or critical tone\n"
            "- neutral: balanced, factual, or no clear sentiment\n\n"
            "Edge cases: If the text contains mixed sentiment, choose the dominant tone. "
            "If equally mixed, use neutral.\n\n"
            "Text: {text}"
        ),

        # Level 6: + role + full guidelines
        (
            "You are a sentiment classification expert. "
            "Classify the sentiment of the following text as positive, negative, or neutral.\n\n"
            "Rules:\n"
            "1. Respond with ONLY one word: positive, negative, or neutral\n"
            "2. Base your judgment on the overall tone, not individual words\n"
            "3. Positive: clearly favorable, optimistic, praising, or satisfied\n"
            "4. Negative: clearly unfavorable, critical, pessimistic, or dissatisfied\n"
            "5. Neutral: factual reporting, balanced views, or no discernible sentiment\n"
            "6. If mixed, choose the dominant sentiment; if equal, use neutral\n"
            "7. Take text at face value; do not attempt sarcasm detection\n\n"
            "Text: {text}"
        ),

        # Level 7: + 1 worked example
        (
            "You are a sentiment classification expert. "
            "Classify the sentiment of the following text as positive, negative, or neutral.\n\n"
            "Rules:\n"
            "1. Respond with ONLY one word: positive, negative, or neutral\n"
            "2. Base your judgment on the overall tone, not individual words\n"
            "3. Positive: clearly favorable, optimistic, praising, or satisfied\n"
            "4. Negative: clearly unfavorable, critical, pessimistic, or dissatisfied\n"
            "5. Neutral: factual reporting, balanced views, or no discernible sentiment\n"
            "6. If mixed, choose the dominant sentiment; if equal, use neutral\n"
            "7. Take text at face value; do not attempt sarcasm detection\n\n"
            "Example:\n"
            "Text: The product works great and I'm very happy with my purchase.\n"
            "Label: positive\n\n"
            "Now classify:\n"
            "Text: {text}"
        ),
    ],

    # ── SUMMARIZATION ─────────────────────────────────────────────────────────
    'summarization': [
        # Level 1: bare
        "Summarize: {text}",

        # Level 2: + task label + text marker
        "Write a summary of the following text:\n\n{text}",

        # Level 3: + length constraint
        "Write a concise summary of the following text in 2-3 sentences:\n\n{text}",

        # Level 4: + content guidelines
        (
            "Write a concise summary of the following text in 2-3 sentences. "
            "Capture the main points, preserve key facts, and avoid personal opinions.\n\n"
            "Text: {text}"
        ),

        # Level 5: + role + output format
        (
            "You are a professional summarizer. "
            "Write a concise summary of the following text in 2-3 sentences.\n\n"
            "Requirements:\n"
            "- Capture the main idea and key supporting points\n"
            "- Preserve important facts and figures\n"
            "- Use neutral, factual language\n"
            "- Do not add opinions or outside information\n\n"
            "Text: {text}"
        ),

        # Level 6: + detailed inclusion/exclusion criteria
        (
            "You are a professional summarizer. "
            "Write a concise summary of the following text in exactly 2-3 sentences.\n\n"
            "Include:\n"
            "- Main claim or central argument\n"
            "- Key facts, figures, or evidence\n"
            "- Important conclusions or outcomes\n\n"
            "Exclude:\n"
            "- Minor details or supporting examples\n"
            "- Repetitive information\n"
            "- Your own opinions or interpretations\n"
            "- Information not present in the source text\n\n"
            "Format: Plain prose, 2-3 sentences, no bullet points.\n\n"
            "Text: {text}"
        ),

        # Level 7: + 1 worked example
        (
            "You are a professional summarizer. "
            "Write a concise summary of the following text in exactly 2-3 sentences.\n\n"
            "Include:\n"
            "- Main claim or central argument\n"
            "- Key facts, figures, or evidence\n"
            "- Important conclusions or outcomes\n\n"
            "Exclude:\n"
            "- Minor details or supporting examples\n"
            "- Repetitive information\n"
            "- Your own opinions or interpretations\n"
            "- Information not present in the source text\n\n"
            "Format: Plain prose, 2-3 sentences, no bullet points.\n\n"
            "Example:\n"
            "Text: Scientists at MIT have developed a new battery that charges a smartphone "
            "in under 5 minutes using a novel anode material that dramatically increases ion "
            "transfer speed. Early tests show the batteries retain 90% capacity after 1,000 "
            "charge cycles, far exceeding current lithium-ion performance.\n"
            "Summary: MIT researchers created a battery that charges smartphones in under "
            "5 minutes via a new anode material for faster ion transfer. The batteries "
            "maintain 90% capacity after 1,000 charge cycles.\n\n"
            "Now summarize:\n"
            "Text: {text}"
        ),
    ],

    # ── INSTRUCTION FOLLOWING ─────────────────────────────────────────────────
    'instruction_following': [
        # Level 1: bare instruction
        "{instruction}",

        # Level 2: + task framing
        "Follow this instruction carefully:\n\n{instruction}",

        # Level 3: + completeness requirement
        (
            "Follow this instruction carefully and completely. "
            "Satisfy every requirement stated.\n\n"
            "Instruction: {instruction}"
        ),

        # Level 4: + constraint adherence
        (
            "Follow this instruction carefully and completely. "
            "Satisfy every requirement stated. "
            "Do not add content that was not requested. "
            "Do not skip any stated constraints.\n\n"
            "Instruction: {instruction}"
        ),

        # Level 5: + role + output format rule
        (
            "You are a precise instruction-following assistant. "
            "Follow the instruction below carefully and completely.\n\n"
            "Rules:\n"
            "- Satisfy every requirement and constraint stated\n"
            "- Do not add unrequested content\n"
            "- Do not omit any required element\n"
            "- Format your output exactly as specified in the instruction\n\n"
            "Instruction: {instruction}"
        ),

        # Level 6: + verification checklist
        (
            "You are a precise instruction-following assistant. "
            "Follow the instruction below carefully and completely.\n\n"
            "Rules:\n"
            "- Read the full instruction before responding\n"
            "- Satisfy every requirement and constraint stated\n"
            "- Do not add content that was not requested\n"
            "- Do not omit any required element\n"
            "- If the instruction specifies length, word count, or format, follow it exactly\n"
            "- After drafting your response, verify each stated constraint is satisfied\n\n"
            "Instruction: {instruction}"
        ),

        # Level 7: + 1 worked example
        (
            "You are a precise instruction-following assistant. "
            "Follow the instruction below carefully and completely.\n\n"
            "Rules:\n"
            "- Read the full instruction before responding\n"
            "- Satisfy every requirement and constraint stated\n"
            "- Do not add content that was not requested\n"
            "- Do not omit any required element\n"
            "- If the instruction specifies length, word count, or format, follow it exactly\n"
            "- After drafting your response, verify each stated constraint is satisfied\n\n"
            "Example:\n"
            "Instruction: Write a 3-word sentence about the ocean using alliteration.\n"
            "Response: Waves wash wonderfully.\n\n"
            "Now follow:\n"
            "Instruction: {instruction}"
        ),
    ],
}


def format_prompt(template: str, task: str, example: dict) -> str:
    """Replace the task-specific placeholder with example['input']."""
    placeholder = '{' + TASK_INPUT_KEY[task] + '}'
    return template.replace(placeholder, example['input'])
