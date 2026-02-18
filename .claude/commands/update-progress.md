Update the PROGRESS.md file to reflect the current state of implementation.

Steps:
1. Read the current PROGRESS.md
2. Check which files exist and what's implemented by reading the relevant source files:
   - `greenprompt/llm.py` — check which provider classes exist
   - `greenprompt/evaluators.py` — check which evaluator classes exist
   - `experiments/prompting_strategies.py` — check TASK_CONFIGS and BENCHMARK_EXAMPLES keys and example counts
   - `experiments/benchmark.py` — check if MODEL_CONFIGS and get_provider() exist
   - `experiments/analysis.py` — check if file exists
   - `tests/test_evaluators.py` — check if file exists
3. Mark tasks as [x] (complete) or [ ] (pending) based on what actually exists in code
4. Update the "Last updated" date to today's date (2026-02-17 or current date)
5. Update the "API Keys Status" table if you have information about which keys are set
6. Write the updated PROGRESS.md
7. Do NOT commit — just update the file. Tell the user what changed.
