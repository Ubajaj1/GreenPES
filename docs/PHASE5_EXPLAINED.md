# Phase 5 Explained — From Scratch, for Beginners

This document explains everything built in Phase 5 in plain English.
No prior coding knowledge assumed.

---

## The Big Picture: What is Phase 5?

By the end of Phase 4, we had a benchmark runner that sent prompts to 6 AI models
and recorded the results in a file called `results/benchmark_results.json`.
That file contains 480 rows — one row per experiment.

**Phase 5 answers the question: "What do those 480 rows actually tell us?"**

We wrote a script (`experiments/analysis.py`) that:
1. Reads the 480 rows
2. Runs statistics to answer 4 research questions
3. Produces 4 charts (figures)
4. Saves everything to a CSV and PNG files

---

## What is a Research Question (RQ)?

A research question is a specific thing we want to find out.
We had 4:

| # | Question | Plain English |
|---|----------|---------------|
| RQ1 | Does prompting strategy significantly affect GreenPES? | Does the *way you write your prompt* change how efficient the AI is? |
| RQ2 | Which strategy is most token-efficient per task? | For each type of task (QA, summarization, etc.), which prompt style uses the fewest tokens while still getting good answers? |
| RQ3 | Do smaller models achieve competitive GreenPES vs larger ones? | Does a cheap/small AI do as well as an expensive/large one on our efficiency score? |
| RQ4 | Is there a quality–efficiency tradeoff? | If you use fewer tokens, does quality go down? Is there a tension between "cheap" and "good"? |

---

## The Data: What is benchmark_results.json?

Think of it as a giant spreadsheet with 480 rows. Each row is one experiment:

```
{
  "model":          "gpt-4o-mini",      ← which AI was used
  "task":           "qa",               ← what type of question was asked
  "strategy":       "concise",          ← how the prompt was written
  "example_id":     1,                  ← which specific question (1–4)
  "greenpes":       33.90,              ← our efficiency score (higher = better)
  "quality":        1.0,                ← how good the answer was (0–1)
  "input_tokens":   15,                 ← words sent TO the AI
  "output_tokens":  11,                 ← words the AI sent BACK
  "total_tokens":   26,                 ← input + output
  ...
}
```

**GreenPES** is the key metric. It rewards AIs that give good answers using few tokens.
Formula: `(quality × task_completed) / (input_tokens + 1.5 × output_tokens) × 1000`

---

## The Script: experiments/analysis.py

The script has 7 main pieces:

```
analysis.py
├── Constants          — lists of model names, task names, strategy names
├── load_and_clean()   — reads the JSON file and checks it's valid
├── rq1_strategy_effect()    — answers RQ1
├── rq2_token_efficiency()   — answers RQ2
├── rq3_model_comparison()   — answers RQ3
├── rq4_quality_tradeoff()   — answers RQ4
├── save_stats_csv()   — saves all statistics to a CSV file
├── save_figures()     — saves all 4 charts as PNG images
└── main()             — the entry point; calls everything in order
```

---

## Piece 1: Constants

```python
STRATEGIES = ['zero_shot', 'zero_shot_verbose', 'few_shot', 'cot', 'concise']
TASKS = ['qa', 'summarization', 'classification', 'instruction_following']
MODEL_ORDER = ['llama-3.1-8b', 'gemini-flash', ..., 'gpt-4o-mini', 'claude-haiku']
```

These are just fixed lists used throughout the script.
`MODEL_ORDER` goes from smallest/cheapest model to largest/most capable — used to
sort the model comparison chart so small models appear first.

---

## Piece 2: load_and_clean()

**What it does:** Reads the JSON file and prepares a table (called a DataFrame).

**Step by step:**
1. Checks the file exists — if not, crashes with a helpful message
2. Reads all 480 rows
3. Removes any rows that have an `"error"` field (failed API calls)
4. Checks that required columns exist (model, task, strategy, greenpes, etc.)
5. Prints a summary of what was loaded

**Why we need it:** Raw data from benchmark runs can have failures.
We can't do statistics on broken data, so we clean it first.

**Real output when run:**
```
Data summary (480 successful runs):
  models (6): claude-haiku, gpt-4o-mini, kimi-k2, llama-3.1-8b, llama-3.3-70b, qwen3-32b
  tasks (4): classification, instruction_following, qa, summarization
  strategys (5): concise, cot, few_shot, zero_shot, zero_shot_verbose
```

---

## Piece 3: rq1_strategy_effect() — Does prompt style matter?

**What it does:** Uses statistics to check if different prompt strategies give
significantly different GreenPES scores.

### Step 1: One-way ANOVA

ANOVA is a statistical test. Think of it like this:

> "I have 5 groups of numbers (one per strategy). Are the group averages
> different enough that it's probably NOT just random luck?"

- If the answer is YES → we say the difference is "statistically significant"
- We measure this with a **p-value**. If p < 0.05, it's significant.

**Our result:** F=16.15, p<0.0001 → YES, strategy significantly affects GreenPES.

### Step 2: Tukey HSD (Post-hoc test)

ANOVA only tells us "at least two strategies differ." It doesn't say WHICH ones.
Tukey HSD compares every pair:

```
✓ zero_shot vs few_shot: p=0.0000       ← significantly different
✗ zero_shot vs concise:  p=0.9999       ← NOT significantly different
```

A `✓` means those two strategies are genuinely different.
A `✗` means we can't tell them apart statistically.

### Step 3: Effect size (η² = eta-squared)

p-value tells you "is it real?" but not "how big is the effect?"
Eta-squared answers: "what fraction of the variation in GreenPES is explained by strategy?"
Our η²=0.12 means strategy explains 12% of the variation. That's a medium effect.

### Figure produced: Heatmap (fig1_strategy_heatmap.png)

A heatmap is a grid where each cell's colour shows a number.
Rows = strategies, columns = tasks, colour = mean GreenPES.
Red = low efficiency, Green = high efficiency.

---

## Piece 4: rq2_token_efficiency() — Which strategy wins per task?

**What it does:** For each task type, finds which strategy achieves the highest
average GreenPES and how many tokens it used.

**No statistical test here** — this is just descriptive: "who won?"

**Real output:**
```
qa:                    best = zero_shot  (GreenPES=14.22, tokens=104)
summarization:         best = concise    (GreenPES=3.06,  tokens=188)
classification:        best = concise    (GreenPES=9.19,  tokens=131)
instruction_following: best = concise    (GreenPES=7.55,  tokens=141)
```

For QA tasks, a simple zero-shot prompt is best.
For everything else, asking the AI to be concise wins.

### Figure produced: Grouped bar chart (fig4_greenpes_distribution.png)

Each group of bars = one task. Each bar within the group = one strategy.
Bar height = mean GreenPES. Error bars show how much the scores varied.

---

## Piece 5: rq3_model_comparison() — Do small models keep up?

**What it does:** Computes the average GreenPES for each of the 6 models,
then shows them ordered from smallest to largest model.

**No statistical test here either** — we're just comparing averages.

**Real output:**
```
llama-3.1-8b:  mean=4.57  (small model)
qwen3-32b:     mean=1.57  (medium, but surprisingly bad!)
llama-3.3-70b: mean=5.67
kimi-k2:       mean=7.93
gpt-4o-mini:   mean=9.48  (best!)
claude-haiku:  mean=5.43
```

Interesting finding: gpt-4o-mini is the most efficient even though it's not the
largest — it gives concise, high-quality answers.
qwen3-32b scores lowest despite being a large model.

### Figure produced: Horizontal bar chart (fig2_model_comparison.png)

One horizontal bar per model. Bar length = mean GreenPES. Error bars = std deviation.
Models are listed top-to-bottom from small → large.

---

## Piece 6: rq4_quality_tradeoff() — Does efficiency hurt quality?

**What it does:** Checks if using fewer tokens is associated with lower quality answers.

### Step 1: Pearson r (correlation coefficient)

Pearson r measures the relationship between two numbers:
- r = +1.0 → when tokens go up, quality goes up perfectly
- r = -1.0 → when tokens go up, quality goes DOWN perfectly
- r = 0.0  → no relationship at all

**Our result:** r = -0.316 → there IS a mild negative correlation.
More tokens = slightly higher quality on average. But it's not a strong effect.

### Step 2: Pareto frontier

Imagine plotting every (strategy, task) combination on a graph:
- X axis = number of tokens used
- Y axis = quality score

The "Pareto frontier" is the line connecting the points where:
"No other strategy gives better quality at the same or lower token cost."

These are the "best value" strategies — good quality without overspending tokens.

### Figure produced: Scatter plot (fig3_quality_efficiency_scatter.png)

Each dot = one (strategy, task) combination.
X = mean tokens, Y = mean quality.
Color = which strategy. Shape = which task.
Dashed line = Pareto frontier (the "efficient" points).

---

## Piece 7: save_stats_csv() and save_figures()

**save_stats_csv:** Takes all the statistics collected from RQ1–RQ4
and writes them to `results/stats_summary.csv` — a spreadsheet you can open in Excel.

**save_figures:** Takes all 4 charts and saves them as high-resolution PNG images
(300 DPI) in `results/figures/`. DPI = dots per inch. 300 DPI is publication quality.

---

## Piece 8: main() — The conductor

`main()` is what runs when you type `python experiments/analysis.py`.
It calls everything else in order:

```
1. Parse command-line arguments (--input, --output-dir)
2. load_and_clean()          → get the data
3. rq1_strategy_effect()     → get RQ1 figure + stats
4. rq2_token_efficiency()    → get RQ2 figure + stats
5. rq3_model_comparison()    → get RQ3 figure + stats
6. rq4_quality_tradeoff()    → get RQ4 figure + stats
7. save_stats_csv()          → write stats_summary.csv
8. save_figures()            → write 4 PNG files
```

---

## The Tests: tests/test_analysis.py

Tests are code that checks other code works correctly.
Instead of manually running the script and looking at the output,
tests do it automatically and tell you immediately if something broke.

### Why tests matter

Imagine you change one line in `rq1_strategy_effect()`. How do you know
you didn't accidentally break it? You run the tests — they check in seconds.

### How tests work in Python (pytest)

```python
def test_something():
    result = my_function(input)
    assert result == expected_output   # if this fails, the test fails
```

`assert` means "I'm claiming this is true — if it's not, something is broken."

---

### Helper functions (not tests themselves)

**make_synthetic_results()**

Instead of using the real 480-row benchmark file in tests (slow, hard to control),
we create fake data that looks exactly like real data but is tiny (16–32 rows).

```python
make_synthetic_results(n_models=2, n_tasks=2, n_strategies=2, n_examples=4)
# Creates: 2 models × 2 tasks × 2 strategies × 4 examples = 32 rows
```

This way tests run in milliseconds, not minutes.

**write_json()**

Saves the fake data to a temporary file so `load_and_clean()` can read it.
Temporary files are deleted automatically by the operating system.

---

### TestLoadAndClean (4 tests)

Tests for the `load_and_clean()` function.

| Test | What it checks |
|------|---------------|
| `test_returns_dataframe` | Does it return a pandas DataFrame (table)? |
| `test_drops_error_records` | If one row has an "error" field, does it get removed? |
| `test_required_columns_present` | Are all 8 required columns in the output? |
| `test_raises_on_missing_file` | Does it give a clear error if the file doesn't exist? |

---

### TestRQ1 (3 tests)

| Test | What it checks |
|------|---------------|
| `test_returns_figure_and_stats` | Does it return (Figure, list)? Is the list non-empty? |
| `test_stats_have_required_keys` | Does each stats row have 'rq', 'test', 'statistic', 'p_value'? |
| `test_stats_rq_label` | Is every row labelled 'RQ1' (not 'RQ2' by accident)? |

---

### TestRQ2 (3 tests)

| Test | What it checks |
|------|---------------|
| `test_returns_figure_and_stats` | Returns (Figure, list)? |
| `test_stats_rq_label` | Every row labelled 'RQ2'? |
| `test_one_winner_per_task` | Exactly one "winner" row per task in the stats? |

The last test is important: for each task we should declare exactly ONE best strategy.
If there are 2 tasks and 3 winner rows, something is wrong.

---

### TestRQ3 (4 tests)

| Test | What it checks |
|------|---------------|
| `test_returns_figure_and_stats` | Returns (Figure, list)? |
| `test_one_stat_row_per_model` | One stats row per model — not more, not fewer? |
| `test_stats_rq_label` | Every row labelled 'RQ3'? |
| `test_stats_have_required_keys` | All 7 keys present? Is `effect_metric == 'std'`? |

---

### TestRQ4 (5 tests)

| Test | What it checks |
|------|---------------|
| `test_returns_figure_and_stats` | Returns (Figure, list)? |
| `test_stats_contain_pearson_r` | Is there exactly 1 row with `test='Pearson r'`? Is r between -1 and 1? |
| `test_stats_rq_label` | Every row labelled 'RQ4'? |
| `test_stats_have_required_keys` | All 7 keys present? |
| `test_handles_constant_quality` | If all quality values are 1.0 (no variation), does it crash? It shouldn't! |

The last test is a **robustness test** — it tests an edge case that could break the code.
`scipy.stats.pearsonr` crashes if you pass it data with zero variation. We added
a guard to handle this, and this test verifies the guard works.

---

### TestEndToEnd (1 test)

This is the most important test. It runs the **entire pipeline** as if you typed
`python experiments/analysis.py` yourself.

```
1. Creates fake data (32 rows)
2. Saves it to a temp folder
3. Runs analysis.py as a separate process (like a real user would)
4. Checks:
   - Did it exit successfully? (returncode == 0)
   - Does stats_summary.csv exist?
   - Does it have rows?
   - Does it contain data for all 4 RQs?
   - Do all 4 PNG files exist?
   - Are they non-empty files (not zero bytes)?
```

The other tests check individual functions. This test checks that everything
works TOGETHER, end-to-end. It catches bugs that only appear when functions
are combined.

---

## The setup_method pattern

Many test classes have this:

```python
class TestRQ1:
    def setup_method(self):
        records = make_synthetic_results(...)
        path = write_json(records)
        self.df = load_and_clean(path)
```

`setup_method` runs automatically before EACH test in the class.
This means every test starts with a fresh, clean copy of the data.
Tests don't accidentally affect each other.

---

## What the `# type: ignore` comments mean

You'll see things like:

```python
groups = [df[df['strategy'] == s]['greenpes'].to_numpy() for s in strategies]  # type: ignore[union-attr]
```

Python has a tool called Pyright that checks code for type errors before running.
Sometimes Pyright is overly strict about pandas (the data table library) and
reports a warning even when the code is correct.

`# type: ignore[...]` tells Pyright: "I know what I'm doing here, don't warn me."
It has NO effect on how the code runs — it's only a note for the type checker.

---

## What the libraries do

| Library | What it does |
|---------|-------------|
| `pandas` | Works with tables (DataFrames). Used everywhere for filtering, grouping, averaging data. |
| `matplotlib` | Makes charts. The low-level drawing tool. |
| `seaborn` | Makes prettier charts on top of matplotlib. Used for the heatmap. |
| `scipy.stats` | Statistical tests. Provides ANOVA, Tukey HSD, and Pearson r. |
| `argparse` | Lets you pass `--input` and `--output-dir` on the command line. |
| `pathlib.Path` | A nicer way to work with file paths (instead of string concatenation). |

---

## Final summary

```
480 benchmark rows
        │
        ▼
load_and_clean()     → 480-row pandas DataFrame
        │
        ├──▶ rq1_strategy_effect()   → heatmap + ANOVA stats
        ├──▶ rq2_token_efficiency()  → bar chart + winner-per-task stats
        ├──▶ rq3_model_comparison()  → horizontal bar chart + per-model stats
        └──▶ rq4_quality_tradeoff()  → scatter plot + Pearson r stat
                    │
                    ▼
        save_stats_csv()   → results/stats_summary.csv
        save_figures()     → results/figures/*.png
```

**20 tests** verify every step of this pipeline works correctly,
from loading a single file all the way to the full end-to-end run.
