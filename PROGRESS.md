# GreenPES Implementation Progress

> **For Claude:** Read this file at the start of every session to know where to resume.

**Goal:** 560-experiment benchmark (7 models × 4 tasks × 5 strategies × 4 examples) for COLM 2026 (deadline: Mar 31)

---

## Phase 1: LLM Providers ✅ COMPLETE

- [x] Task 1.1: OpenAI Provider — `gpt-4o-mini` ✅ tested
- [x] Task 1.2: Anthropic Provider — `claude-haiku-4-5-20251001` ✅ tested (updated from deprecated claude-3-5-haiku-20241022)
- [x] Task 1.3: Together Provider — ❌ DROPPED (replaced by Groq Qwen + Kimi)
- [x] Task 1.4: Groq models — `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `qwen/qwen3-32b`, `moonshotai/kimi-k2-instruct` ✅ all tested
- [x] Task 1.5: Gemini — `gemini-2.0-flash` ✅ tested; `gemini-2.5-pro` ⏭ skipped (free tier limit=0, billing not enabled — Option A chosen)

**Final 7 models (tests passing 7/7):**

| # | Model | Lab | Provider |
|---|-------|-----|----------|
| 1 | llama-3.1-8b-instant | Meta | Groq (free) |
| 2 | llama-3.3-70b-versatile | Meta | Groq (free) |
| 3 | qwen/qwen3-32b | Alibaba | Groq (free) |
| 4 | moonshotai/kimi-k2-instruct | Moonshot AI | Groq (free) |
| 5 | gpt-4o-mini | OpenAI | OpenAI (paid) |
| 6 | claude-haiku-4-5-20251001 | Anthropic | Anthropic (paid) |
| 7 | gemini-2.0-flash | Google | Gemini (free) |

**Model changes from original plan:**
- Replaced Together.AI (Qwen via Together) → Groq Qwen3-32B (free)
- Replaced Mixtral-8x7b (decommissioned) → moonshotai/kimi-k2-instruct
- Dropped gemini-2.5-pro (no free tier) → 7 models total
- Updated Anthropic model to claude-haiku-4-5-20251001 (claude-3-5-haiku EOL Feb 19 2026)

---

## Phase 2: Task Evaluators ✅ COMPLETE

- [x] Task 2.1: `ClassificationEvaluator` in `greenprompt/evaluators.py` + `tests/test_evaluators.py`
- [x] Task 2.2: `InstructionFollowingEvaluator` in `greenprompt/evaluators.py` + tests
- [x] Update `get_evaluator()` to include new task types
- [x] **Bonus:** Revised `QAEvaluator` — replaced `SequenceMatcher` with word-overlap (ROUGE-1 recall), added negation check, fixed `completed` flag for hedging responses
- [x] **Bonus:** Revised `SummarizationEvaluator` — added ROUGE-1 scoring when `ground_truth` provided, removed dead `tolerance` parameter

**Test suite:** `tests/test_evaluators.py` — 36 tests, all passing.

---

## Phase 3: Benchmark Data ✅ COMPLETE

- [x] Task 3.1: `classification` task config + 20 examples in `experiments/prompting_strategies.py`
- [x] Task 3.2: `instruction_following` task config + 20 examples (with `constraints` field)
- [x] Task 3.3: `qa` expanded to 20 examples; `summarization` expanded to 20 examples
- [x] **Bonus:** All 20 summarization examples now have `ground_truth` reference summaries (activates ROUGE-1)
- [x] **Bonus:** All 20 QA examples now have `ground_truth` strings

---

## Phase 4: Benchmark Runner ✅ COMPLETE

- [x] Task 4.1: `experiments/benchmark.py` with all 7 models, full argparse, incremental saving + `--resume`

---

## Phase 5: Analysis Scripts ✅ COMPLETE

- [x] Task 5.1: `experiments/analysis.py` — RQ1–RQ8 statistical analysis + 10 figures + `stats_summary.csv`
- [x] Task 5.2: `experiments/optimizer_benchmark.py` — optimizer vs baseline compressor benchmark, with incremental saving + `--resume`
- [x] Task 5.3: `greenprompt/optimizer.py` — `PromptOptimizer`, `BaselineCompressor`, `OptimizationResult`

**Test suite:** 136 tests passing, 2 skipped.

---

## Phase 6: Environment ✅ COMPLETE

- [x] Task 6.1: `.env.example` updated with all provider keys + setup instructions
- [x] All 5 API keys set and tested in `.env`

---

## Benchmark Runs

### Run 1: Main Benchmark (LLM Judge) — `results/benchmark_judge_full.json`
- **4032 successful** / 4517 total records (485 errors)
- 7 models × 4 tasks × 5 strategies × 30 examples per task
- All records include `judge_scores` (4-dimension LLM judge: correctness, completeness, reasoning, conciseness)
- gemini-flash: 439/600 (quota exhausted); llama-3.3-70b: 593/600 (7 rate-limit errors)

| Model | Successful |
|-------|-----------|
| claude-haiku | 600/600 |
| gpt-4o-mini | 600/600 |
| kimi-k2 | 600/600 |
| llama-3.1-8b | 600/600 |
| qwen3-32b | 600/600 |
| llama-3.3-70b | 593/600 |
| gemini-flash | 439/600 |

### Run 2: Optimizer Benchmark — `results/optimizer_results.json`
- **Status:** ✅ STOPPED INTENTIONALLY (2026-02-27)
- **1,559 successful records** across 3 models (llama-3.1-8b complete, llama-3.3-70b + qwen3-32b partial)
- Stopped early: LLM optimizer made ~9 sequential API calls per example (3 rewrite iterations), taking ~10h per model. llama-3.1-8b (600 records, all tasks/methods complete) is sufficient for RQ8 paired t-tests.
- llama-3.3-70b had 126 errors (Groq daily token quota exhausted mid-run)

| Model | Successful | Notes |
|-------|-----------|-------|
| llama-3.1-8b | 600/600 | ✅ Complete — all 4 tasks, all 5 methods |
| llama-3.3-70b | 475 | ⚠️ Quota errors; QA+summ complete |
| qwen3-32b | 484 | 🔄 Partial; QA+summ+class complete |

---

## Analysis Results — `results/stats_summary.csv` + `results/figures/`

Run on `benchmark_judge_full.json` (4032 records). Figures saved Feb 26 2026.

### RQ1: Strategy Effect on GreenPES
- ANOVA: F=105.98, p<0.0001, η²=0.095
- `concise` ≈ `zero_shot` (best, no significant difference p=0.80)
- `cot` and `few_shot` significantly worse than `concise` and `zero_shot`

### RQ2: Token Efficiency by Task
- QA: `zero_shot` best (GreenPES=11.49, tokens=121)
- Summarization, Classification, Instruction-following: `concise` best

### RQ3: Model Comparison (mean GreenPES)
- gemini-flash (8.36) > gpt-4o-mini (6.94) > kimi-k2 (6.69) > claude-haiku (4.71) > llama-3.3-70b (4.64) > llama-3.1-8b (4.11) > qwen3-32b (1.79)

### RQ4: Quality-Efficiency Tradeoff
- Pearson r=0.169, p<0.0001 (positive correlation: more tokens → slightly better quality)

### RQ5: Strategy Transfer
- Interaction (model×strategy): F=1.865, p=0.0064
- Most models prefer `cot` for raw quality (but `cot` is expensive in tokens)
- `cot` is best for: claude-haiku, gemini-flash, gpt-4o-mini, kimi-k2
- `few_shot` best for llama-3.1-8b; `zero_shot_verbose` best for llama-3.3-70b; `concise` best for qwen3-32b

### RQ6: Model Strategy Agreement
- Universality index = 0.000 (no model pair has Kendall tau > 0.8)
- Models disagree significantly on optimal strategy

### RQ7: Scaling Laws
- 28 (model, task) pairs fitted; most best fit: logarithmic
- Saturation points: QA ~50-280 tokens; Summarization ~180-420; Classification ~80-290; Instruction-following ~80-280

### RQ8: Optimizer Effectiveness (1,559 optimizer records)
- LLM optimizer vs remove_filler: t=3.897, p=0.0001, mean_diff=+0.032 ✓
- LLM optimizer vs truncate_examples: t=11.031, p<0.0001, mean_diff=+0.202 ✓
- LLM optimizer vs add_concise_suffix: t=-0.510, p=0.61, mean_diff=-0.004 ✗ (not significant)
- Mean compression ratios: llm_optimizer=1.595, truncate_examples=1.501, add_concise_suffix=1.282, remove_filler=1.021
- Mean quality retained: add_concise_suffix=1.036, llm_optimizer=1.034, remove_filler=1.001, truncate_examples=0.838

---

**Figures generated:**
- `fig1_strategy_heatmap.png` — RQ1: GreenPES heatmap by task × strategy
- `fig2_token_efficiency.png` — RQ2: token efficiency by strategy × task
- `fig3_model_comparison.png` — RQ3: mean GreenPES by model (ordered by size)
- `fig4_quality_efficiency_scatter.png` — RQ4: quality vs token cost scatter
- `fig5_transfer_heatmap.png` — RQ5: 7×7 strategy transfer matrix
- `fig6_interaction_plot.png` — RQ6: model×strategy interaction lines
- `fig7_scaling_curves.png` — RQ7: scaling curves (4 subplots, one per task)
- `fig8_saturation_points.png` — RQ7: saturation token count per model×task

---

## API Keys Status

| Provider | Key Set | Models Active |
|----------|---------|---------------|
| Groq | ✅ | llama-3.1-8b, llama-3.3-70b, qwen3-32b, kimi-k2 |
| Gemini | ✅ (`GEMINI_API_KEY`) | gemini-2.0-flash (daily quota exhausts with heavy use) |
| OpenAI | ✅ | gpt-4o-mini |
| Anthropic | ✅ | claude-haiku-4-5-20251001 |
| Together | ❌ Not used | Dropped from plan |

---

## Infrastructure Setup ✅ COMPLETE

- [x] `.gitignore` updated — `docs/plans/`, `greenpes_implementation_plan.md`, `.claude/settings.local.json` all protected
- [x] `.claude/settings.json` — SessionStart hook loads this file automatically each session
- [x] `.claude/commands/update-progress.md` — `/update-progress` slash command
- [x] `.claude/commands/end-session.md` — `/end-session` slash command for safe commit + push

---

## Next Steps (in order)

1. **Write paper** — use `results/stats_summary.csv` + `results/figures/` for COLM 2026 (deadline: Mar 31) **← START HERE**
   - All 10 figures available: fig1–fig8 (RQ1–RQ7) + fig9–fig10 (RQ8)
   - All stats in `results/stats_summary.csv`

---

*Last updated: 2026-02-26*
