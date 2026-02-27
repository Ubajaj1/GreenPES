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

### Run 2: Optimizer Benchmark v1 — `results/optimizer_results.json` (SUPERSEDED)
- **Status:** ✅ STOPPED INTENTIONALLY (2026-02-27)
- **1,559 successful records** — used old (naive) optimizer; superseded by v2 with improved optimizer
- Old optimizer flaw: ran 3 generic templates sequentially, no compression gate, add_brevity duplicated baseline

### Run 3: Optimizer Benchmark v2 — `results/optimizer_results_v2.json` (IMPROVED OPTIMIZER)
- **Status:** 🔄 Mini-run complete (n=22 per method); full run pending (Groq RPD limit hit)
- **Improved optimizer**: 3 aggressive task-aware templates, best-of-K selection, 5% compression gate, early stopping
- Mini-run results (llama-3.1-8b, 4 tasks, 3 verbose strategies, ~3 examples each):

| Method | Mean Compression | Quality Retained |
|--------|-----------------|-----------------|
| remove_filler | 1.05x | 0.945 |
| add_concise_suffix | 1.21x | 1.027 |
| truncate_examples | 2.02x | 0.752 |
| **llm_optimizer (improved)** | **1.50x** | **1.033** |

- **LLM optimizer vs add_concise_suffix**: 24% better compression (1.50x vs 1.21x), similar quality retention
- **Full run needed**: `python experiments/optimizer_benchmark.py --models llama-3.1-8b --examples 10 --delay 2.5 --output results/optimizer_results_v2.json --resume` (run after Groq quota resets)

---

## Analysis Results — `results/stats_summary.csv` + `results/figures/`

Run on `benchmark_judge_full.json` (3604 records after qwen3-32b fix). Updated 2026-02-27.

**⚠️ Note**: qwen3-32b re-run in progress (173/600 done); full re-analysis needed once complete.

### RQ1: Strategy Effect on GreenPES (UPDATED)
- ANOVA: F=127.46, p<0.0001, η²=0.124 (stronger than before — qwen3-32b fix)
- `concise` ≈ `zero_shot` (best, no significant difference p=0.85)
- `cot` and `few_shot` significantly worse than `concise` and `zero_shot`

### RQ2: Token Efficiency by Task (UPDATED)
- QA: `zero_shot` best (GreenPES=13.01, tokens=89)
- Summarization, Classification, Instruction-following: `concise` best

### RQ3: Model Comparison (PRELIMINARY — qwen3-32b at n=173)
- gemini-flash (8.36) > **qwen3-32b (7.90)** > gpt-4o-mini (6.94) > kimi-k2 (6.69) > claude-haiku (4.71) > llama-3.3-70b (4.64) > llama-3.1-8b (4.11)
- **KEY CHANGE**: qwen3-32b jumped from last (1.79) to 2nd — was caused by undocumented thinking mode inflating output tokens

### RQ4: Quality-Efficiency Tradeoff (UPDATED)
- Pearson r=0.235, p<0.0001 (stronger than before: was r=0.169)

### RQ5: Strategy Transfer (stable)
- Interaction (model×strategy): F=1.867, p=0.0064
- `cot` is best for: claude-haiku, gemini-flash, gpt-4o-mini, kimi-k2
- `few_shot` best for llama-3.1-8b; `zero_shot_verbose` best for llama-3.3-70b, qwen3-32b

### RQ6: Model Strategy Agreement (stable)
- Universality index = 0.000 (no model pair has Kendall tau > 0.8)

### RQ7: Scaling Laws (IMPROVED — individual records, not averages)
- 26 (model, task) pairs fitted using ~150 individual records each (was 5 averaged points)
- Logarithmic fit best for nearly all pairs; sigmoid for instruction_following
- Saturation points: QA ~20-55 tokens; Summarization ~116-145; Classification ~39-109; Instruction-following ~126-310

### RQ8: Optimizer Effectiveness (SUPERSEDED; v2 pending full run)
- Old v1: LLM optimizer vs add_concise_suffix was NOT significant (p=0.61)
- New v2 (improved optimizer): 1.50x vs 1.21x compression — 24% better, quality similar
- Full run needed for paired t-test statistics

### RQ9: Metric Robustness (NEW)
- Heuristic vs LLM judge quality: Pearson r=0.985 (validates simple evaluator)
- Per-strategy: r≥0.980 across all 5 strategies
- α sensitivity: concise=rank1 at α≥1.5; zero_shot=rank1 at α=1.0 (only top-2 flip)
- Rankings of cot (rank 5) and few_shot/zero_shot_verbose stable across all α values

---

**Figures generated (2026-02-27):**
- `fig1_strategy_heatmap.png` — RQ1: GreenPES heatmap by task × strategy
- `fig2_token_efficiency.png` — RQ2: token efficiency by strategy × task
- `fig3_model_comparison.png` — RQ3: mean GreenPES by model (ordered by size)
- `fig4_quality_efficiency_scatter.png` — RQ4: quality vs token cost scatter
- `fig5_transfer_heatmap.png` — RQ5: 7×7 strategy transfer matrix
- `fig6_interaction_plot.png` — RQ6: model×strategy interaction lines
- `fig7_scaling_curves.png` — RQ7: scaling curves (4 subplots, 150 pts/curve)
- `fig8_saturation_points.png` — RQ7: saturation token count per model×task
- `fig9_compression_scatter.png` — RQ8: optimizer compression vs quality scatter
- `fig10_compression_bars.png` — RQ8: compression bars by original strategy
- `fig11_quality_signal_comparison.png` — RQ9: heuristic vs judge quality (r=0.985)
- `fig12_alpha_sensitivity.png` — RQ9: strategy rankings across α values

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

1. **Complete qwen3-32b re-run** — 427 records remaining; run tomorrow when Groq RPD quota resets:
   ```
   python experiments/benchmark.py --models qwen3-32b --examples 30 --evaluator llm_judge \
     --output results/benchmark_judge_full.json --delay 2.0 --resume
   ```

2. **Run full optimizer benchmark v2** — needs Groq quota; run after qwen3-32b completes:
   ```
   python experiments/optimizer_benchmark.py --models llama-3.1-8b --examples 10 \
     --delay 2.5 --output results/optimizer_results_v2.json
   ```

3. **Final full analysis** — after both runs above complete:
   ```
   python experiments/analysis.py --rqs all \
     --input results/benchmark_judge_full.json \
     --optimizer-input results/optimizer_results_v2.json
   ```

4. **Write paper** — COLM 2026 deadline: Mar 31

---

*Last updated: 2026-02-27*
