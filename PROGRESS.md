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

**Evaluator details:**

| Evaluator | Method | Notes |
|-----------|--------|-------|
| `QAEvaluator` | Word-overlap (ROUGE-1 recall) | Negation check: "not [answer]" → 0.0; punctuation-stripped |
| `SummarizationEvaluator` | Length + coherence + ROUGE-1 | ROUGE-1 active when `ground_truth` provided (weight 50%) |
| `ClassificationEvaluator` | Label containment (case-insensitive) | Negation check; no-ground-truth → 0.7 heuristic |
| `InstructionFollowingEvaluator` | Constraint fraction | Constraints: `bullet_points`, `numbered_list`, `single_word`; partial credit |

---

## Phase 3: Benchmark Data ✅ COMPLETE

- [x] Task 3.1: `classification` task config + 20 examples in `experiments/prompting_strategies.py`
- [x] Task 3.2: `instruction_following` task config + 20 examples (with `constraints` field)
- [x] Task 3.3: `qa` expanded to 20 examples; `summarization` expanded to 20 examples
- [x] **Bonus:** All 20 summarization examples now have `ground_truth` reference summaries (activates ROUGE-1)
- [x] **Bonus:** All 20 QA examples now have `ground_truth` strings

**Current counts:**

| Task | Examples | ground_truth | Notes |
|------|----------|-------------|-------|
| `qa` | 20 | ✅ all | Factual + tech questions people actually ask LLMs |
| `summarization` | 20 | ✅ all | Real-world topic passages; ROUGE-1 now active |
| `classification` | 20 | ✅ all | Authentic review text; 7 pos / 7 neg / 6 neutral |
| `instruction_following` | 20 | N/A | 7 bullet_points / 7 numbered_list / 6 single_word |

**`TASK_CONFIGS`** has 3 few-shot examples per task for the `few_shot` strategy.

---

## Phase 4: Benchmark Runner ✅ COMPLETE

- [x] Task 4.1: `experiments/benchmark.py` — `MODEL_CONFIGS` for all 7 models, `get_provider()` factory, full argparse, constraints plumbing for `instruction_following`

**Key additions:**
- `MODEL_CONFIGS` — single dict mapping model name → `{provider_cls, model, env_key}`
- `get_provider(name)` — reads env var, instantiates provider, raises clear error if key missing
- Constraints plumbing — `instruction_following` builds `InstructionFollowingEvaluator(constraints=example['constraints'])` per example, passes via `scorer.score_prompt(evaluator=...)`
- `TASKS` expanded to all 4
- `print_summary()` — breakdown by model, task, and strategy
- Graceful skip for models with missing API keys (warns, continues)
- Validated with `--mock`: 20/20 runs successful across all 4 tasks

**CLI usage:**
```bash
python experiments/benchmark.py                          # all 7 models, all 4 tasks, 4 examples → 560 runs
python experiments/benchmark.py --models gpt-4o-mini    # single model
python experiments/benchmark.py --tasks qa,classification
python experiments/benchmark.py --examples 20           # all examples per task
python experiments/benchmark.py --quick                 # 1 example × zero_shot (pipeline check)
python experiments/benchmark.py --mock                  # no API calls (CI/testing)
python experiments/benchmark.py --delay 2.0             # slower for Groq free tier (30 RPM)
```

---

## Phase 5: Analysis Scripts 🔲 PENDING

- [ ] Task 5.1: Create `experiments/analysis.py` (RQ1–RQ4 statistical analysis + 4 figures)

**Blocked by:** Need `results/benchmark_results.json` from a real benchmark run. Can stub with mock data.

**RQ plan:**
- RQ1: Does prompting strategy significantly affect GreenPES? (ANOVA across 5 strategies)
- RQ2: Which strategy is most token-efficient per task type?
- RQ3: Do smaller models achieve competitive GreenPES vs larger ones?
- RQ4: Is there a quality–efficiency tradeoff across strategies?

**Figures planned:** strategy × task heatmap, model comparison bar chart, quality vs token scatter, GreenPES distribution violin plot

---

## Phase 6: Environment ✅ COMPLETE

- [x] Task 6.1: `.env.example` updated with all provider keys + setup instructions
- [x] All 5 API keys set and tested in `.env`

---

## API Keys Status

| Provider | Key Set | Models Active |
|----------|---------|---------------|
| Groq | ✅ | llama-3.1-8b, llama-3.3-70b, qwen3-32b, kimi-k2 |
| Gemini | ✅ | gemini-2.0-flash only (2.5-pro needs billing) |
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

1. **Phase 5** — Create `experiments/analysis.py` (RQ1–RQ4 + 4 figures) **← START HERE next session**
2. Run full benchmark: `python experiments/benchmark.py --examples 4`
3. Run analysis: `python experiments/analysis.py`
4. Write paper

---

*Last updated: 2026-02-18*
