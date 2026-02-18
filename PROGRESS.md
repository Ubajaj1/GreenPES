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

## Phase 2: Task Evaluators 🔲 PENDING

- [ ] Task 2.1: `ClassificationEvaluator` in `greenprompt/evaluators.py` + `tests/test_evaluators.py`
- [ ] Task 2.2: `InstructionFollowingEvaluator` in `greenprompt/evaluators.py` + tests
- [ ] Update `get_evaluator()` to include new task types

**Blocked by:** Nothing. **← START HERE next session**

---

## Phase 3: Benchmark Data 🔲 PENDING

- [ ] Task 3.1: Add `classification` task config + 20 examples to `experiments/prompting_strategies.py`
- [ ] Task 3.2: Add `instruction_following` task config + 20 examples
- [ ] Task 3.3: Expand `qa` and `summarization` from 5 → 20 examples each

**Current state:** QA has 5 examples, summarization has 5 examples. `classification` and `instruction_following` not yet in `TASK_CONFIGS` or `BENCHMARK_EXAMPLES`.

---

## Phase 4: Benchmark Runner 🔲 PENDING

- [ ] Task 4.1: Update `experiments/benchmark.py` with `MODEL_CONFIGS` for 7 models, `get_provider()`, and full argparse

**Current state:** Runner only supports old Groq + Gemini hardcoded, only `qa` + `summarization` tasks. Needs full rewrite of `__main__` block.

---

## Phase 5: Analysis Scripts 🔲 PENDING

- [ ] Task 5.1: Create `experiments/analysis.py` (RQ1–RQ4 statistical analysis + 4 figures)

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
- [x] `.claude/commands/update-progress.md` — `/update-progress` slash command (requires Claude Code restart to activate)
- [x] `.claude/commands/end-session.md` — `/end-session` slash command for safe commit + push

---

## Next Steps (in order)

1. **Phase 2** — Implement `ClassificationEvaluator` + `InstructionFollowingEvaluator` (Tasks 2.1, 2.2)
2. **Phase 3** — Expand benchmark data to 20 examples × 4 tasks (Tasks 3.1, 3.2, 3.3)
3. **Phase 4** — Update benchmark runner for all 7 models + 4 tasks (Task 4.1)
4. **Phase 5** — Create analysis script (Task 5.1)
5. Run full benchmark: `python experiments/benchmark.py --models all --tasks all --examples 4`
6. Run analysis: `python experiments/analysis.py`
7. Write paper

---

*Last updated: 2026-02-17*
