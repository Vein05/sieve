# SIEVE Cleanup TODO

Prioritized by severity. Check off as completed.

---

## P0: Silent correctness bugs

- [ ] `assembly_methods/llm_compiler.py:138` -- Ordering queries misclassified as "counting", get wrong prompt
- [ ] `run_llm_summarize_baseline.py:199` -- Uses word count (`split()`) instead of tiktoken `token_count()` for efficiency metrics
- [ ] `run_llm_summarize_baseline.py:254-257` -- Reads `quality_norm` and `exact_correct` keys that don't exist in the dict (always returns 0 via `.get()`)
- [ ] `answer_generation/runner.py:887-888` -- Bare `except Exception: pass` swallows MiniLM errors
- [ ] `compile_once_run_again.py:264-269` -- Generation error key set but never included in output dict
- [ ] `scoring_judge.py:421-436` -- Dedup key collision: same `(question, ref, gen)` across different `question_type`s uses wrong judge prompt

---

## P1: Dead code removal

- [ ] `compiler/execution/rescue.py` -- 1-line placeholder. Delete file
- [ ] `compiler/telemetry/__init__.py` -- Empty package. Delete directory
- [ ] `compiler/execution/deterministic.py` -- `validated_slot_contract` never called externally. Delete
- [ ] `answer_generation/selector_registry.py` -- `_controller_precompute_worker_count` superseded. Delete
- [ ] `answer_generation/selector_registry.py:471` -- Unreachable `if compiled_texts: return` branch. Delete
- [ ] `assembly_methods/__init__.py` -- `LEARNED_METHOD_REQUIRED_MODELS` and `LEGACY_METHOD_ALIASES` are empty dicts, never populated. Delete with their validation logic
- [ ] `assembly_methods/evidence_roles.py:8-12` -- `cached_build_profile` imported twice from `.common`
- [ ] `assembly_methods/proposal_retrieval.py:14-15` -- `get_stems_for_text` and `stem_token` imported but unused in module scope

---

## P2: Duplication consolidation

- [ ] Create `assembly_methods/_constants.py` -- Consolidate ~5 independent copies of comparison/state/ordering marker constants from `evidence_units`, `evidence_roles`, `evidence_renderer`, `query_targets`, `memory_object_builder`
- [ ] Create `_cli_utils.py` -- Extract shared argparse boilerplate (`--provider`, `--model`, `--parallelism`, etc.) from 4 entry-point scripts
- [ ] `answer_generation/generation_client.py` -- Define `DEFAULT_APP_URL` once, remove 3 hardcoded copies of anonymous submission URL
- [ ] `compiler/runtime/bindings.py:767` vs `runtime/text.py:223` -- `_time_unit_from_query` defined identically twice. Keep one, import
- [ ] `compiler/execution/deterministic.py:22` vs `policy.py:14` -- `_plan_metadata_text` defined identically twice. Keep one, import
- [ ] `compiler/execution/evaluation.py` -- Extract `_calendar_delta()` helper to eliminate duplicated month/year arithmetic in `_execute_relative_time` and `_execute_temporal_interval`
- [ ] `compiler/execution/evaluation.py:53,344` -- `_DURATION_UNITS` declared as module constant then re-declared as local variable. Delete the local
- [ ] `assembly_methods/common.py` vs `memory_object_builder.py` -- Two separate spaCy pipeline loads. Consolidate into one
- [ ] `assembly_methods/query_targets.py` vs `common.py` -- `_NUMBER_WORDS` defined as `set` in one, `dict` in the other. Consolidate
- [ ] `answer_generation/prompting.py` -- `_is_relative_duration_query` computed twice under two names. Deduplicate

---

## P3: God functions (>200 lines, must split)

- [ ] `answer_generation/runner.py` -- `_run_phase1_step` (411 lines, 18 kwargs). Split into routing, generation, output serialization
- [ ] `answer_generation/runner.py` -- `write_summary_markdown` (387 lines). Extract to `runner_report.py`
- [ ] `answer_generation/scoring.py` -- `evaluate_run` (280 lines). Split into per-dataset summary helpers
- [ ] `assembly_methods/memory_object_builder.py` -- Whole file is 2,137 lines doing 3 jobs. Split into `json_turn_parser.py`, `attribute_extractor.py`, `memory_object_builder.py`
- [ ] `assembly_methods/query_targets.py` -- `_extract_query_targets_impl` (203 lines, 6+ nesting levels). Decompose into pipeline steps
- [ ] `assembly_methods/evidence_renderer.py` -- `render_multi_session_fact_list` (344 lines). Split into slot/temporal/multi-session renderers
- [ ] `assembly_methods/evidence_roles.py` -- `infer_unit_roles` (375 lines). Extract per-family branches
- [ ] `compiler/execution/selection.py` -- `compile_evidence` (590 lines). Extract `_rematerialize()` helper (eliminates 4x copy-paste), then split into `_select_units`, `_run_rescue_passes`, `_build_result`
- [ ] `controller_v0/logic.py` -- `run_example` (385 lines). Extract scoring sub-loops

---

## P4: Over-engineering removal

- [ ] `compiler/execution/deterministic.py` -- `build_answer_contract` takes 26 `Callable` kwargs that are never varied. Replace with direct imports. This also deletes 8 wrapper functions in `selection.py:109-276`
- [ ] `compiler/runtime/bindings.py` -- Wildcard re-exports from 4 modules, 40+ imports. Have callers import from actual modules directly
- [ ] `compiler/execution/evaluation.py:530-595` -- 9-branch `schema_name` if-elif chain. Replace with dispatch dict

---

## P5: Bad practices cleanup

- [ ] `bootstrap_ci.py` -- No `if __name__ == "__main__"` guard, executes on import
- [ ] `mechanism_attribution.py` -- No `if __name__ == "__main__"` guard, hardcoded `assert len(ids) == 500`
- [ ] `bootstrap_ci.py` -- Pure-Python loop over 10k iterations despite importing numpy. Vectorize
- [ ] `answer_generation/runner.py:43` -- `_SOURCE_RUN_QUESTION_DATE_CACHE` is module-level mutable dict, not thread-safe under ThreadPoolExecutor
- [ ] `answer_generation/generation_client.py` -- Hardcoded `"seed": 42` in OpenRouter payload. Make configurable
- [ ] `answer_generation/selector_registry.py:42` -- `torch` imported unconditionally, crashes if unavailable
- [ ] `controller_v0/constants.py:350-497` -- `CONCEPT_SPECS` contains 30 dataset-specific entries baked into Python. Move to `configs/concept_specs.json`
- [ ] `assembly_methods/evidence_renderer.py` -- Inline `_stop` set allocated on every call in hot render path. Hoist to module level
- [ ] `assembly_methods/memory_object_builder.py` -- `_TEMPORAL_ADVERBS` set re-allocated inside function on every call. Hoist to module level
- [ ] `compiler/pipeline.py:638-751` -- 60-key `selection_meta` dict literal built twice nearly identically. Extract `_build_selection_meta()` factory

---

## P6: Testing infrastructure

- [ ] Create `tests/` directory mirroring source tree
- [ ] Add `pytest.ini` or `pyproject.toml` with test config
- [ ] Write smoke tests for entry points (do they parse args, load data?)
- [ ] Write unit tests for pure functions in `common.py` (stemming, tokenization, profile building)
- [ ] Write unit tests for `query_targets.py` target extraction
- [ ] Write integration test for compile-once-replay-again path
- [ ] Add CI (GitHub Actions) running pytest on push

---

## Done

(Move completed items here with date)
