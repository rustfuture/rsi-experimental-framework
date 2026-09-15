# Changelog

## 0.3.0 — 2026-09-16

Research-correction round. The experiment definition changed, so results are versioned.

- **Matching fix (breaking, versioned)**: keyword matching is now whole-token instead of
  substring, so `safe` no longer matches inside `unsafe` and `clear` inside `unclear`.
  `experiment_version = "v2-token-match"`, `schema_version = 2`. v1 artifacts moved to
  `results/archive-v1/` and never pooled with v2 numbers.
- **Reporting fix**: `report.md` and the README results block are rendered from the JSON
  artifacts by `rsi_framework.reporting`. The stale hand-written README table
  (`0.812 ± 0.054` train, `1.00 ± 0.00` rejections, …) that disagreed with
  `multi_seed_results.json` is deleted. `--check` and `test_report_numbers_match_artifacts`
  fail on any drift.
- **Counting fix**: accepted versions **including the baseline** are reported separately
  from accepted **new changes**. `accepted_count` no longer silently means "baseline plus
  changes".
- **Provenance**: `results/provenance.json` records source commit, dirty paths, config
  hash, dataset hash, splits, run command, and interpreter.
- **Provider injection**: `run_experiment(..., candidate_generator=...)` plus
  `JsonFileProposalProvider`; every proposal passes `validate_proposal` (single edit,
  allowed vocabulary, no keyword in both polarities, bounded bias) and invalid proposals
  are recorded as rejections. `OpenWeightColabL4Stub` is documented as an unwired
  boundary, not an integration.
- **Lineage fix**: history rows carry a unique `event_id` separate from the policy
  `version`; `build_lineage_tree` keys on events, so repeated policy versions across
  generations no longer overwrite records. Missing parents are reported as orphans and
  cycles are detected instead of looping.
- **Ablations**: added `ablation_random_selection` as a genuine score-free control;
  `ablation_no_selection` is relabelled as dev-blind greedy selection; each mode records
  the mechanism it changes, its seed, and its dataset hash, and comparability is asserted.
- **Terminology**: `rollback_regression` renamed to `rejected_dev_regression` because the
  rule rejects a candidate and never reverts applied state. v1 label kept as an alias.
- **Scope corrections**: byte-equality claim limited to the four runtime-free artifacts;
  leakage test described as a scoped invariance check; the held-out sample size (8) is
  stated wherever an outcome is reported; multi-seed runs are described as replays of one
  shared synthetic pool.
- Added `--provider`, `--proposals`, and `--readme` CLI flags.

## 0.2.0 — 2026-09-15

- Added multi-seed empirical runner supporting >= 3 seeds with mean ± std aggregation (`run_multi_seed_experiment`).
- Added comparative ablation suite (`ablation_no_mutation`, `ablation_no_selection`, `ablation_no_rollback`).
- Added zero held-out leakage invariance unit test over the candidate search history.
- Added reproducibility test validating identical SHA-256 digests and file bytes on repeated executions.
- Added candidate lineage ancestry tracking and tree reconstruction (`build_lineage_tree`, `extract_lineage_chain`).
- Added abstract `OpenWeightLLMProvider` and `OpenWeightColabL4Stub` in `rsi_framework.providers` (unwired boundary).
- Added CLI flags `--seeds`, `--ablation`, and `--run-all-benchmarks`.
- Hardened CI workflow with least-privilege permissions (`contents: read`) and `persist-credentials: false`.

## 0.1.0 — 2026-09-15

- Added deterministic candidate generation/evaluation/selection harness.
- Added immutable candidate versions, parent lineage, rollback labels, and split isolation.
- Added JSON/JSONL/CSV/SVG/report artifacts, tests, exact reproduction commands, and CI.
