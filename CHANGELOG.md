# Changelog

## 0.2.0 — 2026-09-15

- Added multi-seed empirical runner supporting >= 3 seeds with mean ± std aggregation (`run_multi_seed_experiment`).
- Added comparative ablation suite (`ablation_no_mutation`, `ablation_no_selection`, `ablation_no_rollback`).
- Added formal zero held-out leakage verification unit test proving bit-for-bit invariance of candidate search history.
- Added byte-level reproducibility test validating identical SHA-256 digests and file bytes on repeated executions.
- Added candidate lineage ancestry tracking and tree reconstruction (`build_lineage_tree`, `extract_lineage_chain`).
- Added abstract `OpenWeightLLMProvider` and `OpenWeightColabL4Stub` in `rsi_framework.providers` for zero-paid-API local/Colab L4 expansion.
- Added CLI flags `--seeds`, `--ablation`, and `--run-all-benchmarks`.
- Hardened CI workflow with least-privilege permissions (`contents: read`) and `persist-credentials: false`.

## 0.1.0 — 2026-09-15

- Added deterministic candidate generation/evaluation/selection harness.
- Added immutable candidate versions, parent lineage, rollback labels, and split isolation.
- Added JSON/JSONL/CSV/SVG/report artifacts, tests, exact reproduction commands, and CI.
