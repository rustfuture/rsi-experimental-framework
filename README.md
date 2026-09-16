# RSI Experimental Framework

This private research repository tests the experimental accounting and verifiable verification needed to study one foundational question:

> Can a fixed base LLM-driven system improve held-out task performance through iterative candidate generation, evaluation, and selection?

## First Milestone: Deterministic Verifier Harness Baseline

The first milestone is deliberately a **deterministic harness baseline, not an LLM experiment**. A transparent keyword-policy mutator stands in for candidate generation. It runs with Python 3.10+ and standard library only (zero third-party dependencies, zero paid APIs), records immutable candidate versions via canonical SHA-256 digests, evaluates train/dev partitions during selection, isolates the held-out partition strictly for post-selection measurement, and rejects dev regressions via automated rollback.

### Key Capabilities

1. **Candidate Versioning & Lineage Tracking**: Canonical JSON hashing gives deterministic candidate IDs (`candidate-<sha256[:12]>`). Policy identity (`version`) is separated from occurrence identity (`event_id`), so the same policy may reappear in later generations or under a different parent without overwriting lineage records. `build_lineage_tree()` and `extract_lineage_events()` reconstruct ancestry with explicit orphan and cycle reporting.
2. **Scoped Held-out Isolation Check**: The held-out split is never passed to candidate generation, scoring, or selection. `test_heldout_data_does_not_affect_selection` checks one specific invariance on the fixed synthetic dataset — inverting every held-out label and rewriting every held-out sentence leaves the selection chain, versions, and dev scores unchanged. This is a regression test, not a mathematical proof of zero leakage for arbitrary code paths.
3. **Scoped Byte-Level Reproducibility**: With an identical config and seed, `first_run.json`, `history.jsonl`, `metrics.csv`, and `progress.svg` are byte-for-byte identical. Artifacts that embed `runtime_seconds` (`multi_seed_*`, `ablation_*`, `report.md`) are numerically reproducible but not byte-reproducible, and `provenance.json` records the checkout state by design.
4. **Multi-Seed Replay**: `--seeds 42,1337,2026` records train/dev/held-out accuracy, rejection counts, accepted counts, runtime, and SHA-256 digests. Aggregates are reported as mean ± sample standard deviation (`ddof=1`) with explicit seed count and per-split sample sizes. Every seed re-shuffles **the same** 32-sentence synthetic pool, so these are replays of one toy dataset, not independent real-world samples.
5. **Ablation Suite**: `ablation_no_mutation`, `ablation_no_selection` (dev-blind greedy selection — still score-based), `ablation_no_rollback`, and `ablation_random_selection` (the genuine score-free control). Each run records which mechanism it switches off and is executed with the same seed, dataset hash, and proposal budget.
6. **Injectable, Validated Provider Boundary**: `run_experiment(..., candidate_generator=...)` replaces the built-in mutation generator without touching the loop. `JsonFileProposalProvider` reads externally produced proposals, and `LocalTransformersProvider` runs a local HuggingFace Transformers model as an optional candidate source (`--provider llm`). Every proposal from every source is checked by `validate_proposal` (single edit, allowed vocabulary, no shared polarity, bounded bias) before it can be scored. `OpenWeightColabL4Stub` remains a historical boundary marker that intentionally raises; it is no longer the only open-weight path.

### Exact Reproduction Commands

```bash
cd rsi-experimental-framework

# Run full test suite (harness mechanics, lineage, provenance, reporting consistency)
python3 -m unittest discover -s tests -v

# Run single deterministic baseline run and render report.md from the artifacts
python3 -m rsi_framework --config config/default.json --output results

# Verify bit-for-bit reproducibility of the deterministic artifacts only
python3 -m rsi_framework --config config/default.json --output /tmp/rsi-replay
cmp results/first_run.json /tmp/rsi-replay/first_run.json

# Run multi-seed evaluation and ablation benchmarks, refreshing report.md + README block
python3 -m rsi_framework --run-all-benchmarks --output results --readme README.md

# Verify that report.md and the README results block match the JSON artifacts
python3 -m rsi_framework.reporting --results results --readme README.md --check
```

### Empirical Results Summary

> The block below is generated from `results/*.json` by
> `python -m rsi_framework.reporting --results results --readme README.md --update`.
> A regression test (`test_report_numbers_match_artifacts`) fails if it drifts from
> the artifacts. Experiment version: `v2-token-match`; v1 archives live in
> [`results/archive-v1/`](results/archive-v1/) and must not be pooled with these numbers.

<!-- BEGIN GENERATED: empirical results (do not edit by hand) -->
### 1. Single Run (Seed 20260915)
- **Method label**: `HARNESS_BASELINE_NOT_LLM`; experiment version `v2-token-match`
- **Train accuracy**: 0.625 (10/16) -> 1.000 (16/16)
- **Dev accuracy**: 0.375 (3/8) -> 1.000 (8/8)
- **Held-out accuracy**: 0.375 (3/8) -> 1.000 (8/8)
- **Outcome**: `improvement` (4 accepted new changes, 5 accepted versions including the baseline, 0 rejected dev regressions)

### 2. Multi-Seed Replay (Seeds: 42, 1337, 2026)
- **N seeds**: 3; **held-out examples per run**: 8; **std**: sample standard deviation (`ddof=1`)
- **Train accuracy (final)**: 0.792 ± 0.361
- **Dev accuracy (final)**: 0.875 ± 0.216
- **Held-out accuracy (final)**: 0.875 ± 0.216
- **Held-out gain**: +0.292 ± 0.260
- **Rejected dev regressions**: 2.67 ± 4.62
- **Accepted versions including the baseline**: 3.67 ± 2.31
- **Accepted new changes (baseline excluded)**: 2.67 ± 2.31

All seeds re-shuffle the same 32-row synthetic pool; these are replays of one toy dataset, not independent real-world samples. `improvement` / `flat` / `regression` are operational decision labels, not significance results.

### 3. Ablations
| Variant | Held-out accuracy | Held-out gain | Accepted new | Outcome |
|---|---:|---:|---:|---|
| `baseline_full` | 1.000 | +0.625 | 4 | improvement |
| `ablation_no_mutation` | 0.375 | +0.000 | 0 | flat |
| `ablation_no_selection` | 1.000 | +0.625 | 4 | improvement |
| `ablation_no_rollback` | 1.000 | +0.625 | 4 | improvement |
| `ablation_random_selection` | 0.375 | +0.000 | 8 | flat |
<!-- END GENERATED: empirical results -->

All raw and machine-readable data are recorded under [`results/`](results/):
- [`results/first_run.json`](results/first_run.json): Canonical single run record (deterministic).
- [`results/multi_seed_results.json`](results/multi_seed_results.json): Multi-seed aggregates (embeds runtime).
- [`results/multi_seed_metrics.csv`](results/multi_seed_metrics.csv): Tabular multi-seed data.
- [`results/ablation_results.json`](results/ablation_results.json): Ablation benchmark results with per-mode mechanism descriptions.
- [`results/history.jsonl`](results/history.jsonl): Step-by-step search trajectory (deterministic).
- [`results/metrics.csv`](results/metrics.csv): Generation metrics (deterministic).
- [`results/provenance.json`](results/provenance.json): Source commit, dirty paths, config hash, dataset hash, run command.
- [`results/report.md`](results/report.md): Human-readable empirical summary, rendered from the JSON artifacts.
- [`results/progress.svg`](results/progress.svg): Dev accuracy trajectory (deterministic).
- [`results/archive-v1/`](results/archive-v1/): Superseded v1 (substring-matching) artifacts, preserved verbatim.

## Second Milestone: Optional Local Model Provider (Implementation Complete)

A real optional local open-weight provider is now implemented:

- `LocalTransformersProvider` (`rsi_framework/providers.py`) loads `--model` with HuggingFace Transformers on a validated `--device`.
- `LLMMutationGenerator` converts provider output into candidate policies.
- CLI support: `--provider llm`, `--model`, and `--device`.

Status, stated precisely:

- **Implementation complete.** The provider, the generator, and the CLI wiring exist and are covered by boundary tests. Device selection is explicit and validated: an unavailable runtime (`torch`/`transformers`) or an unavailable requested device exits with `EXPERIMENT_BLOCKED_BY_RUNTIME` instead of silently substituting another device.
- **No real-model RSI experiment has been run or recorded.** Every result in `results/` is the deterministic harness baseline (`HARNESS_BASELINE_NOT_LLM`); none was produced by a language model. No intelligence improvement is claimed for the deterministic toy baseline, and none is claimed for the unrecorded provider path.

Every proposal from every provider is still passed through the central `validate_proposal` gate before it can be scored: no-ops, multi-element edits, out-of-vocabulary or malformed keywords, keywords shared across polarities, and out-of-bound bias are rejected.

Invocation example (requires `torch` and `transformers` installed in the active environment; the deterministic baseline does not):

```bash
python3 -m rsi_framework --provider llm --model <MODEL_ID> --device auto
```

## Limitations & Scientific Boundary

- **Harness Validation, Not Intelligence**: The mutation search loop validates search state accounting, rejection mechanics, and split containment. It does not establish general reasoning, emergence, or unbounded self-improvement.
- **`improvement` / `flat` / `regression` Are Operational Labels**: They come from one held-out comparison with a fixed tolerance on one run. They are not statistical significance, an equivalence proof, or a confidence-interval barrier.
- **Tiny Held-out Split**: Each run measures 8 held-out examples, so a single example moves accuracy by 0.125. Do not read a research result out of 8 examples.
- **Lexical Synthetic Data**: The 8-concept synthetic sentence pool is shared by every seed (seeds re-shuffle it), so multi-seed runs are replays of one toy dataset, not independent real-world samples.
- **Provider Boundary**: `JsonFileProposalProvider`, the `candidate_generator` injection point, and an optional local `LocalTransformersProvider` (`--provider llm`) are implemented, and every proposal is validated by `validate_proposal`. **No real-model RSI experiment has been run and recorded**, so the empirical results on this page are deterministic-baseline only. `OpenWeightColabL4Stub` remains a historical boundary marker that intentionally raises.
- **Zero Paid APIs**: No cloud LLM APIs (OpenAI, Anthropic, etc.) are used or required.

See [`DESIGN.md`](DESIGN.md) for the falsifiable hypotheses, mechanics, and literature citations.
