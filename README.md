# RSI Experimental Framework

This private research repository tests the experimental accounting and verifiable verification needed to study one foundational question:

> Can a fixed base LLM-driven system improve held-out task performance through iterative candidate generation, evaluation, and selection?

## First Milestone: Deterministic Verifier Harness Baseline

The first milestone is deliberately a **deterministic harness baseline, not an LLM experiment**. A transparent keyword-policy mutator stands in for candidate generation. It runs with Python 3.10+ and standard library only (zero third-party dependencies, zero paid APIs), records immutable candidate versions via canonical SHA-256 digests, evaluates train/dev partitions during selection, isolates the held-out partition strictly for post-selection measurement, and rejects dev regressions via automated rollback.

### Key Capabilities

1. **Candidate Versioning & Lineage Tracking**: Canonical JSON hashing guarantees deterministic, collision-resistant candidate IDs (`candidate-<sha256[:12]>`). Full lineage ancestry is reconstructed via `build_lineage_tree()` and `extract_lineage_chain()`.
2. **Zero Held-out Leakage Guarantee**: The held-out split is never passed to candidate generation, evaluation, or selection. A unit test (`test_heldout_data_does_not_affect_selection`) formally proves that corrupting or inverting held-out data results in 100% identical candidate selection history, versions, and dev scores.
3. **Byte-Level Reproducibility**: Consecutive runs with identical config and seed produce byte-for-byte identical JSON and CSV artifacts (`cmp`-verified).
4. **Multi-Seed Statistical Verification**: Out-of-the-box support for multi-seed execution (`--seeds 42,1337,2026`), recording train/dev/heldout accuracy, regression counts, accepted candidate counts, execution time, and SHA-256 artifact digests with mean ± std.
5. **Ablation Suite**: Full comparative ablation suite testing `ablation_no_mutation`, `ablation_no_selection`, and `ablation_no_rollback`.
6. **Open-Weight Provider Interface**: Abstract `OpenWeightLLMProvider` and `OpenWeightColabL4Stub` interfaces in `rsi_framework.providers` establish clean boundaries for future local GPU / Google Colab L4 open-weight model integration (e.g. vLLM, HuggingFace Transformers) with strictly zero paid API calls.

## Exact Reproduction Commands

```bash
cd rsi-experimental-framework

# Run full test suite (10 unit tests covering leakage, reproducibility, ablations, lineage)
python3 -m unittest discover -s tests -v

# Run single deterministic baseline run
python3 -m rsi_framework --config config/default.json --output results

# Verify bit-for-bit reproducibility
python3 -m rsi_framework --config config/default.json --output /tmp/rsi-replay
cmp results/first_run.json /tmp/rsi-replay/first_run.json

# Run multi-seed evaluation and ablation benchmarks
python3 -m rsi_framework --run-all-benchmarks --output results
```

## Empirical Results Summary

### 1. Single Run (Seed 20260915)
- **Method Label**: `HARNESS_BASELINE_NOT_LLM`
- **Train Accuracy**: 0.625 -> 0.812 (+0.188)
- **Dev Accuracy**: 0.375 -> 0.625 (+0.250)
- **Held-out Accuracy**: 0.375 -> 0.750 (+0.375)
- **Outcome**: `improvement` (3 candidates accepted, 1 rollback)

### 2. Multi-Seed Benchmark (Seeds: 42, 1337, 2026)
- **Train Accuracy**: 0.812 ± 0.054
- **Dev Accuracy**: 0.625 ± 0.108
- **Held-out Accuracy**: 0.708 ± 0.072
- **Held-out Gain**: +0.083 ± 0.072
- **Regression Count**: 1.00 ± 0.00
- **Accepted Candidates Count**: 2.33 ± 0.47

### 3. Ablations
| Variant | Held-out Accuracy | Held-out Gain | Accepted | Outcome |
|---|---:|---:|---:|---|
| `baseline_full` | 0.750 | +0.375 | 3 | improvement |
| `ablation_no_mutation` | 0.375 | +0.000 | 1 | flat |
| `ablation_no_selection` | 0.750 | +0.375 | 4 | improvement |
| `ablation_no_rollback` | 0.750 | +0.375 | 4 | improvement |

All raw and machine-readable data are recorded under [`results/`](results/):
- [`results/first_run.json`](results/first_run.json): Canonical single run record.
- [`results/multi_seed_results.json`](results/multi_seed_results.json): Statistical multi-seed aggregates.
- [`results/multi_seed_metrics.csv`](results/multi_seed_metrics.csv): Tabular multi-seed data.
- [`results/ablation_results.json`](results/ablation_results.json): Ablation benchmark results.
- [`results/history.jsonl`](results/history.jsonl): Step-by-step search trajectory.
- [`results/metrics.csv`](results/metrics.csv): Generation metrics.
- [`results/report.md`](results/report.md): Human-readable empirical summary.
- [`results/progress.svg`](results/progress.svg): Dev accuracy trajectory.

## Limitations & Scientific Boundary

- **Harness Validation, Not Intelligence**: The mutation search loop validates search state accounting, rollback mechanisms, and split containment. It does not establish general reasoning, emergence, or unbounded self-improvement.
- **Lexical Synthetic Data**: The 8-concept synthetic classification task is structured to verify deterministic harness behavior without training compute.
- **Provider Boundary**: Real open-weight LLMs (e.g. running on local GPU or Colab L4) must be attached via `OpenWeightLLMProvider` before drawing conclusions regarding neural policy optimization.
- **Zero Paid APIs**: No cloud LLM APIs (OpenAI, Anthropic, etc.) are used or required.

See [`DESIGN.md`](DESIGN.md) for theoretical foundations, falsifiability criteria, and literature citations.
