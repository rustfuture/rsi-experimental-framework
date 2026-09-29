# Technical Reference

This document contains extended reference material, artifact descriptions, repository structure, and benchmark commands relocated from the main README.

The earlier MIT badge image URL is retained here for existing references: [legacy MIT badge](https://img.shields.io/badge/license-MIT-green).

## Framework Specifications

| Feature | Details |
|---|---|
| Language | Python 3.10+; deterministic path requires only the standard library |
| Experiment | Keyword-policy search over a fixed 32-sentence synthetic pool |
| Candidate gate | `validate_proposal` (single edit, vocabulary, polarity, bias bound) |
| Providers | Deterministic mutator (default), JSON proposals, optional local Transformers |
| Reproducibility | Canonical artifacts are byte-identical; CI re-runs and compares |
| Tests | 35 unit tests (CI on Python 3.11) |

## Result Artifacts

All raw and machine-readable data live under [`results/`](../results/):

| Artifact | Contents |
|---|---|
| [`results/first_run.json`](../results/first_run.json) | Canonical single-run record (byte-reproducible) |
| [`results/history.jsonl`](../results/history.jsonl) | Step-by-step search trajectory |
| [`results/metrics.csv`](../results/metrics.csv) | Generation metrics |
| [`results/multi_seed_results.json`](../results/multi_seed_results.json) | Multi-seed aggregates (embed runtime) |
| [`results/ablation_results.json`](../results/ablation_results.json) | Ablation runs with mechanism descriptions |
| [`results/provenance.json`](../results/provenance.json) | Source commit, config hash, dataset hash, command |
| [`results/report.md`](../results/report.md) | Rendered human-readable summary |
| [`pool-and-ablations.md`](pool-and-ablations.md) | Note on what the 32-sentence pool and the ablation table can and cannot show |
| [`results/archive-v1/`](../results/archive-v1/) | Superseded v1 artifacts, preserved verbatim |

### Historical Notes on Versioning

Experiment version `v2-token-match` uses whole-token matching so tokens do not match sub-strings across negation prefixes (for example, `safe` does not match inside `unsafe`). Superseded v1 archives live in [`results/archive-v1/`](../results/archive-v1/) and must not be pooled or combined with v2 numbers.

## Benchmark and Replay Commands

CI verifies that committed artifacts match live executions. You can run multi-seed evaluations and ablations locally:

```bash
# Multi-seed evaluation and ablations, updating report.md and README block in place
python3 -m rsi_framework --run-all-benchmarks --output results --readme README.md

# Compare artifacts without modifying committed files
python3 -m rsi_framework --config config/default.json --output /tmp/rsi-replay
python3 -m rsi_framework --run-all-benchmarks --output /tmp/rsi-benchmarks
```

## Repository Map

| Path | Contents |
|---|---|
| `rsi_framework/core.py` | Experiment loop, scoring, lineage, ablations |
| `rsi_framework/providers.py` | Provider boundary, validation gate, local provider |
| `rsi_framework/reporting.py` | Provenance, report rendering, README check |
| `rsi_framework/cli.py` | `python -m rsi_framework` entry point |
| `tests/` | Core, provider-boundary, and device-resolution tests |
| `results/` | Committed deterministic artifacts |
| `notebooks/rsi_open_weight_colab.ipynb` | Optional GPU walkthrough |
