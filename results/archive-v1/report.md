# RSI Experimental Framework — Empirical Report

## Research Question
> Can a fixed base LLM-driven system improve held-out task performance through iterative candidate generation, evaluation, and selection?

Method Label: `HARNESS_BASELINE_NOT_LLM`
Overall Single-Run Outcome: **improvement**

## 1. Single-Run Benchmark (Seed: 20260915)

| Split | Baseline Accuracy | Final Policy Accuracy | Status |
|---|---:|---:|---|
| Train | 0.625 | 0.812 | Evaluated for selection |
| Dev | 0.375 | 0.625 | Evaluated for selection & rollback |
| Held-out | 0.375 | 0.750 | Post-selection measurement only |

- **Accepted Candidates Count**: 3
- **Total Search Generations**: 8
- **Final Policy Hash**: `candidate-f41795a16605`

## 2. Multi-Seed Empirical Verification (Seeds: 42, 1337, 2026)

To guard against single-seed anomalies, the loop was evaluated across independent dataset shuffles and search initializations.

| Seed | Train Acc | Dev Acc | Held-out Acc | Held-out Gain | Regressions | Accepted | Runtime (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.688 | 1.000 | 0.625 | +0.125 | 0 | 3 | 0.0039 |
| 1337 | 0.812 | 0.625 | 0.750 | +0.125 | 6 | 3 | 0.0038 |
| 2026 | 0.375 | 0.625 | 0.625 | +0.000 | 8 | 1 | 0.0040 |

### Statistical Summary (Mean ± Std)
- **Train Accuracy**: 0.625 ± 0.225
- **Dev Accuracy**: 0.750 ± 0.216
- **Held-out Accuracy**: 0.667 ± 0.072
- **Held-out Gain**: +0.083 ± 0.072
- **Regression Count**: 4.67 ± 4.16
- **Accepted Count**: 2.33 ± 1.15

## 3. Ablation Analysis

To verify the necessity of each component in the iterative loop, we compare the full system against restricted variants:
- `ablation_no_mutation`: Candidate mutation disabled.
- `ablation_no_selection`: Selection occurs without dev partition guidance.
- `ablation_no_rollback`: Dev regression triggers no rollback (greedy train acceptance).

| Variant | Train Acc | Dev Acc | Held-out Acc | Held-out Gain | Regressions | Accepted | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| `baseline_full` | 0.812 | 0.625 | 0.750 | +0.375 | 0 | 3 | improvement |
| `ablation_no_mutation` | 0.625 | 0.375 | 0.375 | +0.000 | 0 | 1 | flat |
| `ablation_no_selection` | 0.812 | 0.625 | 0.750 | +0.375 | 0 | 3 | improvement |
| `ablation_no_rollback` | 0.812 | 0.625 | 0.750 | +0.375 | 0 | 3 | improvement |

## 4. Methodological Safeguards

1. **Zero Held-out Leakage**: Held-out split data is never passed to candidate generation, scoring, or selection. Modifying held-out labels produces the exact bit-for-bit candidate selection chain.
2. **Byte-Level Reproducibility**: Given the same seed and configuration, identical JSON/CSV artifacts and SHA-256 digests are generated deterministically.
3. **Explicit Rollback**: Candidates causing dev degradation beyond tolerance are rejected with label `rollback_regression`.

## 5. Limitations & Scientific Honesty

- **Harness Baseline Only**: This milestone evaluates harness accounting and selection mechanics. No weights were trained and no LLM API was invoked.
- **Synthetic Task Scope**: Lexical sentiment heuristics do not demonstrate broad reasoning or general recursive self-improvement (RSI).
- **Provider Boundary**: Real open-weight LLMs (e.g. running on local GPU or Colab L4) must be attached via the defined `OpenWeightLLMProvider` protocol before making claims regarding neural policy improvement.
