# RSI Experimental Framework

This private research repository tests the accounting needed to study one question:

> Can a fixed base LLM-driven system improve held-out task performance through iterative candidate generation, evaluation, and selection?

## First milestone

The first milestone is deliberately a **deterministic harness baseline, not an LLM experiment**. A transparent keyword-policy mutator stands in for candidate generation. It runs with Python 3.10+ and no paid API, records immutable candidate versions, evaluates train/dev during selection, evaluates held-out only for the final measurement, and rejects dev regressions (rollback).

## Reproduce exactly

```bash
cd rsi-experimental-framework
python3 -m unittest discover -s tests -v
python3 -m rsi_framework --config config/default.json --output results
python3 -m rsi_framework --config config/default.json --output /tmp/rsi-replay
cmp results/first_run.json /tmp/rsi-replay/first_run.json
```

The committed first run is in [`results/first_run.json`](results/first_run.json), with line-oriented history in [`results/history.jsonl`](results/history.jsonl), tabular metrics in [`results/metrics.csv`](results/metrics.csv), a short report in [`results/report.md`](results/report.md), and a lightweight SVG plot in [`results/progress.svg`](results/progress.svg). Baseline and final held-out scores are measured after selection for comparison; held-out data never affects generation or selection.

## Provider boundary

`CandidateGenerator` is a small protocol. `DeterministicMutationGenerator` is the only implementation in this milestone. A later open-weight/Colab L4 adapter may implement the same interface, but this repository never labels deterministic mutations as LLM output and has no network/model dependency.

## Limitations

The dataset is synthetic and lexical; it cannot establish general intelligence, self-improvement, or transfer. The baseline has no learned representation, no paid API, and no model-generated candidates. Only one seed and one task family are recorded. Future work must add pre-registered tasks, multiple seeds, a real fixed base model, evaluator calibration, and leakage checks before making scientific claims.

See [`DESIGN.md`](DESIGN.md) for the protocol and failure cases.
