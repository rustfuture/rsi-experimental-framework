# Design

## Research protocol

Each run starts from a versioned baseline policy. For each generation, the generator emits candidate policies with a parent version and mutation label. Every candidate is evaluated against the training and development partitions. Selection is lexicographic on `(train accuracy, dev accuracy, candidate version)`; a candidate is accepted only when that pair improves over the current candidate. A development regression triggers `rollback_regression`, leaving the previous candidate active. The held-out partition is not passed to generation or selection and is measured after selection for a baseline/final comparison.

The result stores the configuration and seed, split counts, every selected proposal, parent links, candidate policy content, scores, decision, accepted-version chain, and limitations. Candidate IDs are SHA-256 hashes of canonical policy JSON, so a changed policy cannot reuse a version.

## Dataset and baseline

The synthetic task has eight lexical concepts (four positive and four negative), shuffled with a configured seed and split 50/25/25. The baseline intentionally knows only `good` and `bad`, while the mutation vocabulary contains the task concepts. This makes the first run useful for validating the loop, but not evidence about an LLM.

## Provider boundary

The `CandidateGenerator` protocol accepts a current `Candidate`, generation, and seed and returns candidates. A model adapter must provide its model/provider metadata, preserve parent/version fields, and never access held-out examples. Network access and paid APIs are out of scope for this milestone.

## Failure cases and safeguards

- malformed or non-deterministic providers must fail a reproducibility check before comparison;
- held-out metrics are not available to the generator or selector;
- dev regressions are explicitly recorded and rolled back;
- ties are deterministic via candidate version ordering;
- synthetic lexical gains must not be described as self-improvement.

## Planned extensions

Add multi-seed confidence intervals, task-level splits, evaluator agreement checks, a frozen open-weight provider, budget accounting, and adversarial leakage tests. Those extensions are intentionally not claimed by this first milestone.
