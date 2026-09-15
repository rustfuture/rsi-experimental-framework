# Design

## Research Protocol & Question

> Can a fixed base LLM-driven system improve held-out task performance through iterative candidate generation, evaluation, and selection?

This repository is a **harness**: it studies the accounting, validation, and reporting
mechanics that a self-improvement experiment needs before any model is attached. The
mutation source is a transparent deterministic rule, not a model, and the method is
labelled `HARNESS_BASELINE_NOT_LLM` everywhere. `OpenWeightLLMProvider` /
`OpenWeightColabL4Stub` are unwired boundary markers; no LLM inference has been run.

It is motivated by verifier-in-the-loop discrete search (Huang et al., 2024; Kumar et
al., 2024), where base weights are frozen and the "policy" is external non-parametric
state. Citations are motivation, not evidence: none of them is a proof that this
harness measures what they measure.

### Falsifiable Hypotheses

These hypotheses are **specified, not preregistered**. The design document and the
first result artifacts were committed together, so the commit history does not
establish that the hypotheses were fixed before the experiment was run.

- **$H_0$ (Null)**: Iterative search with selection and dev-regression rejection does
  not strictly improve held-out accuracy over the frozen baseline policy on this task
  ($\Delta_{\text{heldout}} \le 0$), measured across the listed seeds.
- **$H_1$ (Alternative)**: Selection on the dev partition steers the search toward
  policies that also score higher on the held-out partition ($\Delta_{\text{heldout}} > 0$)
  while dev regressions are rejected.

Both are evaluated with an **operational decision rule**, not a significance test: the
run is labelled `improvement` when the final policy's held-out accuracy is strictly
greater than the baseline's, `flat` when equal, and `regression` when lower, with a
fixed `regression_tolerance` used only for the dev-regression rejection. With 8
held-out examples a one-example difference is 0.125 accuracy, so these labels are
descriptive of a single run and must not be read as significance or equivalence.

## Matching Rule (experiment version `v2-token-match`)

A policy scores a text by counting how many of its positive/negative keywords occur as
**whole lowercase word tokens**:

- v1 matched substrings, so `safe` fired inside `unsafe` and `clear` fired inside
  `unclear`; a positive and its negated form cancelled each other and could make the
  documented keyword semantics wrong.
- v2 tokenizes with `[a-z0-9]+` and compares tokens. The change alters the experiment
  definition, so it is versioned via `experiment_version` and `schema_version`. v1
  artifacts are preserved under `results/archive-v1/`; v1 and v2 numbers are never
  pooled.

## Optimization Mechanics & Rejection Rule

The candidate loop implements monotonic greedy local search:

1. **Candidate Proposal**: a generator produces mutations of the active policy. The
   generator is injected (`run_experiment(..., candidate_generator=...)`); the default
   is the deterministic `DeterministicMutationGenerator`.
2. **Validation Gate**: every proposal passes `validate_proposal` before it is scored —
   exactly one edit (add one keyword, or move the bias by one), keywords a single
   lowercase token, no keyword in both polarities, no keyword outside the allowed
   vocabulary, bias within bounds. Malformed or out-of-policy proposals are recorded
   under `rejected_proposals` and never influence selection.
3. **Lexicographic Scoring**: candidates are ranked by
   `(-train_accuracy, -dev_accuracy, version_hash)`.
4. **Dev-Regression Rejection**: if the best candidate satisfies
   `dev_candidate + regression_tolerance < dev_parent`, the candidate is **rejected**
   with decision `rejected_dev_regression` and its mutation is never applied. Because
   the active policy state is only ever replaced by an *accepted* candidate, nothing is
   ever reverted: this is candidate rejection, not rollback of an applied state. The
   historical label `rollback_regression` appears only in v1 archives. The rule is a
   fixed-tolerance comparison, not a confidence-interval or trust-region barrier.
5. **Held-out Isolation**: the held-out split is measured once, after selection stops,
   and never enters generation, scoring, or the rejection rule.

## Scoped Data-Isolation Check

- **Partition split**: Train 50% (16), Dev 25% (8), Held-out 25% (8) of a fixed
  32-sentence synthetic pool.
- **What is verified**: `test_heldout_data_does_not_affect_selection` inverts every
  held-out label and rewrites every held-out sentence and checks that the recorded
  selection chain, versions, and dev scores are unchanged. That is a regression test for
  one code path on one dataset — it is **not** a mathematical proof of zero leakage, and
  no such proof is claimed.
- **What is not independent**: `make_dataset(seed)` builds the same 32 rows for every
  seed and only shuffles them. Multi-seed runs therefore replay one toy dataset; they
  are not independent real-world samples and their spread is not a sampling error.

## Provider Boundary & Hardware Mapping

```
+-------------------------------------------------------------+
|                      RSI Experiment Loop                     |
+-------------------------------------------------------------+
                              |
              generate(current, generation, seed)
                              |
                              v
             +----------------------------------+
             |    CandidateGenerator (injected)  |
             +----------------------------------+
               /                |               \
              v                 v                v
+------------------------+ +----------------+ +----------------------+
| DeterministicMutation  | | JsonFile       | | OpenWeightLLMProvider |
| Generator (default,    | | Proposal       | | (stub: raises, not    |
| stdlib only)           | | Provider       | | wired, no inference)  |
+------------------------+ +----------------+ +----------------------+
```

`JsonFileProposalProvider` is the concrete non-deterministic-capable path: an external
local model runner writes proposals to JSON and the harness validates each one. It
performs no inference itself. The stub exists so an accidental silent GPU/network call
is impossible; presenting it as a completed integration would be false.

No paid, closed APIs (e.g. OpenAI, Anthropic) are used. The protocol specifies greedy
decoding (`temperature=0.0`) and fixed seeds for reproducibility, but that only
constrains a future provider; it is not evidence about one.

## Ablations

Each mode runs with the same seed, dataset hash, and proposal budget. The mechanism
switched off is recorded per mode in the artifact (`ABLATION_MECHANISMS`):

- `baseline_full` — unmodified loop.
- `ablation_no_mutation` — generator never called; isolates the contribution of mutation.
- `ablation_no_selection` — dev-blind greedy selection by train accuracy; a **weaker
  selection rule**, not the absence of selection.
- `ablation_no_rollback` — dev-regression rejection disabled.
- `ablation_random_selection` — the genuine score-free control: a seed-controlled RNG
  picks one candidate per generation and it is accepted unconditionally.

## Reproducibility Scope

Byte equality is claimed only for `first_run.json`, `history.jsonl`, `metrics.csv`, and
`progress.svg`, which contain no timestamp or runtime. Files embedding `runtime_seconds`
and `report.md` (which renders them) are numerically reproducible but not
byte-reproducible. `provenance.json` records commit, dirty paths, config hash, dataset
hash, and the run command and is deliberately excluded from the claim.

## Citations & Academic References

1. **Huang et al. (ICLR 2024)**: *Large Language Models Cannot Self-Correct Reasoning
   Yet* — argues external, sound verifiers are needed for iterative refinement. Scope:
   about LLM self-correction, not about this keyword harness.
2. **Prechelt (1998)**: *Early Stopping — But When?* (Neural Networks: Tricks of the
   Trade) — formalizes validation-based stopping. This harness borrows the idea of a
   validation-triggered stop; it does not reproduce the paper's learning-curve criteria.
3. **Schulman et al. (ICML 2015)**: *Trust Region Policy Optimization* — motivates
   bounding how far a policy step may move. The fixed dev tolerance here is **not** a
   trust-region or confidence-interval bound and should not be described as one.
4. **Popper (1959)**: *The Logic of Scientific Discovery* — motivation for stating
   falsifiable hypotheses; it does not make the stated hypotheses preregistered.
5. **Khattab et al. (2023/ICLR 2024)**: *DSPy: Compiling Declarative Language Model
   Calls into Self-Improving Pipelines* — closest published framing for treating
   external policy state as the object of search.
