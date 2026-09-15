# Design

## Research Protocol & Question

> Can a fixed base LLM-driven system improve held-out task performance through iterative candidate generation, evaluation, and selection?

This repository formalizes **Verifier-in-the-Loop Discrete Policy Search** (Huang et al., 2024; Kumar et al., 2024). When base model weights are frozen, "policy" constitutes external, non-parametric state (decision heuristics, prompt configurations, or programmatic rules).

### Falsifiable Hypotheses (Popper, 1959)

- **$H_0$ (Null Hypothesis)**: Iterative search with selection and rollback does not achieve statistically significant out-of-distribution improvement on held-out tasks ($\Delta_{\text{heldout}} \le 0$) compared to the frozen baseline policy across multiple independent initializations.
- **$H_1$ (Alternative Hypothesis)**: Iterative candidate evaluation on the development partition steers selection toward policies that generalize to held-out tasks ($\Delta_{\text{heldout}} > 0$) while bounding regressions via early rollback.

### Optimization Mechanics & Rollback Theory

The candidate loop implements Monotonic Greedy Local Search ($T \to 0$ simulated annealing):
1. **Candidate Proposal**: Mutations perturb the active candidate policy.
2. **Lexicographic Scoring**: Candidates are ranked by `(-train_accuracy, -dev_accuracy, candidate_version_hash)`.
3. **Rollback Safeguard**: If a proposed candidate regresses dev set performance beyond `regression_tolerance` ($dev_{\text{cand}} + \epsilon < dev_{\text{parent}}$), the candidate is rejected with reason `rollback_regression`. This acts as an epistemic trust-region barrier (Schulman et al., 2015) and validation early stopping mechanism (Prechelt, 1998) to prevent catastrophic over-fitting.
4. **Held-out Isolation**: The held-out split is strictly evaluated post-selection to assess empirical generalization.

## Zero Data Leakage Architecture

- **Partition Disjointness**: The dataset is split into Train (50%), Dev (25%), and Held-out (25%).
- **Verification of Non-Influence**: `test_heldout_data_does_not_affect_selection` confirms that altering, corrupting, or inverting held-out labels has zero influence on candidate generation, search decisions, or dev scores.

## Provider Boundary & Hardware Mapping

```
+-------------------------------------------------------------+
|                      RSI Experiment Loop                     |
+-------------------------------------------------------------+
                              |
              propose_candidates(current, gen, seed)
                              |
                              v
             +----------------------------------+
             |    CandidateGenerator Protocol   |
             +----------------------------------+
               /                              \
              /                                \
             v                                  v
+-------------------------------+  +--------------------------------+
| DeterministicMutationGenerator|  |    OpenWeightLLMProvider       |
| (Active Milestone: zero-cost, |  | (Future: Colab L4 / local GPU  |
| standard library harness)     |  | vLLM / HuggingFace offline)    |
+-------------------------------+  +--------------------------------+
```

No paid, closed APIs (e.g. OpenAI, Anthropic) are used. The protocol specifies greedy decoding (`temperature=0.0`) and fixed seeds for reproducibility.

## Citations & Academic References

1. **Huang et al. (ICLR 2024)**: *Large Language Models Cannot Self-Correct Reasoning Yet*. Analyzes the necessity of external, sound verifiers in iterative refinement loops.
2. **Prechelt (1998)**: *Early Stopping — But When?* Neural Networks: Tricks of the Trade. Formalizes rollback criteria on validation regression.
3. **Schulman et al. (ICML 2015)**: *Trust Region Policy Optimization*. Theoretical bounds on monotonic policy improvement.
4. **Popper (1959)**: *The Logic of Scientific Discovery*. Formulates falsifiability criteria for empirical hypothesis testing.
5. **Khattab et al. (2023)**: *DSPy: Compiling Declarative Language Model Calls into State-of-the-Art Pipelines*.
