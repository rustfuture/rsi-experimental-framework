"""RSI experimental framework: a deterministic baseline for iterative selection."""

__version__ = "0.2.0"

from .core import (
    Candidate,
    DeterministicMutationGenerator,
    Example,
    Policy,
    Scores,
    build_lineage_tree,
    evaluate,
    extract_lineage_chain,
    make_dataset,
    run_ablation_experiments,
    run_experiment,
    run_multi_seed_experiment,
)
from .providers import (
    CandidateGenerator,
    OpenWeightColabL4Stub,
    OpenWeightLLMProvider,
)

__all__ = [
    "Candidate",
    "CandidateGenerator",
    "DeterministicMutationGenerator",
    "Example",
    "OpenWeightColabL4Stub",
    "OpenWeightLLMProvider",
    "Policy",
    "Scores",
    "build_lineage_tree",
    "evaluate",
    "extract_lineage_chain",
    "make_dataset",
    "run_ablation_experiments",
    "run_experiment",
    "run_multi_seed_experiment",
]
