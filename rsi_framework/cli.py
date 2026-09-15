import argparse
import json
import sys
from pathlib import Path

from .core import make_dataset, run_ablation_experiments, run_experiment, run_multi_seed_experiment
from .providers import JsonFileProposalProvider
from .reporting import collect_provenance, load_artifacts, render_report, update_readme


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deterministic RSI harness baseline")
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list of seeds (e.g. 42,1337,2026)")
    parser.add_argument(
        "--ablation",
        type=str,
        choices=[
            "ablation_no_mutation",
            "ablation_no_selection",
            "ablation_no_rollback",
            "ablation_random_selection",
        ],
        default=None,
        help="Run single experiment with specific ablation mode",
    )
    parser.add_argument(
        "--run-all-benchmarks",
        action="store_true",
        help="Run baseline run, multi-seed benchmarks (42, 1337, 2026), and all ablations",
    )
    parser.add_argument(
        "--provider",
        choices=["deterministic", "json"],
        default="deterministic",
        help=(
            "Candidate source. 'deterministic' (default) is the built-in mutation generator; "
            "'json' reads externally produced proposals from --proposals and validates each one. "
            "Neither option runs an LLM."
        ),
    )
    parser.add_argument(
        "--proposals",
        type=Path,
        default=None,
        help="JSON proposal file used when --provider json",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="Also refresh the generated results block in this README",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else sys.argv[1:])

    config = json.loads(args.config.read_text())

    if args.ablation:
        config["ablation_mode"] = args.ablation

    generator = None
    if args.provider == "json":
        if args.proposals is None:
            raise SystemExit("--provider json requires --proposals PATH")
        generator = JsonFileProposalProvider(args.proposals)

    result = run_experiment(config, args.output, candidate_generator=generator)

    multi_seed_summary = None
    recorded_seeds = [int(config["seed"])]
    if args.seeds:
        seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        multi_seed_summary = run_multi_seed_experiment(config, seeds=seed_list, output_dir=args.output)
        recorded_seeds = seed_list
    elif args.run_all_benchmarks:
        recorded_seeds = [42, 1337, 2026]
        multi_seed_summary = run_multi_seed_experiment(config, seeds=recorded_seeds, output_dir=args.output)

    ablation_summary = None
    if args.run_all_benchmarks:
        ablation_summary = run_ablation_experiments(config, output_dir=args.output)

    # Provenance records the checkout state and is not part of the byte-equality claim.
    examples = make_dataset(int(config["seed"]), int(config.get("examples_per_pattern", 4)))
    collect_provenance(
        args.output,
        config,
        examples,
        command="python -m rsi_framework " + " ".join(sys.argv[1:]),
        seeds=recorded_seeds,
    )

    # report.md is always rendered from the artifacts on disk, never hand-edited.
    artifacts = load_artifacts(args.output)
    (args.output / "report.md").write_text(render_report(artifacts).rstrip() + "\n")
    if args.readme is not None:
        update_readme(args.readme, artifacts)

    output_summary = {
        "experiment_version": result["experiment_version"],
        "outcome": result["outcome"],
        "final": result["final"],
        "accepted_versions_including_baseline": result["accepted_baseline_included_count"],
        "accepted_new_changes": result["accepted_new_change_count"],
        "rejected_proposals": result["rejected_proposal_count"],
    }
    if multi_seed_summary:
        output_summary["multi_seed_summary"] = {
            "seeds": multi_seed_summary["seeds"],
            "heldout_gain_mean": multi_seed_summary["heldout_gain"]["mean"],
            "heldout_gain_std": multi_seed_summary["heldout_gain"]["std"],
        }
    if ablation_summary:
        output_summary["ablation_summary"] = [
            {"mode": r["mode"], "heldout_gain": r["heldout_gain"], "outcome": r["outcome"]}
            for r in ablation_summary["runs"]
        ]

    print(json.dumps(output_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
