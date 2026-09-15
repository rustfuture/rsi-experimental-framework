import argparse
import json
from pathlib import Path

from .core import _write_report, run_ablation_experiments, run_experiment, run_multi_seed_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic RSI harness baseline")
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list of seeds (e.g. 42,1337,2026)")
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["ablation_no_mutation", "ablation_no_selection", "ablation_no_rollback"],
        default=None,
        help="Run single experiment with specific ablation mode",
    )
    parser.add_argument(
        "--run-all-benchmarks",
        action="store_true",
        help="Run baseline run, multi-seed benchmarks (42, 1337, 2026), and all ablations",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())

    if args.ablation:
        config["ablation_mode"] = args.ablation

    result = run_experiment(config, args.output)

    multi_seed_summary = None
    if args.seeds:
        seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        multi_seed_summary = run_multi_seed_experiment(config, seeds=seed_list, output_dir=args.output)
    elif args.run_all_benchmarks:
        multi_seed_summary = run_multi_seed_experiment(config, seeds=[42, 1337, 2026], output_dir=args.output)

    ablation_summary = None
    if args.run_all_benchmarks:
        ablation_summary = run_ablation_experiments(config, output_dir=args.output)

    if multi_seed_summary or ablation_summary:
        _write_report(
            result,
            args.output / "report.md",
            multi_seed_summary=multi_seed_summary,
            ablation_summary=ablation_summary,
        )

    output_summary = {
        "outcome": result["outcome"],
        "final": result["final"],
        "accepted_versions": result["accepted_versions"],
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
