import argparse
import json
from pathlib import Path

from .core import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic RSI harness baseline")
    parser.add_argument("--config", type=Path, default=Path("config/default.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = run_experiment(config, args.output)
    print(json.dumps({"outcome": result["outcome"], "final": result["final"], "accepted_versions": result["accepted_versions"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
