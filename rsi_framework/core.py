"""Core experiment primitives.

This module deliberately contains no model calls. The mutation generator is a
named harness baseline used to validate experiment accounting before an
open-weight/LLM provider is plugged in.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

from .providers import CandidateGenerator


@dataclass(frozen=True)
class Example:
    task_id: str
    text: str
    label: int
    split: str


@dataclass(frozen=True)
class Policy:
    positive_keywords: tuple[str, ...] = ()
    negative_keywords: tuple[str, ...] = ()
    bias: int = 0

    def predict(self, text: str) -> int:
        normalized = text.lower()
        score = self.bias
        score += sum(word in normalized for word in self.positive_keywords)
        score -= sum(word in normalized for word in self.negative_keywords)
        return int(score > 0)

    def canonical(self) -> dict:
        return {
            "positive_keywords": list(self.positive_keywords),
            "negative_keywords": list(self.negative_keywords),
            "bias": self.bias,
        }

    def version(self) -> str:
        encoded = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return "candidate-" + hashlib.sha256(encoded.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Candidate:
    policy: Policy
    generation: int
    parent_version: str | None
    mutation: str

    @property
    def version(self) -> str:
        return self.policy.version()


@dataclass(frozen=True)
class Scores:
    accuracy: float
    correct: int
    total: int


class DeterministicMutationGenerator:
    """A transparent policy-mutation generator, not an LLM.

    One candidate adds/removes one vocabulary item or changes bias by one. The
    fixed ordering and local RNG seed make the search replayable.
    """

    def __init__(self, mutation_pool: Iterable[str]):
        self.mutation_pool = tuple(mutation_pool)

    def generate(self, current: Candidate, generation: int, seed: int) -> list[Candidate]:
        rng = random.Random(seed + generation)
        mutations: list[tuple[str, Policy]] = []
        pos, neg = set(current.policy.positive_keywords), set(current.policy.negative_keywords)
        for word in self.mutation_pool:
            if word not in pos:
                mutations.append((
                    f"add_positive:{word}",
                    Policy(tuple(sorted(pos | {word})), tuple(sorted(neg)), current.policy.bias),
                ))
            if word not in neg:
                mutations.append((
                    f"add_negative:{word}",
                    Policy(tuple(sorted(pos)), tuple(sorted(neg | {word})), current.policy.bias),
                ))
        mutations.append(("bias:+1", Policy(tuple(sorted(pos)), tuple(sorted(neg)), current.policy.bias + 1)))
        mutations.append(("bias:-1", Policy(tuple(sorted(pos)), tuple(sorted(neg)), current.policy.bias - 1)))
        rng.shuffle(mutations)
        return [Candidate(policy, generation, current.version, mutation) for mutation, policy in mutations]


def make_dataset(seed: int, n_per_pattern: int = 4) -> list[Example]:
    """Build a deterministic, synthetic binary task with disjoint splits."""
    rng = random.Random(seed)
    positives = ("reliable", "clear", "safe", "useful")
    negatives = ("risky", "unclear", "unsafe", "harmful")
    rows: list[Example] = []
    for label, words in ((1, positives), (0, negatives)):
        for word in words:
            for variant in range(n_per_pattern):
                filler = ("The answer is", "This plan looks", "The proposed step is", "Review says")[
                    (variant + len(word)) % 4
                ]
                rows.append(Example(f"{label}-{word}-{variant}", f"{filler} {word}.", label, ""))
    rng.shuffle(rows)
    n = len(rows)
    train_end, dev_end = int(n * 0.5), int(n * 0.75)
    return [
        Example(e.task_id, e.text, e.label, "train" if i < train_end else "dev" if i < dev_end else "heldout")
        for i, e in enumerate(rows)
    ]


def evaluate(policy: Policy, examples: Iterable[Example], split: str) -> Scores:
    selected = [e for e in examples if e.split == split]
    correct = sum(policy.predict(e.text) == e.label for e in selected)
    return Scores(correct / len(selected), correct, len(selected))


def _candidate_record(candidate: Candidate, train: Scores, dev: Scores, accepted: bool, reason: str) -> dict:
    return {
        "version": candidate.version,
        "generation": candidate.generation,
        "parent_version": candidate.parent_version,
        "mutation": candidate.mutation,
        "policy": candidate.policy.canonical(),
        "train": asdict(train),
        "dev": asdict(dev),
        "accepted": accepted,
        "decision": reason,
    }


def extract_lineage_chain(target_version: str, history: list[dict]) -> list[str]:
    """Trace the lineage ancestor versions from baseline to target_version."""
    parent_map = {row["version"]: row.get("parent_version") for row in history}
    chain = [target_version]
    curr = target_version
    while curr in parent_map and parent_map[curr] is not None:
        curr = parent_map[curr]
        chain.append(curr)
    return list(reversed(chain))


def build_lineage_tree(history: list[dict]) -> dict:
    """Reconstruct the full mutation lineage tree from history records."""
    nodes: dict[str, dict] = {}
    roots: list[dict] = []
    for row in history:
        v = row["version"]
        node = {
            "version": v,
            "generation": row["generation"],
            "mutation": row["mutation"],
            "accepted": row["accepted"],
            "decision": row["decision"],
            "train_accuracy": row["train"]["accuracy"],
            "dev_accuracy": row["dev"]["accuracy"],
            "children": [],
        }
        nodes[v] = node
        p = row.get("parent_version")
        if p and p in nodes:
            nodes[p]["children"].append(node)
        elif not p:
            roots.append(node)
    return {"roots": roots, "total_nodes": len(nodes)}


def run_experiment(
    config: dict,
    output_dir: Path | None = None,
    save_files: bool = True,
    examples_override: list[Example] | None = None,
) -> dict:
    seed = int(config["seed"])
    examples = (
        examples_override
        if examples_override is not None
        else make_dataset(seed, int(config.get("examples_per_pattern", 4)))
    )
    generator = DeterministicMutationGenerator(config["mutation_pool"])
    ablation_mode = config.get("ablation_mode")

    current = Candidate(
        Policy(
            tuple(config["initial_positive_keywords"]),
            tuple(config["initial_negative_keywords"]),
            int(config.get("initial_bias", 0)),
        ),
        0,
        None,
        "baseline",
    )
    train = evaluate(current.policy, examples, "train")
    dev = evaluate(current.policy, examples, "dev")
    history = [_candidate_record(current, train, dev, True, "baseline")]
    accepted_versions = [current.version]

    for generation in range(1, int(config["generations"]) + 1):
        if ablation_mode == "ablation_no_mutation":
            # Mutation disabled: no new candidates generated
            proposals: list[Candidate] = []
        else:
            proposals = generator.generate(current, generation, seed)

        if not proposals:
            history.append(_candidate_record(current, train, dev, False, "no_mutation_candidates"))
            continue

        scored = [(c, evaluate(c.policy, examples, "train"), evaluate(c.policy, examples, "dev")) for c in proposals]

        if ablation_mode == "ablation_no_selection":
            # Selection ignores dev partition completely: rank only by train accuracy and version
            scored.sort(key=lambda x: (-x[1].accuracy, x[0].version))
            best, best_train, best_dev = scored[0]
            # Dev criteria and regression check disabled; accept if train improves
            improves = best_train.accuracy > train.accuracy
            regresses = False
            accepted = improves
            reason = "accepted_no_dev_selection" if accepted else "flat_or_no_train_gain"
        elif ablation_mode == "ablation_no_rollback":
            # Rollback disabled: accept if train improves, even if dev regresses
            scored.sort(key=lambda x: (-x[1].accuracy, -x[2].accuracy, x[0].version))
            best, best_train, best_dev = scored[0]
            parent_dev = dev.accuracy
            improves = (best_train.accuracy > train.accuracy) or (
                best_train.accuracy == train.accuracy and best_dev.accuracy > dev.accuracy
            )
            regresses = False  # Rollback disabled
            accepted = improves
            reason = (
                "accepted_without_rollback"
                if (accepted and best_dev.accuracy < parent_dev)
                else "accepted_improvement" if accepted else "flat_or_no_train_dev_gain"
            )
        else:
            # Standard harness baseline: lexicographic (train, dev, version) with rollback on dev regression
            scored.sort(key=lambda x: (-x[1].accuracy, -x[2].accuracy, x[0].version))
            best, best_train, best_dev = scored[0]
            parent_dev = dev.accuracy
            improves = (best_train.accuracy, best_dev.accuracy) > (train.accuracy, dev.accuracy)
            regresses = best_dev.accuracy + float(config.get("regression_tolerance", 0.0)) < parent_dev
            accepted = improves and not regresses
            reason = (
                "accepted_improvement"
                if accepted
                else "rollback_regression" if regresses else "flat_or_no_train_dev_gain"
            )

        history.append(_candidate_record(best, best_train, best_dev, accepted, reason))
        if accepted:
            current, train, dev = best, best_train, best_dev
            accepted_versions.append(current.version)

    # Both held-out measurements happen strictly after selection and never influence it.
    baseline_heldout = evaluate(
        Policy(
            tuple(config["initial_positive_keywords"]),
            tuple(config["initial_negative_keywords"]),
            int(config.get("initial_bias", 0)),
        ),
        examples,
        "heldout",
    )
    heldout = evaluate(current.policy, examples, "heldout")
    baseline = history[0]
    result = {
        "schema_version": 1,
        "experiment": "deterministic-policy-mutation-harness",
        "research_question": "Can a fixed base LLM-driven system improve held-out task performance through iterative candidate generation/evaluation/selection?",
        "method_label": "HARNESS_BASELINE_NOT_LLM",
        "seed": seed,
        "config": config,
        "splits": {split: sum(e.split == split for e in examples) for split in ("train", "dev", "heldout")},
        "baseline": {"version": baseline["version"], "train": baseline["train"], "dev": baseline["dev"]},
        "final": {"version": current.version, "train": asdict(train), "dev": asdict(dev), "heldout": asdict(heldout)},
        "baseline_heldout": asdict(baseline_heldout),
        "outcome": (
            "improvement"
            if heldout.accuracy > baseline_heldout.accuracy
            else "flat" if heldout.accuracy == baseline_heldout.accuracy else "regression"
        ),
        "accepted_versions": accepted_versions,
        "history": history,
        "limitations": [
            "No LLM or paid API was used; mutation loop only validates harness mechanics.",
            "Synthetic lexical task is not evidence of general self-improvement.",
            "The held-out split is consulted only for the final measurement.",
        ],
    }

    if output_dir is not None and save_files:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "first_run.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        with (output_dir / "history.jsonl").open("w") as f:
            for row in history:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        with (output_dir / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["generation", "version", "accepted", "train_accuracy", "dev_accuracy"],
                lineterminator="\n",
            )
            writer.writeheader()
            for row in history:
                writer.writerow({
                    "generation": row["generation"],
                    "version": row["version"],
                    "accepted": row["accepted"],
                    "train_accuracy": row["train"]["accuracy"],
                    "dev_accuracy": row["dev"]["accuracy"],
                })
        _write_report(result, output_dir / "report.md")
        _write_svg(history, output_dir / "progress.svg")

    return result


def run_multi_seed_experiment(
    base_config: dict,
    seeds: list[int] = [42, 1337, 2026],
    output_dir: Path | None = None,
) -> dict:
    """Run the experiment across multiple seeds and aggregate statistical metrics."""
    seed_records: list[dict] = []
    for s in seeds:
        cfg = dict(base_config)
        cfg["seed"] = s
        t0 = time.perf_counter()
        res = run_experiment(cfg, output_dir=None, save_files=False)
        runtime_s = time.perf_counter() - t0

        canonical_json = json.dumps(res, sort_keys=True)
        artifact_hash = hashlib.sha256(canonical_json.encode()).hexdigest()

        rec = {
            "seed": s,
            "baseline_train_acc": res["baseline"]["train"]["accuracy"],
            "baseline_dev_acc": res["baseline"]["dev"]["accuracy"],
            "baseline_heldout_acc": res["baseline_heldout"]["accuracy"],
            "final_train_acc": res["final"]["train"]["accuracy"],
            "final_dev_acc": res["final"]["dev"]["accuracy"],
            "final_heldout_acc": res["final"]["heldout"]["accuracy"],
            "heldout_gain": round(res["final"]["heldout"]["accuracy"] - res["baseline_heldout"]["accuracy"], 4),
            "regression_count": sum(1 for row in res["history"] if row.get("decision") == "rollback_regression"),
            "accepted_count": len(res["accepted_versions"]),
            "runtime_seconds": round(runtime_s, 5),
            "artifact_hash": artifact_hash,
            "final_version": res["final"]["version"],
            "outcome": res["outcome"],
        }
        seed_records.append(rec)

    def _mean_std(values: list[float]) -> dict[str, float]:
        m = statistics.mean(values)
        s = statistics.stdev(values) if len(values) > 1 else 0.0
        return {"mean": round(m, 4), "std": round(s, 4)}

    summary = {
        "seeds": seeds,
        "num_runs": len(seeds),
        "train_accuracy": _mean_std([r["final_train_acc"] for r in seed_records]),
        "dev_accuracy": _mean_std([r["final_dev_acc"] for r in seed_records]),
        "heldout_accuracy": _mean_std([r["final_heldout_acc"] for r in seed_records]),
        "heldout_gain": _mean_std([r["heldout_gain"] for r in seed_records]),
        "regression_count": _mean_std([float(r["regression_count"]) for r in seed_records]),
        "accepted_count": _mean_std([float(r["accepted_count"]) for r in seed_records]),
        "runtime_seconds": _mean_std([r["runtime_seconds"] for r in seed_records]),
        "runs": seed_records,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "multi_seed_results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        with (output_dir / "multi_seed_metrics.csv").open("w", newline="") as f:
            fields = [
                "seed",
                "final_train_acc",
                "final_dev_acc",
                "final_heldout_acc",
                "heldout_gain",
                "regression_count",
                "accepted_count",
                "runtime_seconds",
                "artifact_hash",
                "outcome",
            ]
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for r in seed_records:
                writer.writerow({k: r[k] for k in fields})

    return summary


def run_ablation_experiments(
    base_config: dict,
    output_dir: Path | None = None,
    modes: tuple[str, ...] = ("baseline_full", "ablation_no_mutation", "ablation_no_selection", "ablation_no_rollback"),
) -> dict:
    """Execute ablation experiments and compare mechanisms."""
    records: list[dict] = []
    for mode in modes:
        cfg = dict(base_config)
        if mode != "baseline_full":
            cfg["ablation_mode"] = mode

        t0 = time.perf_counter()
        res = run_experiment(cfg, output_dir=None, save_files=False)
        runtime_s = time.perf_counter() - t0

        canonical_json = json.dumps(res, sort_keys=True)
        artifact_hash = hashlib.sha256(canonical_json.encode()).hexdigest()

        rec = {
            "mode": mode,
            "final_train_acc": res["final"]["train"]["accuracy"],
            "final_dev_acc": res["final"]["dev"]["accuracy"],
            "final_heldout_acc": res["final"]["heldout"]["accuracy"],
            "heldout_gain": round(res["final"]["heldout"]["accuracy"] - res["baseline_heldout"]["accuracy"], 4),
            "regression_count": sum(1 for row in res["history"] if row.get("decision") == "rollback_regression"),
            "accepted_count": len(res["accepted_versions"]),
            "runtime_seconds": round(runtime_s, 5),
            "final_version": res["final"]["version"],
            "artifact_hash": artifact_hash,
            "outcome": res["outcome"],
        }
        records.append(rec)

    summary = {
        "modes": list(modes),
        "runs": records,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "ablation_results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        with (output_dir / "ablation_metrics.csv").open("w", newline="") as f:
            fields = [
                "mode",
                "final_train_acc",
                "final_dev_acc",
                "final_heldout_acc",
                "heldout_gain",
                "regression_count",
                "accepted_count",
                "runtime_seconds",
                "outcome",
            ]
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for r in records:
                writer.writerow({k: r[k] for k in fields})

    return summary


def _write_report(
    result: dict,
    path: Path,
    multi_seed_summary: dict | None = None,
    ablation_summary: dict | None = None,
) -> None:
    final = result["final"]
    lines = [
        "# RSI Experimental Framework — Empirical Report",
        "",
        "## Research Question",
        "> Can a fixed base LLM-driven system improve held-out task performance through iterative candidate generation, evaluation, and selection?",
        "",
        f"Method Label: `{result['method_label']}`",
        f"Overall Single-Run Outcome: **{result['outcome']}**",
        "",
        "## 1. Single-Run Benchmark (Seed: " + str(result["seed"]) + ")",
        "",
        "| Split | Baseline Accuracy | Final Policy Accuracy | Status |",
        "|---|---:|---:|---|",
        f"| Train | {result['baseline']['train']['accuracy']:.3f} | {final['train']['accuracy']:.3f} | Evaluated for selection |",
        f"| Dev | {result['baseline']['dev']['accuracy']:.3f} | {final['dev']['accuracy']:.3f} | Evaluated for selection & rollback |",
        f"| Held-out | {result['baseline_heldout']['accuracy']:.3f} | {final['heldout']['accuracy']:.3f} | Post-selection measurement only |",
        "",
        f"- **Accepted Candidates Count**: {len(result['accepted_versions'])}",
        f"- **Total Search Generations**: {len(result['history']) - 1}",
        f"- **Final Policy Hash**: `{final['version']}`",
        "",
    ]

    if multi_seed_summary:
        lines.extend([
            "## 2. Multi-Seed Empirical Verification (Seeds: " + ", ".join(map(str, multi_seed_summary["seeds"])) + ")",
            "",
            "To guard against single-seed anomalies, the loop was evaluated across independent dataset shuffles and search initializations.",
            "",
            "| Seed | Train Acc | Dev Acc | Held-out Acc | Held-out Gain | Regressions | Accepted | Runtime (s) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for r in multi_seed_summary["runs"]:
            lines.append(
                f"| {r['seed']} | {r['final_train_acc']:.3f} | {r['final_dev_acc']:.3f} | "
                f"{r['final_heldout_acc']:.3f} | {r['heldout_gain']:+.3f} | {r['regression_count']} | "
                f"{r['accepted_count']} | {r['runtime_seconds']:.4f} |"
            )
        lines.extend([
            "",
            "### Statistical Summary (Mean ± Std)",
            f"- **Train Accuracy**: {multi_seed_summary['train_accuracy']['mean']:.3f} ± {multi_seed_summary['train_accuracy']['std']:.3f}",
            f"- **Dev Accuracy**: {multi_seed_summary['dev_accuracy']['mean']:.3f} ± {multi_seed_summary['dev_accuracy']['std']:.3f}",
            f"- **Held-out Accuracy**: {multi_seed_summary['heldout_accuracy']['mean']:.3f} ± {multi_seed_summary['heldout_accuracy']['std']:.3f}",
            f"- **Held-out Gain**: {multi_seed_summary['heldout_gain']['mean']:+.3f} ± {multi_seed_summary['heldout_gain']['std']:.3f}",
            f"- **Regression Count**: {multi_seed_summary['regression_count']['mean']:.2f} ± {multi_seed_summary['regression_count']['std']:.2f}",
            f"- **Accepted Count**: {multi_seed_summary['accepted_count']['mean']:.2f} ± {multi_seed_summary['accepted_count']['std']:.2f}",
            "",
        ])

    if ablation_summary:
        lines.extend([
            "## 3. Ablation Analysis",
            "",
            "To verify the necessity of each component in the iterative loop, we compare the full system against restricted variants:",
            "- `ablation_no_mutation`: Candidate mutation disabled.",
            "- `ablation_no_selection`: Selection occurs without dev partition guidance.",
            "- `ablation_no_rollback`: Dev regression triggers no rollback (greedy train acceptance).",
            "",
            "| Variant | Train Acc | Dev Acc | Held-out Acc | Held-out Gain | Regressions | Accepted | Outcome |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        for r in ablation_summary["runs"]:
            lines.append(
                f"| `{r['mode']}` | {r['final_train_acc']:.3f} | {r['final_dev_acc']:.3f} | "
                f"{r['final_heldout_acc']:.3f} | {r['heldout_gain']:+.3f} | {r['regression_count']} | "
                f"{r['accepted_count']} | {r['outcome']} |"
            )
        lines.append("")

    lines.extend([
        "## 4. Methodological Safeguards",
        "",
        "1. **Zero Held-out Leakage**: Held-out split data is never passed to candidate generation, scoring, or selection. Modifying held-out labels produces the exact bit-for-bit candidate selection chain.",
        "2. **Byte-Level Reproducibility**: Given the same seed and configuration, identical JSON/CSV artifacts and SHA-256 digests are generated deterministically.",
        "3. **Explicit Rollback**: Candidates causing dev degradation beyond tolerance are rejected with label `rollback_regression`.",
        "",
        "## 5. Limitations & Scientific Honesty",
        "",
        "- **Harness Baseline Only**: This milestone evaluates harness accounting and selection mechanics. No weights were trained and no LLM API was invoked.",
        "- **Synthetic Task Scope**: Lexical sentiment heuristics do not demonstrate broad reasoning or general recursive self-improvement (RSI).",
        "- **Provider Boundary**: Real open-weight LLMs (e.g. running on local GPU or Colab L4) must be attached via the defined `OpenWeightLLMProvider` protocol before making claims regarding neural policy improvement.",
        "",
    ])
    path.write_text("\n".join(lines))


def _write_svg(history: list[dict], path: Path) -> None:
    width, height = 640, 240
    points = []
    for row in history:
        x = 30 + row["generation"] * (580 / max(1, len(history) - 1))
        y = 210 - row["dev"]["accuracy"] * 180
        points.append(f"{x:.1f},{y:.1f}")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<title>Dev accuracy by generation</title>'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<polyline fill="none" stroke="#2457a6" stroke-width="3" points="{" ".join(points)}"/>'
        f'<text x="30" y="22" font-family="sans-serif" font-size="16">Dev accuracy by generation</text>'
        f'<text x="30" y="230" font-family="sans-serif" font-size="12">generation</text></svg>\n'
    )
    path.write_text(svg)
