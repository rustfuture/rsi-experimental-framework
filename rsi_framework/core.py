"""Core experiment primitives.

This module deliberately contains no model calls.  The mutation generator is a
named harness baseline used to validate experiment accounting before an
open-weight/LLM provider is plugged in.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol


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
        return {"positive_keywords": list(self.positive_keywords),
                "negative_keywords": list(self.negative_keywords), "bias": self.bias}

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


class CandidateGenerator(Protocol):
    def generate(self, current: Candidate, generation: int, seed: int) -> list[Candidate]: ...


class DeterministicMutationGenerator:
    """A transparent policy-mutation generator, not an LLM.

    One candidate adds/removes one vocabulary item or changes bias by one.  The
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
                mutations.append((f"add_positive:{word}", Policy(tuple(sorted(pos | {word})), tuple(sorted(neg)), current.policy.bias)))
            if word not in neg:
                mutations.append((f"add_negative:{word}", Policy(tuple(sorted(pos)), tuple(sorted(neg | {word})), current.policy.bias)))
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
                filler = ("The answer is", "This plan looks", "The proposed step is", "Review says")[(variant + len(word)) % 4]
                rows.append(Example(f"{label}-{word}-{variant}", f"{filler} {word}.", label, ""))
    rng.shuffle(rows)
    n = len(rows)
    train_end, dev_end = int(n * 0.5), int(n * 0.75)
    return [Example(e.task_id, e.text, e.label, "train" if i < train_end else "dev" if i < dev_end else "heldout")
            for i, e in enumerate(rows)]


def evaluate(policy: Policy, examples: Iterable[Example], split: str) -> Scores:
    selected = [e for e in examples if e.split == split]
    correct = sum(policy.predict(e.text) == e.label for e in selected)
    return Scores(correct / len(selected), correct, len(selected))


def _candidate_record(candidate: Candidate, train: Scores, dev: Scores, accepted: bool, reason: str) -> dict:
    return {"version": candidate.version, "generation": candidate.generation,
            "parent_version": candidate.parent_version, "mutation": candidate.mutation,
            "policy": candidate.policy.canonical(), "train": asdict(train), "dev": asdict(dev),
            "accepted": accepted, "decision": reason}


def run_experiment(config: dict, output_dir: Path) -> dict:
    seed = int(config["seed"])
    examples = make_dataset(seed, int(config.get("examples_per_pattern", 4)))
    generator = DeterministicMutationGenerator(config["mutation_pool"])
    current = Candidate(Policy(tuple(config["initial_positive_keywords"]), tuple(config["initial_negative_keywords"]), int(config.get("initial_bias", 0))), 0, None, "baseline")
    train, dev = evaluate(current.policy, examples, "train"), evaluate(current.policy, examples, "dev")
    history = [_candidate_record(current, train, dev, True, "baseline")]
    accepted_versions = [current.version]
    for generation in range(1, int(config["generations"]) + 1):
        proposals = generator.generate(current, generation, seed)
        scored = [(c, evaluate(c.policy, examples, "train"), evaluate(c.policy, examples, "dev")) for c in proposals]
        scored.sort(key=lambda x: (-x[1].accuracy, -x[2].accuracy, x[0].version))
        best, best_train, best_dev = scored[0]
        parent_dev = dev.accuracy
        improves = (best_train.accuracy, best_dev.accuracy) > (train.accuracy, dev.accuracy)
        regresses = best_dev.accuracy + float(config.get("regression_tolerance", 0.0)) < parent_dev
        accepted = improves and not regresses
        reason = "accepted_improvement" if accepted else "rollback_regression" if regresses else "flat_or_no_train_dev_gain"
        history.append(_candidate_record(best, best_train, best_dev, accepted, reason))
        if accepted:
            current, train, dev = best, best_train, best_dev
            accepted_versions.append(current.version)
    # Both held-out measurements happen after selection and never influence it.
    baseline_heldout = evaluate(Policy(tuple(config["initial_positive_keywords"]), tuple(config["initial_negative_keywords"]), int(config.get("initial_bias", 0))), examples, "heldout")
    heldout = evaluate(current.policy, examples, "heldout")
    baseline = history[0]
    result = {
        "schema_version": 1, "experiment": "deterministic-policy-mutation-harness",
        "research_question": "Can a fixed base LLM-driven system improve held-out task performance through iterative candidate generation/evaluation/selection?",
        "method_label": "HARNESS_BASELINE_NOT_LLM", "seed": seed, "config": config,
        "splits": {split: sum(e.split == split for e in examples) for split in ("train", "dev", "heldout")},
        "baseline": {"version": baseline["version"], "train": baseline["train"], "dev": baseline["dev"]},
        "final": {"version": current.version, "train": asdict(train), "dev": asdict(dev), "heldout": asdict(heldout)},
        "baseline_heldout": asdict(baseline_heldout),
        "outcome": "improvement" if heldout.accuracy > baseline_heldout.accuracy else "flat" if heldout.accuracy == baseline_heldout.accuracy else "regression",
        "accepted_versions": accepted_versions, "history": history,
        "limitations": ["No LLM or paid API was used; mutation loop only validates harness mechanics.", "Synthetic lexical task is not evidence of general self-improvement.", "The held-out split is consulted only for the final measurement."]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "first_run.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output_dir / "history.jsonl").open("w") as f:
        for row in history:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["generation", "version", "accepted", "train_accuracy", "dev_accuracy"], lineterminator="\n")
        writer.writeheader()
        for row in history:
            writer.writerow({"generation": row["generation"], "version": row["version"], "accepted": row["accepted"], "train_accuracy": row["train"]["accuracy"], "dev_accuracy": row["dev"]["accuracy"]})
    _write_report(result, output_dir / "report.md")
    _write_svg(history, output_dir / "progress.svg")
    return result


def _write_report(result: dict, path: Path) -> None:
    final = result["final"]
    lines = ["# First deterministic run", "", f"Outcome: **{result['outcome']}**", "", f"Method: `{result['method_label']}`", "", "| split | baseline | final |", "|---|---:|---:|", f"| train | {result['baseline']['train']['accuracy']:.3f} | {final['train']['accuracy']:.3f} |", f"| dev | {result['baseline']['dev']['accuracy']:.3f} | {final['dev']['accuracy']:.3f} |", f"| heldout (post-selection measurements) | {result['baseline_heldout']['accuracy']:.3f} | {final['heldout']['accuracy']:.3f} |", "", "The result is a harness validation, not evidence that an LLM improved itself.", ""]
    path.write_text("\n".join(lines))


def _write_svg(history: list[dict], path: Path) -> None:
    width, height = 640, 240
    points = []
    for row in history:
        x = 30 + row["generation"] * (580 / max(1, len(history) - 1))
        y = 210 - row["dev"]["accuracy"] * 180
        points.append(f"{x:.1f},{y:.1f}")
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><title>Dev accuracy by generation</title><rect width="100%" height="100%" fill="white"/><polyline fill="none" stroke="#2457a6" stroke-width="3" points="{" ".join(points)}"/><text x="30" y="22" font-family="sans-serif" font-size="16">Dev accuracy by generation</text><text x="30" y="230" font-family="sans-serif" font-size="12">generation</text></svg>\n'
    path.write_text(svg)
