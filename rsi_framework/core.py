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
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .providers import CandidateGenerator, ProposalError, validate_proposal

#: Bumped whenever the experiment *definition* changes (matching rule, decision
#: rule, or candidate mechanics). Results produced under different versions must
#: never be pooled or compared as if they came from the same experiment.
EXPERIMENT_VERSION = "v2-token-match"

#: The selection rule rejects a candidate; it does not revert an already-applied
#: policy state. The historical label ``rollback_regression`` is kept as an alias
#: for reading v1 archives (see ``results/archive-v1/``).
ROLLBACK_DECISION = "rejected_dev_regression"
LEGACY_ROLLBACK_DECISION = "rollback_regression"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> frozenset[str]:
    """Lowercase word tokens, so ``safe`` never matches inside ``unsafe``."""
    return frozenset(_TOKEN_RE.findall(text.lower()))


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
        tokens = tokenize(text)
        score = self.bias
        score += sum(word in tokens for word in self.positive_keywords)
        score -= sum(word in tokens for word in self.negative_keywords)
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
    """A proposed policy plus its provenance.

    ``version`` identifies the *policy* (canonical hash) and may legitimately
    repeat across generations or with different parents. ``event_id`` identifies
    the *occurrence* of that proposal in one run's history and is therefore
    unique. Lineage reconstruction keys on the event, never on the version.
    """

    policy: Policy
    generation: int
    parent_version: str | None
    mutation: str
    event_id: str | None = None

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
            # A word may live in exactly one polarity: a word in both sets would
            # cancel itself out and silently produce a no-op candidate.
            if word not in pos and word not in neg:
                mutations.append((
                    f"add_positive:{word}",
                    Policy(tuple(sorted(pos | {word})), tuple(sorted(neg)), current.policy.bias),
                ))
            if word not in neg and word not in pos:
                mutations.append((
                    f"add_negative:{word}",
                    Policy(tuple(sorted(pos)), tuple(sorted(neg | {word})), current.policy.bias),
                ))
        mutations.append(("bias:+1", Policy(tuple(sorted(pos)), tuple(sorted(neg)), current.policy.bias + 1)))
        mutations.append(("bias:-1", Policy(tuple(sorted(pos)), tuple(sorted(neg)), current.policy.bias - 1)))
        rng.shuffle(mutations)
        return [Candidate(policy, generation, current.version, mutation) for mutation, policy in mutations]


def config_hash(config: dict) -> str:
    """Canonical hash of the run configuration, including experiment version."""
    payload = {"experiment_version": EXPERIMENT_VERSION, "config": config}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def dataset_hash(examples: Iterable[Example]) -> str:
    """Canonical hash of the evaluation data actually used by a run."""
    rows = sorted((e.split, e.task_id, e.text, e.label) for e in examples)
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


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


def _candidate_record(
    candidate: Candidate,
    train: Scores,
    dev: Scores,
    accepted: bool,
    reason: str,
    event_id: str,
    parent_event_id: str | None,
) -> dict:
    return {
        "event_id": event_id,
        "version": candidate.version,
        "generation": candidate.generation,
        "parent_version": candidate.parent_version,
        "parent_event_id": parent_event_id,
        "mutation": candidate.mutation,
        "policy": candidate.policy.canonical(),
        "train": asdict(train),
        "dev": asdict(dev),
        "accepted": accepted,
        "decision": reason,
    }


#: Which mechanism each ablation switches off. Kept next to the runner so the
#: report and the CLI cannot drift from the implemented behaviour.
ABLATION_MECHANISMS: dict[str, str] = {
    "baseline_full": "Unmodified loop: dev-aware lexicographic selection with dev-regression rejection.",
    "ablation_no_mutation": "Candidate proposal disabled (generator never called); isolates the contribution of mutation.",
    "ablation_no_selection": "Dev-blind greedy selection: candidates ranked by train accuracy (and version) only. Dev is still measured but cannot influence the pick. This is a weaker selection rule, not the absence of selection.",
    "ablation_no_rollback": "Dev-regression rejection disabled: the best candidate by (train, dev, version) is accepted even when dev regresses. Selection is otherwise unchanged.",
    "ablation_random_selection": "Score-free selection: one candidate per generation is chosen by a seed-controlled RNG and accepted unconditionally; no train or dev score participates in the choice.",
}


def extract_lineage_events(target: str, history: list[dict]) -> list[str]:
    """Trace ancestor *event ids* from the root event to ``target``.

    ``target`` may be an ``event_id`` or a policy ``version`` (the most recent
    event carrying that version is used). The walk is cycle-safe: if a malformed
    history links a node back to one of its ancestors the walk stops and the
    caller can detect the truncation by comparing against
    :func:`lineage_integrity`.
    """
    by_event = {row["event_id"]: row for row in history if "event_id" in row}
    by_version: dict[str, list[str]] = {}
    for row in history:
        if "event_id" in row:
            by_version.setdefault(row["version"], []).append(row["event_id"])

    if target in by_event:
        start = target
    elif target in by_version:
        start = by_version[target][-1]
    else:
        return [target]

    chain: list[str] = [start]
    seen = {start}
    current = start
    while True:
        row = by_event.get(current)
        if row is None:
            break
        parent = row.get("parent_event_id")
        if parent is None:
            break
        if parent in seen:
            # Cycle: stop rather than loop forever.
            break
        seen.add(parent)
        chain.append(parent)
        current = parent
    return list(reversed(chain))


def lineage_integrity(history: list[dict]) -> dict:
    """Report duplicate versions, orphaned parents and cycles in a history."""
    by_event = {row["event_id"]: row for row in history if "event_id" in row}
    version_counts: dict[str, int] = {}
    for row in history:
        version_counts[row["version"]] = version_counts.get(row["version"], 0) + 1

    orphans: list[str] = []
    cycles: list[str] = []
    for event_id, row in by_event.items():
        parent = row.get("parent_event_id")
        if parent is not None and parent not in by_event:
            orphans.append(event_id)
        seen = {event_id}
        current = parent
        while current is not None and current in by_event:
            if current in seen:
                cycles.append(event_id)
                break
            seen.add(current)
            current = by_event[current].get("parent_event_id")

    return {
        "total_events": len(by_event),
        "duplicate_version_count": sum(1 for c in version_counts.values() if c > 1),
        "orphan_event_ids": sorted(orphans),
        "cyclic_event_ids": sorted(set(cycles)),
    }


def extract_lineage_chain(target_version: str, history: list[dict]) -> list[str]:
    """Trace ancestor policy *versions* from the root to ``target_version``."""
    events = extract_lineage_events(target_version, history)
    by_event = {row["event_id"]: row for row in history if "event_id" in row}
    versions = [by_event[e]["version"] if e in by_event else e for e in events]
    return versions


def build_lineage_tree(history: list[dict]) -> dict:
    """Reconstruct the mutation lineage tree from history records.

    Nodes are keyed by ``event_id`` so that the same policy version appearing in
    several generations (or under several parents) is preserved instead of being
    overwritten. Parents that are absent from the history are attached to an
    ``orphans`` list rather than being dropped silently.
    """
    nodes: dict[str, dict] = {}
    roots: list[dict] = []
    orphans: list[dict] = []

    for row in history:
        event_id = row.get("event_id")
        if event_id is None:
            # Legacy v1 records have no event id; synthesise a stable one.
            event_id = f"legacy-g{row['generation']}-{row['version']}"
        node = {
            "event_id": event_id,
            "version": row["version"],
            "generation": row["generation"],
            "mutation": row["mutation"],
            "accepted": row["accepted"],
            "decision": row["decision"],
            "parent_version": row.get("parent_version"),
            "parent_event_id": row.get("parent_event_id"),
            "train_accuracy": row["train"]["accuracy"],
            "dev_accuracy": row["dev"]["accuracy"],
            "children": [],
        }
        nodes[event_id] = node

    for node in nodes.values():
        parent_event = node["parent_event_id"]
        if parent_event is not None and parent_event in nodes:
            nodes[parent_event]["children"].append(node)
            continue
        if node["parent_version"] is None:
            roots.append(node)
            continue
        orphans.append(node)

    integrity = lineage_integrity(
        [
            {
                "event_id": node["event_id"],
                "version": node["version"],
                "parent_event_id": node["parent_event_id"],
            }
            for node in nodes.values()
        ]
    )
    return {
        "roots": roots,
        "orphans": orphans,
        "total_nodes": len(nodes),
        "duplicate_version_count": integrity["duplicate_version_count"],
        "cyclic_event_ids": integrity["cyclic_event_ids"],
    }


def run_experiment(
    config: dict,
    output_dir: Path | None = None,
    save_files: bool = True,
    examples_override: list[Example] | None = None,
    candidate_generator: CandidateGenerator | None = None,
) -> dict:
    seed = int(config["seed"])
    examples = (
        examples_override
        if examples_override is not None
        else make_dataset(seed, int(config.get("examples_per_pattern", 4)))
    )
    # Injection point: the harness never hard-codes the deterministic generator.
    # A provider (local model, JSON proposal file, test fake) can be supplied.
    generator = (
        candidate_generator
        if candidate_generator is not None
        else DeterministicMutationGenerator(config["mutation_pool"])
    )
    ablation_mode = config.get("ablation_mode")
    allowed_vocabulary = config.get("provider_keyword_vocabulary")
    max_edits = int(config.get("provider_max_policy_edits", 1))
    bias_bound = int(config.get("provider_bias_bound", 5))

    current = Candidate(
        Policy(
            tuple(config["initial_positive_keywords"]),
            tuple(config["initial_negative_keywords"]),
            int(config.get("initial_bias", 0)),
        ),
        0,
        None,
        "baseline",
        event_id="g0",
    )
    train = evaluate(current.policy, examples, "train")
    dev = evaluate(current.policy, examples, "dev")
    history = [_candidate_record(current, train, dev, True, "baseline", "g0", None)]
    accepted_versions = [current.version]      # includes the baseline candidate
    accepted_new_changes: list[str] = []       # excludes the baseline candidate
    rejected_proposals: list[dict] = []
    # The event that currently holds the active policy. Only an accepted candidate
    # advances it, so a rejected candidate can never become a recorded parent.
    active_event_id = "g0"

    for generation in range(1, int(config["generations"]) + 1):
        if ablation_mode == "ablation_no_mutation":
            # Mutation disabled: no new candidates generated
            raw_proposals: list[Candidate] = []
        else:
            raw_proposals = list(generator.generate(current, generation, seed))
            rejected_proposals.extend(getattr(generator, "rejections", []))

        # Validate every proposal before it can influence selection. Malformed or
        # out-of-policy mutations are rejected and never scored.
        proposals: list[Candidate] = []
        for index, candidate in enumerate(raw_proposals):
            event_id = f"g{generation}-c{index}"
            candidate = Candidate(
                candidate.policy,
                generation,
                candidate.parent_version,
                candidate.mutation,
                event_id=candidate.event_id or event_id,
            )
            try:
                validate_proposal(
                    current,
                    candidate.policy,
                    allowed_vocabulary=allowed_vocabulary,
                    max_edits=max_edits,
                    bias_bound=bias_bound,
                )
            except ProposalError as exc:
                rejected_proposals.append({
                    "event_id": candidate.event_id,
                    "generation": generation,
                    "version": candidate.version,
                    "mutation": candidate.mutation,
                    "reason": str(exc),
                })
                continue
            proposals.append(candidate)

        if not proposals:
            reason = "no_mutation_candidates" if not raw_proposals else "no_valid_proposals"
            history.append(
                _candidate_record(current, train, dev, False, reason, f"g{generation}", active_event_id)
            )
            continue

        scored = [(c, evaluate(c.policy, examples, "train"), evaluate(c.policy, examples, "dev")) for c in proposals]

        if ablation_mode == "ablation_random_selection":
            # Score-free control: pick one candidate with a seed-controlled RNG and
            # accept it unconditionally. No train or dev score enters the choice.
            picker = random.Random(seed * 1000003 + generation)
            best, best_train, best_dev = picker.choice(scored)
            accepted = True
            reason = "accepted_random_selection"
        elif ablation_mode == "ablation_no_selection":
            # Dev-blind greedy selection: rank by train accuracy and version only.
            scored.sort(key=lambda x: (-x[1].accuracy, x[0].version))
            best, best_train, best_dev = scored[0]
            improves = best_train.accuracy > train.accuracy
            accepted = improves
            reason = "accepted_dev_blind_train_gain" if accepted else "flat_or_no_train_gain"
        elif ablation_mode == "ablation_no_rollback":
            # Dev-regression rejection disabled: accept the best candidate even if dev regresses.
            scored.sort(key=lambda x: (-x[1].accuracy, -x[2].accuracy, x[0].version))
            best, best_train, best_dev = scored[0]
            parent_dev = dev.accuracy
            improves = (best_train.accuracy > train.accuracy) or (
                best_train.accuracy == train.accuracy and best_dev.accuracy > dev.accuracy
            )
            accepted = improves
            reason = (
                "accepted_without_rollback"
                if (accepted and best_dev.accuracy < parent_dev)
                else "accepted_improvement" if accepted else "flat_or_no_train_dev_gain"
            )
        else:
            # Standard harness baseline: lexicographic (train, dev, version) with a
            # fixed-tolerance dev-regression rejection (reject the candidate; the
            # applied policy state is never mutated, so nothing is rolled back).
            scored.sort(key=lambda x: (-x[1].accuracy, -x[2].accuracy, x[0].version))
            best, best_train, best_dev = scored[0]
            parent_dev = dev.accuracy
            improves = (best_train.accuracy, best_dev.accuracy) > (train.accuracy, dev.accuracy)
            regresses = best_dev.accuracy + float(config.get("regression_tolerance", 0.0)) < parent_dev
            accepted = improves and not regresses
            reason = (
                "accepted_improvement"
                if accepted
                else ROLLBACK_DECISION if regresses else "flat_or_no_train_dev_gain"
            )

        history.append(
            _candidate_record(best, best_train, best_dev, accepted, reason, best.event_id, active_event_id)
        )
        if accepted:
            current, train, dev = best, best_train, best_dev
            active_event_id = best.event_id
            accepted_versions.append(current.version)
            accepted_new_changes.append(current.version)

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
        "schema_version": 2,
        "experiment_version": EXPERIMENT_VERSION,
        "experiment": "deterministic-policy-mutation-harness",
        "research_question": "Can a fixed base LLM-driven system improve held-out task performance through iterative candidate generation/evaluation/selection?",
        "method_label": "HARNESS_BASELINE_NOT_LLM",
        "seed": seed,
        "config_hash": config_hash(config),
        "dataset_hash": dataset_hash(examples),
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
        "accepted_baseline_included_count": len(accepted_versions),
        "accepted_new_changes": accepted_new_changes,
        "accepted_new_change_count": len(accepted_new_changes),
        "rejected_proposals": rejected_proposals,
        "rejected_proposal_count": len(rejected_proposals),
        "history": history,
        "limitations": [
            "No LLM or paid API was used; mutation loop only validates harness mechanics.",
            "Synthetic lexical task is not evidence of general self-improvement.",
            "The held-out split is consulted only for the final measurement.",
            "All seeds re-split the same 32-row synthetic sentence pool; runs are replays, not independent real-world datasets.",
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
        _write_svg(history, output_dir / "progress.svg")

    return result


def run_multi_seed_experiment(
    base_config: dict,
    seeds: list[int] = [42, 1337, 2026],
    output_dir: Path | None = None,
) -> dict:
    """Run the experiment across multiple seeds and aggregate statistical metrics.

    Honest scope: every seed re-shuffles the *same* fixed synthetic sentence pool
    (``make_dataset`` builds an identical row set and only shuffles it), so these
    runs are repeated replays of one toy dataset rather than independent
    real-world samples. ``heldout.n`` is therefore the true sample count, not
    ``num_runs * heldout.n``.
    """
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
            "regression_count": sum(
                1
                for row in res["history"]
                if row.get("decision") in (ROLLBACK_DECISION, LEGACY_ROLLBACK_DECISION)
            ),
            "accepted_baseline_included_count": res["accepted_baseline_included_count"],
            "accepted_new_change_count": res["accepted_new_change_count"],
            "rejected_proposal_count": res["rejected_proposal_count"],
            "heldout_correct": res["final"]["heldout"]["correct"],
            "heldout_total": res["final"]["heldout"]["total"],
            "train_total": res["splits"]["train"],
            "dev_total": res["splits"]["dev"],
            "config_hash": res["config_hash"],
            "dataset_hash": res["dataset_hash"],
            "experiment_version": res["experiment_version"],
            "runtime_seconds": round(runtime_s, 5),
            "artifact_hash": artifact_hash,
            "final_version": res["final"]["version"],
            "outcome": res["outcome"],
        }
        seed_records.append(rec)

    def _mean_std(values: list[float]) -> dict[str, float]:
        m = statistics.mean(values)
        s = statistics.stdev(values) if len(values) > 1 else 0.0
        return {"mean": round(m, 4), "std": round(s, 4), "n": len(values), "stdev_ddof": 1}

    summary = {
        "experiment_version": EXPERIMENT_VERSION,
        "seeds": seeds,
        "num_runs": len(seeds),
        "train_accuracy": _mean_std([r["final_train_acc"] for r in seed_records]),
        "dev_accuracy": _mean_std([r["final_dev_acc"] for r in seed_records]),
        "heldout_accuracy": _mean_std([r["final_heldout_acc"] for r in seed_records]),
        "heldout_gain": _mean_std([r["heldout_gain"] for r in seed_records]),
        "regression_count": _mean_std([float(r["regression_count"]) for r in seed_records]),
        "accepted_baseline_included_count": _mean_std(
            [float(r["accepted_baseline_included_count"]) for r in seed_records]
        ),
        "accepted_new_changes": _mean_std([float(r["accepted_new_change_count"]) for r in seed_records]),
        "rejected_proposal_count": _mean_std([float(r["rejected_proposal_count"]) for r in seed_records]),
        "runtime_seconds": _mean_std([r["runtime_seconds"] for r in seed_records]),
        "heldout_examples_per_seed": seed_records[0]["heldout_total"] if seed_records else 0,
        "train_examples_per_seed": seed_records[0]["train_total"] if seed_records else 0,
        "dev_examples_per_seed": seed_records[0]["dev_total"] if seed_records else 0,
        "same_synthetic_pool_across_seeds": True,
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
                "accepted_baseline_included_count",
                "accepted_new_change_count",
                "rejected_proposal_count",
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
    modes: tuple[str, ...] = (
        "baseline_full",
        "ablation_no_mutation",
        "ablation_no_selection",
        "ablation_no_rollback",
        "ablation_random_selection",
    ),
) -> dict:
    """Execute ablation experiments and compare mechanisms.

    Every mode runs with the identical seed, dataset (same ``examples`` built
    from the same config seed) and proposal-generation budget; only the named
    mechanism is switched off. See :data:`ABLATION_MECHANISMS`.
    """
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
            "mechanism": ABLATION_MECHANISMS.get(mode, "unspecified mechanism"),
            "seed": res["seed"],
            "dataset_hash": res["dataset_hash"],
            "config_hash": res["config_hash"],
            "generations": int(cfg["generations"]),
            "final_train_acc": res["final"]["train"]["accuracy"],
            "final_dev_acc": res["final"]["dev"]["accuracy"],
            "final_heldout_acc": res["final"]["heldout"]["accuracy"],
            "heldout_gain": round(res["final"]["heldout"]["accuracy"] - res["baseline_heldout"]["accuracy"], 4),
            "regression_count": sum(
                1
                for row in res["history"]
                if row.get("decision") in (ROLLBACK_DECISION, LEGACY_ROLLBACK_DECISION)
            ),
            "accepted_baseline_included_count": res["accepted_baseline_included_count"],
            "accepted_new_change_count": res["accepted_new_change_count"],
            "rejected_proposal_count": res["rejected_proposal_count"],
            "runtime_seconds": round(runtime_s, 5),
            "final_version": res["final"]["version"],
            "artifact_hash": artifact_hash,
            "outcome": res["outcome"],
        }
        records.append(rec)

    summary = {
        "experiment_version": EXPERIMENT_VERSION,
        "modes": list(modes),
        "mechanisms": {m: ABLATION_MECHANISMS.get(m, "unspecified mechanism") for m in modes},
        "comparable_seed_data_budget": all(
            r["dataset_hash"] == records[0]["dataset_hash"] and r["seed"] == records[0]["seed"]
            for r in records
        ),
        "runs": records,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "ablation_results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        with (output_dir / "ablation_metrics.csv").open("w", newline="") as f:
            fields = [
                "mode",
                "mechanism",
                "final_train_acc",
                "final_dev_acc",
                "final_heldout_acc",
                "heldout_gain",
                "regression_count",
                "accepted_baseline_included_count",
                "accepted_new_change_count",
                "rejected_proposal_count",
                "runtime_seconds",
                "outcome",
            ]
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for r in records:
                writer.writerow({k: r[k] for k in fields})

    return summary


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
