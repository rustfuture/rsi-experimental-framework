import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rsi_framework.core import (
    EXPERIMENT_VERSION,
    ROLLBACK_DECISION,
    Candidate,
    DeterministicMutationGenerator,
    Example,
    Policy,
    build_lineage_tree,
    extract_lineage_chain,
    extract_lineage_events,
    lineage_integrity,
    make_dataset,
    run_ablation_experiments,
    run_experiment,
    run_multi_seed_experiment,
    tokenize,
)
from rsi_framework.providers import (
    JsonFileProposalProvider,
    OpenWeightColabL4Stub,
    ProposalError,
    validate_proposal,
)
from rsi_framework.reporting import (
    README_BEGIN,
    README_END,
    check_consistency,
    load_artifacts,
    render_readme_results,
    render_report,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads((ROOT / "config/default.json").read_text())


def _history_row(
    event_id: str,
    version: str,
    *,
    generation: int,
    parent_event_id: str | None,
    parent_version: str | None,
    accepted: bool = True,
    decision: str = "accepted_improvement",
) -> dict:
    return {
        "event_id": event_id,
        "version": version,
        "generation": generation,
        "parent_version": parent_version,
        "parent_event_id": parent_event_id,
        "mutation": f"mutation:{event_id}",
        "accepted": accepted,
        "decision": decision,
        "train": {"accuracy": 0.5, "correct": 8, "total": 16},
        "dev": {"accuracy": 0.5, "correct": 4, "total": 8},
    }


class HarnessTests(unittest.TestCase):
    def test_policy_is_deterministic(self):
        policy = Policy(("safe",), ("risky",))
        self.assertEqual(policy.predict("A safe and clear plan"), 1)
        self.assertEqual(policy.predict("A risky plan"), 0)
        self.assertEqual(policy.version(), policy.version())

    def test_token_matching_does_not_cross_negation(self):
        """Whole-token matching: a keyword must not fire inside its negation."""
        self.assertEqual(tokenize("This plan is unsafe."), frozenset({"this", "plan", "is", "unsafe"}))
        safe = Policy(("safe",), ())
        self.assertEqual(safe.predict("The proposed step is unsafe."), 0)
        self.assertEqual(safe.predict("The answer is safe."), 1)
        clear = Policy(("clear",), ())
        self.assertEqual(clear.predict("Review says unclear."), 0)
        self.assertEqual(clear.predict("Review says clear."), 1)
        # A candidate may not place one word in both polarities.
        current = Candidate(Policy(("good",), ("bad",)), 0, None, "baseline")
        with self.assertRaises(ProposalError):
            validate_proposal(current, Policy(("good", "safe"), ("bad", "safe")))

    def test_splits_are_disjoint_and_complete(self):
        rows = make_dataset(42)
        self.assertEqual([e.task_id for e in rows], [e.task_id for e in make_dataset(42)])
        self.assertEqual(sum(e.split == "train" for e in rows), 16)
        self.assertEqual(sum(e.split == "dev" for e in rows), 8)
        self.assertEqual(sum(e.split == "heldout" for e in rows), 8)

    def test_all_seeds_resplit_the_same_pool(self):
        """Multi-seed runs are replays of one fixed pool; only the shuffle changes."""
        pools = [
            sorted((e.task_id, e.text, e.label) for e in make_dataset(seed))
            for seed in (42, 1337, 2026)
        ]
        self.assertEqual(pools[0], pools[1])
        self.assertEqual(pools[1], pools[2])

    def test_end_to_end_and_reproducible(self):
        config = _config()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run_experiment(config, Path(first))
            b = run_experiment(config, Path(second))
            self.assertEqual(a, b)
            self.assertTrue((Path(first) / "metrics.csv").exists())
            self.assertGreaterEqual(len(a["history"]), 2)
            self.assertEqual(a["splits"]["heldout"], 8)
            self.assertEqual(a["schema_version"], 2)
            self.assertEqual(a["experiment_version"], EXPERIMENT_VERSION)
            self.assertIn(a["outcome"], {"improvement", "flat", "regression"})
            # Accepted versions including the baseline and accepted new changes are
            # different numbers and must both be recorded.
            self.assertEqual(a["accepted_baseline_included_count"], len(a["accepted_versions"]))
            self.assertEqual(
                a["accepted_baseline_included_count"],
                a["accepted_new_change_count"] + 1,
            )

    def test_config_and_dataset_hashes_are_recorded(self):
        config = _config()
        result = run_experiment(config, output_dir=None, save_files=False)
        self.assertEqual(len(result["config_hash"]), 64)
        self.assertEqual(len(result["dataset_hash"]), 64)
        # The hash changes with the configuration and is stable otherwise.
        other = dict(config)
        other["generations"] = int(config["generations"]) + 1
        second = run_experiment(other, output_dir=None, save_files=False)
        self.assertNotEqual(result["config_hash"], second["config_hash"])

    def test_dev_regression_is_rejected_not_rolled_back(self):
        config = {
            "seed": 0,
            "generations": 1,
            "examples_per_pattern": 4,
            "initial_positive_keywords": [],
            "initial_negative_keywords": [],
            "initial_bias": 1,
            "mutation_pool": ["reliable"],
            "regression_tolerance": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_experiment(config, Path(directory))
        self.assertEqual(result["history"][1]["decision"], ROLLBACK_DECISION)
        # Rejection means the candidate was never applied: the active policy is the baseline.
        self.assertEqual(result["final"]["version"], result["baseline"]["version"])
        self.assertEqual(result["accepted_new_change_count"], 0)

    def test_heldout_data_does_not_affect_selection(self):
        """Scoped invariance check (not a proof): corrupting held-out data leaves
        the recorded selection chain, versions and dev scores unchanged."""
        config = _config()
        seed = int(config["seed"])
        dataset_a = make_dataset(seed)

        dataset_b = []
        for e in dataset_a:
            if e.split == "heldout":
                dataset_b.append(Example(e.task_id, "CORRUPTED_HELDOUT " + e.text, 1 - e.label, e.split))
            else:
                dataset_b.append(e)

        res_a = run_experiment(config, output_dir=None, save_files=False, examples_override=dataset_a)
        res_b = run_experiment(config, output_dir=None, save_files=False, examples_override=dataset_b)

        self.assertEqual(res_a["accepted_versions"], res_b["accepted_versions"])
        self.assertEqual(res_a["final"]["version"], res_b["final"]["version"])
        self.assertEqual(res_a["final"]["train"], res_b["final"]["train"])
        self.assertEqual(res_a["final"]["dev"], res_b["final"]["dev"])
        self.assertEqual(len(res_a["history"]), len(res_b["history"]))
        for h_a, h_b in zip(res_a["history"], res_b["history"]):
            self.assertEqual(h_a["event_id"], h_b["event_id"])
            self.assertEqual(h_a["version"], h_b["version"])
            self.assertEqual(h_a["accepted"], h_b["accepted"])
            self.assertEqual(h_a["decision"], h_b["decision"])
            self.assertEqual(h_a["dev"], h_b["dev"])
        self.assertNotEqual(res_a["final"]["heldout"], res_b["final"]["heldout"])

    def test_deterministic_and_runtime_artifacts_are_separated(self):
        """Byte equality holds only for runtime-free artifacts; files that embed
        runtime are numerically reproducible but not byte-identical."""
        config = _config()
        with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
            p1, p2 = Path(dir1), Path(dir2)
            for path in (p1, p2):
                run_experiment(config, path)
                run_multi_seed_experiment(config, seeds=[42, 1337], output_dir=path)

            for fname in ("first_run.json", "history.jsonl", "metrics.csv", "progress.svg"):
                b1 = (p1 / fname).read_bytes()
                b2 = (p2 / fname).read_bytes()
                self.assertEqual(b1, b2, f"Byte mismatch in {fname}")
                self.assertEqual(hashlib.sha256(b1).hexdigest(), hashlib.sha256(b2).hexdigest())

            def strip(obj):
                if isinstance(obj, dict):
                    return {k: strip(v) for k, v in obj.items() if k != "runtime_seconds"}
                if isinstance(obj, list):
                    return [strip(v) for v in obj]
                return obj

            a = strip(json.loads((p1 / "multi_seed_results.json").read_text()))
            b = strip(json.loads((p2 / "multi_seed_results.json").read_text()))
            self.assertEqual(a, b)
            # Sanity: the raw files are NOT byte-identical, so the scoped claim is real.
            self.assertNotEqual(
                (p1 / "multi_seed_results.json").read_bytes(),
                (p2 / "multi_seed_results.json").read_bytes(),
            )

    def test_multi_seed_execution_and_statistics(self):
        config = _config()
        seeds = [42, 1337, 2026]
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_multi_seed_run(config, seeds, Path(tmpdir))

            self.assertEqual(summary["num_runs"], 3)
            self.assertEqual(summary["seeds"], seeds)
            self.assertIn("mean", summary["heldout_accuracy"])
            self.assertIn("std", summary["heldout_accuracy"])
            self.assertEqual(summary["heldout_accuracy"]["n"], 3)
            self.assertEqual(summary["heldout_accuracy"]["stdev_ddof"], 1)
            self.assertEqual(summary["heldout_examples_per_seed"], 8)
            self.assertTrue(summary["same_synthetic_pool_across_seeds"])
            self.assertIn("mean", summary["heldout_gain"])
            self.assertTrue((Path(tmpdir) / "multi_seed_results.json").exists())
            self.assertTrue((Path(tmpdir) / "multi_seed_metrics.csv").exists())
            self.assertEqual(len(summary["runs"]), 3)
            for r in summary["runs"]:
                self.assertIn("artifact_hash", r)
                self.assertIn("runtime_seconds", r)
                self.assertEqual(
                    r["accepted_baseline_included_count"],
                    r["accepted_new_change_count"] + 1,
                )

    def test_ablation_experiments(self):
        config = _config()
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_ablation_experiments(config, output_dir=Path(tmpdir))
            modes = {r["mode"]: r for r in summary["runs"]}

            self.assertIn("baseline_full", modes)
            self.assertIn("ablation_no_mutation", modes)
            self.assertIn("ablation_no_selection", modes)
            self.assertIn("ablation_no_rollback", modes)
            self.assertIn("ablation_random_selection", modes)

            # Ablations run with the same seed, data and budget so they are comparable.
            self.assertTrue(summary["comparable_seed_data_budget"])
            hashes = {r["dataset_hash"] for r in summary["runs"]}
            self.assertEqual(len(hashes), 1)
            seeds = {r["seed"] for r in summary["runs"]}
            self.assertEqual(len(seeds), 1)

            # Each mode documents the mechanism it switches off.
            for r in summary["runs"]:
                self.assertTrue(r["mechanism"])

            # Mutation disabled: no new change can ever be accepted.
            self.assertEqual(modes["ablation_no_mutation"]["accepted_new_change_count"], 0)
            self.assertEqual(modes["ablation_no_mutation"]["heldout_gain"], 0.0)

            # The score-free random control accepts one candidate per generation.
            self.assertEqual(
                modes["ablation_random_selection"]["accepted_new_change_count"],
                int(config["generations"]),
            )

            self.assertTrue((Path(tmpdir) / "ablation_results.json").exists())
            self.assertTrue((Path(tmpdir) / "ablation_metrics.csv").exists())

    def test_candidate_lineage_and_tree(self):
        config = _config()
        result = run_experiment(config, output_dir=None, save_files=False)
        tree = build_lineage_tree(result["history"])
        self.assertGreaterEqual(tree["total_nodes"], 2)
        self.assertEqual(tree["total_nodes"], len(result["history"]))
        self.assertEqual(tree["cyclic_event_ids"], [])
        chain = extract_lineage_chain(result["final"]["version"], result["history"])
        self.assertGreaterEqual(len(chain), 1)
        self.assertEqual(chain[0], result["baseline"]["version"])
        self.assertEqual(chain[-1], result["final"]["version"])

    def test_recorded_parent_matches_the_active_policy(self):
        """A rejected candidate must never become a recorded parent: every
        parent_event_id must resolve to an earlier event with that parent_version."""
        config = _config()
        result = run_experiment(config, output_dir=None, save_files=False)
        by_event = {row["event_id"]: row for row in result["history"]}
        self.assertTrue(any(not row["accepted"] for row in result["history"]))
        for index, row in enumerate(result["history"]):
            parent = row["parent_event_id"]
            if parent is None:
                self.assertEqual(row["generation"], 0)
                continue
            self.assertIn(parent, by_event)
            self.assertLess(
                [r["event_id"] for r in result["history"]].index(parent),
                index,
                "parent event must appear before its child",
            )
            self.assertEqual(by_event[parent]["version"], row["parent_version"])
            self.assertTrue(by_event[parent]["accepted"])
        self.assertEqual(lineage_integrity(result["history"])["orphan_event_ids"], [])

    def test_repeated_policy_version_does_not_overwrite_events(self):
        """The same policy hash may recur in later generations; each occurrence is a
        separate event and no lineage record may be overwritten."""
        version = "candidate-aaaa"
        history = [
            _history_row("g0", version, generation=0, parent_event_id=None, parent_version=None, decision="baseline"),
            _history_row("g1", "candidate-bbbb", generation=1, parent_event_id="g0", parent_version=version),
            _history_row("g2", version, generation=2, parent_event_id="g1", parent_version="candidate-bbbb"),
            _history_row("g3", version, generation=3, parent_event_id="g2", parent_version=version),
        ]
        tree = build_lineage_tree(history)
        self.assertEqual(tree["total_nodes"], 4)
        self.assertEqual(tree["duplicate_version_count"], 1)
        self.assertEqual(tree["cyclic_event_ids"], [])
        # Both occurrences of the repeated version are preserved as distinct nodes.
        g2 = next(node for node in _flatten(tree["roots"]) if node["event_id"] == "g2")
        self.assertEqual(g2["version"], version)
        self.assertEqual(len(g2["children"]), 1)
        self.assertEqual(g2["children"][0]["event_id"], "g3")
        self.assertEqual(g2["parent_event_id"], "g1")
        self.assertNotEqual(g2["parent_event_id"], "g0")

    def test_branching_history(self):
        """One parent with several children must not collapse to a single child."""
        history = [
            _history_row("g0", "v0", generation=0, parent_event_id=None, parent_version=None, decision="baseline"),
            _history_row("g1-a", "v1a", generation=1, parent_event_id="g0", parent_version="v0"),
            _history_row("g1-b", "v1b", generation=1, parent_event_id="g0", parent_version="v0"),
            _history_row("g1-c", "v1c", generation=1, parent_event_id="g0", parent_version="v0"),
        ]
        tree = build_lineage_tree(history)
        root = tree["roots"][0]
        self.assertEqual(root["event_id"], "g0")
        self.assertEqual({child["event_id"] for child in root["children"]}, {"g1-a", "g1-b", "g1-c"})

    def test_orphan_parent_is_reported_not_dropped(self):
        history = [
            _history_row("g0", "v0", generation=0, parent_event_id=None, parent_version=None, decision="baseline"),
            _history_row("g1", "v1", generation=1, parent_event_id="missing", parent_version="v-missing"),
        ]
        tree = build_lineage_tree(history)
        self.assertEqual(tree["total_nodes"], 2)
        self.assertEqual([node["event_id"] for node in tree["orphans"]], ["g1"])
        integrity = lineage_integrity(history)
        self.assertEqual(integrity["orphan_event_ids"], ["g1"])

    def test_cycle_in_lineage_terminates(self):
        history = [
            _history_row("g1", "v1", generation=1, parent_event_id="g2", parent_version="v2"),
            _history_row("g2", "v2", generation=2, parent_event_id="g1", parent_version="v1"),
        ]
        integrity = lineage_integrity(history)
        self.assertTrue(integrity["cyclic_event_ids"])
        # Must terminate rather than loop forever.
        chain = extract_lineage_events("g1", history)
        self.assertLessEqual(len(chain), len(history))
        events = extract_lineage_events("g4", history)
        self.assertEqual(events, ["g4"])

    def test_proposal_validation_rejects_invalid_mutations(self):
        current = Candidate(Policy(("good",), ("bad",)), 0, None, "baseline")
        vocabulary = ["good", "bad", "reliable", "clear", "safe"]

        def is_valid(policy):
            try:
                validate_proposal(current, policy, allowed_vocabulary=vocabulary, max_edits=1, bias_bound=2)
                return True
            except ProposalError:
                return False

        self.assertTrue(is_valid(Policy(("good", "reliable"), ("bad",))))
        self.assertTrue(is_valid(Policy(("good",), ("bad",), 1)))
        self.assertFalse(is_valid(Policy(("good",), ("bad",))))          # no-op
        self.assertFalse(is_valid(Policy(("good", "reliable", "clear"), ("bad",))))  # 2 edits
        self.assertFalse(is_valid(Policy(("good", "unlisted"), ("bad",))))           # outside vocabulary
        self.assertFalse(is_valid(Policy(("good",), ("bad", "unlisted"))))           # outside vocabulary
        self.assertFalse(is_valid(Policy(("good",), ("bad",), 99)))                  # bias bound
        self.assertFalse(is_valid(Policy(("good",), ("bad", "bad "))))               # illegal token

    def test_provider_injection_and_json_file_proposals(self):
        """The loop consumes an injected provider and rejects malformed proposals."""
        config = {
            "seed": 7,
            "generations": 1,
            "examples_per_pattern": 4,
            "initial_positive_keywords": ["good"],
            "initial_negative_keywords": ["bad"],
            "initial_bias": 0,
            "mutation_pool": ["reliable"],
            "regression_tolerance": 0.0,
            "provider_keyword_vocabulary": ["good", "bad", "reliable"],
        }
        proposals = {
            "candidates": [
                {
                    "mutation": "add_positive:reliable",
                    "policy": {"positive_keywords": ["good", "reliable"], "negative_keywords": ["bad"], "bias": 0},
                },
                {
                    "mutation": "broken_two_edits",
                    "policy": {"positive_keywords": ["good", "reliable"], "negative_keywords": ["bad", "reliable"], "bias": 0},
                },
                {
                    "mutation": "out_of_vocabulary",
                    "policy": {"positive_keywords": ["good", "unlisted"], "negative_keywords": ["bad"], "bias": 0},
                },
                {
                    "mutation": "noop",
                    "policy": {"positive_keywords": ["good"], "negative_keywords": ["bad"], "bias": 0},
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proposals.json"
            path.write_text(json.dumps(proposals))
            provider = JsonFileProposalProvider(path)
            result = run_experiment(config, output_dir=None, save_files=False, candidate_generator=provider)

        self.assertEqual(result["rejected_proposal_count"], 3)
        reasons = " ".join(p["reason"] for p in result["rejected_proposals"])
        self.assertIn("both polarities", reasons)
        self.assertIn("outside the allowed vocabulary", reasons)
        self.assertIn("no-op", reasons)
        # The one valid proposal was scored and selected.
        self.assertEqual(result["final"]["version"], Policy(("good", "reliable"), ("bad",), 0).version())

    def test_injected_provider_replaces_default_generator(self):
        class StubGenerator:
            generator_type = "stub"

            def generate(self, current, generation, seed):
                return [
                    Candidate(Policy(("good", "reliable"), ("bad",)), generation, current.version, "stub:mutation"),
                ]

        config = {
            "seed": 3,
            "generations": 1,
            "examples_per_pattern": 4,
            "initial_positive_keywords": ["good"],
            "initial_negative_keywords": ["bad"],
            "initial_bias": 0,
            "mutation_pool": ["clear"],
        }
        result = run_experiment(config, output_dir=None, save_files=False, candidate_generator=StubGenerator())
        self.assertIn("stub:mutation", [row["mutation"] for row in result["history"]])

    def test_open_weight_stub_boundary(self):
        """The stub is a boundary marker: it raises and performs no inference."""
        stub = OpenWeightColabL4Stub(model_name="meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(stub.model_name, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertTrue(stub.is_deterministic)
        with self.assertRaises(NotImplementedError) as ctx:
            stub.propose_keywords((), (), "task")
        self.assertIn("Colab L4", str(ctx.exception))
        self.assertIn("no inference", str(ctx.exception))

    def test_report_numbers_match_artifacts(self):
        """Report/README numbers are generated from results/*.json and must not drift."""
        results_dir = ROOT / "results"
        artifacts = load_artifacts(results_dir)
        self.assertIn("multi_seed", artifacts)
        self.assertIn("ablation", artifacts)

        expected_report = render_report(artifacts).rstrip() + "\n"
        self.assertEqual((results_dir / "report.md").read_text(), expected_report)

        readme = (ROOT / "README.md").read_text()
        self.assertIn(README_BEGIN, readme)
        block = readme.split(README_BEGIN, 1)[1].split(README_END, 1)[0].strip()
        self.assertEqual(block, render_readme_results(artifacts).strip())

        # Independent check that the artifact values actually appear in the output.
        multi = artifacts["multi_seed"]
        for key in ("train_accuracy", "dev_accuracy", "heldout_accuracy", "heldout_gain"):
            self.assertIn(f"{multi[key]['mean']:.3f}", expected_report)
        for key in ("accepted_new_changes", "accepted_baseline_included_count", "regression_count"):
            self.assertIn(f"{multi[key]['mean']:.2f}", expected_report)
        first = artifacts["first_run"]
        self.assertIn(str(first["accepted_new_change_count"]), expected_report)

        self.assertEqual(check_consistency(results_dir, ROOT / "README.md"), [])

    def test_report_renderer_rejects_stale_readme_numbers(self):
        """The renderer must not reproduce the stale v1 numbers that were wrong."""
        report = render_report(load_artifacts(ROOT / "results"))
        for stale in ("0.812 ± 0.054", "0.625 ± 0.108", "1.00 ± 0.00", "0.708 ± 0.072"):
            self.assertNotIn(stale, report)
        self.assertIn("Accepted candidate versions **including the baseline**", report)
        self.assertIn("Accepted **new changes**", report)


def run_multi_seed_run(config, seeds, output_dir):
    return run_multi_seed_experiment(config, seeds=seeds, output_dir=output_dir)


def _flatten(nodes):
    for node in nodes:
        yield node
        yield from _flatten(node["children"])


if __name__ == "__main__":
    unittest.main()
