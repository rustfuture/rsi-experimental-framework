import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rsi_framework.core import (
    Candidate,
    Example,
    Policy,
    build_lineage_tree,
    extract_lineage_chain,
    make_dataset,
    run_ablation_experiments,
    run_experiment,
    run_multi_seed_experiment,
)
from rsi_framework.providers import OpenWeightColabL4Stub


class HarnessTests(unittest.TestCase):
    def test_policy_is_deterministic(self):
        policy = Policy(("safe",), ("risky",))
        self.assertEqual(policy.predict("A safe and clear plan"), 1)
        self.assertEqual(policy.predict("A risky plan"), 0)
        self.assertEqual(policy.version(), policy.version())

    def test_splits_are_disjoint_and_complete(self):
        rows = make_dataset(42)
        self.assertEqual([e.task_id for e in rows], [e.task_id for e in make_dataset(42)])
        self.assertEqual(sum(e.split == "train" for e in rows), 16)
        self.assertEqual(sum(e.split == "dev" for e in rows), 8)
        self.assertEqual(sum(e.split == "heldout" for e in rows), 8)

    def test_end_to_end_and_reproducible(self):
        config = json.loads(Path("config/default.json").read_text())
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run_experiment(config, Path(first))
            b = run_experiment(config, Path(second))
            self.assertEqual(a, b)
            self.assertTrue((Path(first) / "metrics.csv").exists())
            self.assertGreaterEqual(len(a["history"]), 2)
            self.assertEqual(a["splits"]["heldout"], 8)
            self.assertIn(a["outcome"], {"improvement", "flat", "regression"})

    def test_dev_regression_is_recorded_and_rolled_back(self):
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
        self.assertEqual(result["history"][1]["decision"], "rollback_regression")
        self.assertEqual(result["final"]["version"], result["baseline"]["version"])

    def test_heldout_data_does_not_affect_selection(self):
        """Proof of zero held-out leakage: candidate search history is bit-for-bit invariant to held-out data."""
        config = json.loads(Path("config/default.json").read_text())
        seed = int(config["seed"])
        dataset_a = make_dataset(seed)

        # Corrupt and invert every heldout label and sentence in dataset_b
        dataset_b = []
        for e in dataset_a:
            if e.split == "heldout":
                # Invert label and replace text
                dataset_b.append(Example(e.task_id, "CORRUPTED_HELDOUT " + e.text, 1 - e.label, e.split))
            else:
                dataset_b.append(e)

        res_a = run_experiment(config, output_dir=None, save_files=False, examples_override=dataset_a)
        res_b = run_experiment(config, output_dir=None, save_files=False, examples_override=dataset_b)

        # Candidate selection sequence, versions, decisions, and dev scores must be 100% identical
        self.assertEqual(res_a["accepted_versions"], res_b["accepted_versions"])
        self.assertEqual(res_a["final"]["version"], res_b["final"]["version"])
        self.assertEqual(res_a["final"]["train"], res_b["final"]["train"])
        self.assertEqual(res_a["final"]["dev"], res_b["final"]["dev"])
        self.assertEqual(len(res_a["history"]), len(res_b["history"]))
        for h_a, h_b in zip(res_a["history"], res_b["history"]):
            self.assertEqual(h_a["version"], h_b["version"])
            self.assertEqual(h_a["accepted"], h_b["accepted"])
            self.assertEqual(h_a["decision"], h_b["decision"])
            self.assertEqual(h_a["dev"], h_b["dev"])
        # Held-out evaluations should reflect the difference, proving it was evaluated independently post-selection
        self.assertNotEqual(res_a["final"]["heldout"], res_b["final"]["heldout"])

    def test_byte_level_reproducibility(self):
        """Identical config and seed must produce byte-level identical JSON and CSV artifacts."""
        config = json.loads(Path("config/default.json").read_text())
        with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
            p1, p2 = Path(dir1), Path(dir2)
            run_experiment(config, p1)
            run_experiment(config, p2)

            # Check exact byte equality on first_run.json, history.jsonl, and metrics.csv
            for fname in ("first_run.json", "history.jsonl", "metrics.csv"):
                b1 = (p1 / fname).read_bytes()
                b2 = (p2 / fname).read_bytes()
                self.assertEqual(b1, b2, f"Byte mismatch in {fname}")
                hash1 = hashlib.sha256(b1).hexdigest()
                hash2 = hashlib.sha256(b2).hexdigest()
                self.assertEqual(hash1, hash2, f"SHA-256 mismatch in {fname}")

    def test_multi_seed_execution_and_statistics(self):
        """Multi-seed harness executes >= 3 seeds and calculates statistical aggregate metrics."""
        config = json.loads(Path("config/default.json").read_text())
        seeds = [42, 1337, 2026]
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_multi_seed_experiment(config, seeds=seeds, output_dir=Path(tmpdir))

            self.assertEqual(summary["num_runs"], 3)
            self.assertEqual(summary["seeds"], seeds)
            self.assertIn("mean", summary["heldout_accuracy"])
            self.assertIn("std", summary["heldout_accuracy"])
            self.assertIn("mean", summary["heldout_gain"])
            self.assertTrue((Path(tmpdir) / "multi_seed_results.json").exists())
            self.assertTrue((Path(tmpdir) / "multi_seed_metrics.csv").exists())
            self.assertEqual(len(summary["runs"]), 3)
            for r in summary["runs"]:
                self.assertIn("artifact_hash", r)
                self.assertIn("runtime_seconds", r)

    def test_ablation_experiments(self):
        """Ablation harness correctly disables mutation, selection, and rollback."""
        config = json.loads(Path("config/default.json").read_text())
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_ablation_experiments(config, output_dir=Path(tmpdir))
            modes = {r["mode"]: r for r in summary["runs"]}

            self.assertIn("baseline_full", modes)
            self.assertIn("ablation_no_mutation", modes)
            self.assertIn("ablation_no_selection", modes)
            self.assertIn("ablation_no_rollback", modes)

            # When mutation is disabled, no new candidates can be accepted
            self.assertEqual(modes["ablation_no_mutation"]["accepted_count"], 1)  # baseline only
            self.assertEqual(modes["ablation_no_mutation"]["heldout_gain"], 0.0)

            # Metrics files should exist
            self.assertTrue((Path(tmpdir) / "ablation_results.json").exists())
            self.assertTrue((Path(tmpdir) / "ablation_metrics.csv").exists())

    def test_candidate_lineage_and_tree(self):
        """Candidate versions form a traceable lineage tree with parent links."""
        config = json.loads(Path("config/default.json").read_text())
        result = run_experiment(config, output_dir=None, save_files=False)
        tree = build_lineage_tree(result["history"])
        self.assertGreaterEqual(tree["total_nodes"], 2)
        chain = extract_lineage_chain(result["final"]["version"], result["history"])
        self.assertGreaterEqual(len(chain), 1)
        self.assertEqual(chain[0], result["baseline"]["version"])
        self.assertEqual(chain[-1], result["final"]["version"])

    def test_open_weight_stub_boundary(self):
        """Stub raises informative error documenting open-weight Colab L4 requirements."""
        stub = OpenWeightColabL4Stub(model_name="meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(stub.model_name, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertTrue(stub.is_deterministic)
        with self.assertRaises(NotImplementedError) as ctx:
            stub.propose_keywords((), (), "task")
        self.assertIn("Colab L4", str(ctx.exception))
        self.assertIn("HARNESS_BASELINE_NOT_LLM", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
