import json
import tempfile
import unittest
from pathlib import Path

from rsi_framework.core import Policy, make_dataset, run_experiment


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
            "seed": 0, "generations": 1, "examples_per_pattern": 4,
            "initial_positive_keywords": [], "initial_negative_keywords": [],
            "initial_bias": 1, "mutation_pool": ["reliable"],
            "regression_tolerance": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_experiment(config, Path(directory))
        self.assertEqual(result["history"][1]["decision"], "rollback_regression")
        self.assertEqual(result["final"]["version"], result["baseline"]["version"])


if __name__ == "__main__":
    unittest.main()
