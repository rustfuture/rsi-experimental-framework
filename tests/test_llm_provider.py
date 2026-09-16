import unittest
from rsi_framework.providers import LLMMutationGenerator, ProposalError, resolve_device
from rsi_framework.core import Candidate, Policy

class MockProvider:
    def __init__(self, proposals):
        self.proposals = proposals
        self.model_name = "mock"
        self.is_deterministic = True

    def propose_keywords(self, cp, cn, tp, n_proposals=4, seed=42):
        if isinstance(self.proposals, Exception):
            raise self.proposals
        return self.proposals

class LLMMutationGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.current = Candidate(Policy(("good",), ("bad",), 0), 0, None, "baseline")

    def test_valid_proposals(self):
        provider = MockProvider([("add_positive", "verifiable"), ("bias_up", "up")])
        gen = LLMMutationGenerator(provider, "test")
        candidates = gen.generate(self.current, 1, 42)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].policy.positive_keywords, ("good", "verifiable"))
        self.assertEqual(candidates[1].policy.bias, 1)

    def test_invalid_action(self):
        provider = MockProvider([("invalid_action", "word")])
        gen = LLMMutationGenerator(provider, "test")
        candidates = gen.generate(self.current, 1, 42)
        self.assertEqual(len(candidates), 0)
        self.assertEqual(len(gen.rejections), 1)
        self.assertIn("invalid_action", gen.rejections[0]["reason"])

    def test_provider_failure(self):
        provider = MockProvider(RuntimeError("API timeout"))
        gen = LLMMutationGenerator(provider, "test")
        candidates = gen.generate(self.current, 1, 42)
        self.assertEqual(len(candidates), 0)
        self.assertEqual(len(gen.rejections), 1)
        self.assertIn("API timeout", gen.rejections[0]["reason"])

    def test_blocked_by_runtime(self):
        provider = MockProvider(RuntimeError("EXPERIMENT_BLOCKED_BY_RUNTIME: torch missing"))
        gen = LLMMutationGenerator(provider, "test")
        with self.assertRaises(RuntimeError) as cm:
            gen.generate(self.current, 1, 42)
        self.assertIn("EXPERIMENT_BLOCKED_BY_RUNTIME", str(cm.exception))

class DeviceResolutionTests(unittest.TestCase):
    """`--device` is explicit: an unavailable request must fail, never fall back."""

    def test_cpu_is_always_accepted(self):
        self.assertEqual(resolve_device("cpu"), "cpu")

    def test_auto_selects_a_real_device(self):
        self.assertIn(resolve_device("auto"), {"cpu", "cuda", "mps"})

    def test_unknown_device_fails_clearly(self):
        with self.assertRaises(RuntimeError) as cm:
            resolve_device("not-a-real-device")
        self.assertIn("EXPERIMENT_BLOCKED_BY_RUNTIME", str(cm.exception))
        self.assertIn("not-a-real-device", str(cm.exception))

    def test_blank_device_fails_clearly(self):
        with self.assertRaises(RuntimeError) as cm:
            resolve_device("   ")
        self.assertIn("EXPERIMENT_BLOCKED_BY_RUNTIME", str(cm.exception))

    def test_unavailable_explicit_device_fails_without_fallback(self):
        try:
            import torch
        except ImportError:
            with self.assertRaises(RuntimeError) as cm:
                resolve_device("cuda")
            self.assertIn("EXPERIMENT_BLOCKED_BY_RUNTIME", str(cm.exception))
            return
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError) as cm:
                resolve_device("cuda")
            self.assertIn("EXPERIMENT_BLOCKED_BY_RUNTIME", str(cm.exception))
        else:
            self.assertEqual(resolve_device("cuda"), "cuda")


if __name__ == "__main__":
    unittest.main()
