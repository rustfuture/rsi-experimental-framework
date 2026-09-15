import json
from pathlib import Path
import tempfile
import unittest

from rsi_framework.core import Candidate, Policy, run_experiment
from rsi_framework.providers import JsonFileProposalProvider, ProposalError, validate_proposal


class ProviderBoundaryTests(unittest.TestCase):
    def test_malformed_entries_are_recorded_without_losing_valid_candidates(self):
        config = json.loads(Path('config/default.json').read_text())
        config['generations'] = 1
        config['initial_positive_keywords'] = []
        config['initial_negative_keywords'] = []
        config['initial_bias'] = 0
        word = config['mutation_pool'][0]
        entries = [None, {'policy': {'bias': 'oops'}},
                   {'policy': {'positive_keywords': None}},
                   {'policy': {'positive_keywords': [word], 'bias': 0}}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'proposals.json'
            path.write_text(json.dumps(entries))
            result = run_experiment(config, candidate_generator=JsonFileProposalProvider(path))
        self.assertEqual(len(result['rejected_proposals']), 3)
        self.assertEqual(len(result['history']), 2)

    def test_duplicates_and_non_integer_bias_are_rejected(self):
        current = Candidate(Policy(('safe',), (), 0), 0, None, 'baseline')
        for policy in (Policy(('safe', 'safe', 'clear'), (), 0),
                       Policy(('safe', 'clear'), (), 0.5),
                       Policy(('safe', 'clear'), (), True)):
            with self.subTest(policy=policy), self.assertRaises(ProposalError):
                validate_proposal(current, policy)
