"""Provider interface, proposal validation, and candidate generation abstractions.

This module defines the boundaries for policy candidate generation.
The primary baseline is deterministic mutation (no LLM, no network, no API cost).
An abstract interface and stub are provided for future local or open-weight models
(e.g. running on Colab L4 or local GPU via vLLM / HuggingFace Transformers).

The harness is *injectable*: :func:`rsi_framework.core.run_experiment` accepts any
object matching :class:`CandidateGenerator`. Proposals coming from an external or
non-deterministic source are validated by :func:`validate_proposal` before they can
be scored, so a broken or out-of-policy model output is rejected instead of
silently changing the search.

Nothing here is an LLM. `JsonFileProposalProvider` merely reads candidate
proposals that some other process (for example a local model runner) wrote to a
file; it performs no inference and no network access.
"""

from __future__ import annotations

import abc
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Protocol

if TYPE_CHECKING:
    from .core import Candidate, Policy

#: A keyword must be a single lowercase token. This blocks an injected provider
#: from smuggling whitespace, punctuation, or free text into the matcher.
_KEYWORD_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ProposalError(ValueError):
    """Raised when a candidate proposal is malformed or outside policy."""


def validate_proposal(
    current: "Candidate",
    proposed: "Policy",
    *,
    allowed_vocabulary: Iterable[str] | None = None,
    max_edits: int = 1,
    bias_bound: int = 5,
) -> None:
    """Validate a proposed policy against the current policy.

    Raises :class:`ProposalError` when the proposal is a no-op, changes more than
    ``max_edits`` elements, uses keywords outside the allowed vocabulary or the
    keyword syntax, puts one keyword in both polarities, or moves the bias outside
    ``[-bias_bound, bias_bound]``.
    """
    positive = tuple(proposed.positive_keywords)
    negative = tuple(proposed.negative_keywords)

    if type(proposed.bias) is not int:
        raise ProposalError("bias must be an integer")

    for word in positive + negative:
        if not isinstance(word, str) or not _KEYWORD_RE.match(word):
            raise ProposalError(f"illegal keyword {word!r}: expected a single lowercase token")

    if len(set(positive)) != len(positive) or len(set(negative)) != len(negative):
        raise ProposalError("duplicate keywords are not allowed")

    overlap = set(positive) & set(negative)
    if overlap:
        raise ProposalError(f"keyword(s) {sorted(overlap)} appear in both polarities")

    if allowed_vocabulary is not None:
        allowed = set(allowed_vocabulary)
        unknown = (set(positive) | set(negative)) - allowed
        if unknown:
            raise ProposalError(f"keyword(s) {sorted(unknown)} are outside the allowed vocabulary")

    if abs(int(proposed.bias)) > int(bias_bound):
        raise ProposalError(f"bias {proposed.bias} exceeds bound +/-{bias_bound}")

    edits = (
        len(set(positive) ^ set(current.policy.positive_keywords))
        + len(set(negative) ^ set(current.policy.negative_keywords))
        + (1 if int(proposed.bias) != int(current.policy.bias) else 0)
    )
    if edits == 0:
        raise ProposalError("proposal is identical to the current policy (no-op)")
    if edits > max_edits:
        raise ProposalError(f"proposal changes {edits} elements; the budget is {max_edits}")


class CandidateGenerator(Protocol):
    """Protocol implemented by candidate generators."""

    def generate(self, current: "Candidate", generation: int, seed: int) -> list["Candidate"]:
        """Generate candidate mutations from the current policy."""
        ...


class BaseCandidateGenerator(abc.ABC):
    """Abstract base class for candidate generators."""

    @property
    @abc.abstractmethod
    def generator_type(self) -> str:
        """Name/type of the generator."""
        ...

    @abc.abstractmethod
    def generate(self, current: "Candidate", generation: int, seed: int) -> list["Candidate"]:
        """Generate candidate mutations from the current candidate."""
        ...


class JsonFileProposalProvider:
    """Read externally produced candidate proposals from a JSON file.

    This is the concrete injection path for a local/open-weight model without
    giving the harness any network dependency: an external runner writes

    .. code-block:: json

        {"candidates": [
            {"mutation": "add_positive:verifiable",
             "policy": {"positive_keywords": ["good", "verifiable"],
                        "negative_keywords": ["bad"], "bias": 0}}
        ]}

    and the harness validates every entry via :func:`validate_proposal`. A
    top-level JSON list is also accepted. Malformed entries are recorded in
    ``rejections`` for the harness; valid siblings continue through validation.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._cache: list[dict] | None = None
        self.rejections: list[dict] = []

    @property
    def generator_type(self) -> str:
        return "json_file_proposals"

    def _load(self) -> list[dict]:
        if self._cache is None:
            payload = json.loads(self.path.read_text())
            if isinstance(payload, dict):
                entries = payload.get("candidates", [])
                if not isinstance(entries, list):
                    raise ProposalError("candidates must be a list")
                self._cache = entries
            elif isinstance(payload, list):
                self._cache = list(payload)
            else:
                raise ProposalError(f"{self.path}: expected an object or a list")
        return self._cache

    def generate(self, current: "Candidate", generation: int, seed: int) -> list["Candidate"]:
        from .core import Candidate, Policy  # local import avoids an import cycle

        proposals: list[Candidate] = []
        self.rejections = []
        for index, entry in enumerate(self._load()):
            try:
                if not isinstance(entry, dict) or not isinstance(entry.get("policy"), dict):
                    raise ProposalError("candidate and policy must be objects")
                payload = entry["policy"]
                for key in ("positive_keywords", "negative_keywords"):
                    words = payload.get(key, [])
                    if not isinstance(words, list) or any(not isinstance(w, str) for w in words):
                        raise ProposalError(f"{key} must be a list of strings")
                if type(payload.get("bias", 0)) is not int:
                    raise ProposalError("bias must be an integer")
                if not isinstance(entry.get("mutation", "external"), str):
                    raise ProposalError("mutation must be a string")
            except ProposalError as exc:
                self.rejections.append({"event_id": f"g{generation}-ext{index}",
                                        "generation": generation, "reason": str(exc)})
                continue
            policy_payload = entry.get("policy", {})
            policy = Policy(
                tuple(policy_payload.get("positive_keywords", ())),
                tuple(policy_payload.get("negative_keywords", ())),
                int(policy_payload.get("bias", 0)),
            )
            proposals.append(
                Candidate(
                    policy,
                    generation,
                    current.version,
                    entry.get("mutation", f"external:{index}"),
                    event_id=f"g{generation}-ext{index}",
                )
            )
        return proposals


class OpenWeightLLMProvider(abc.ABC):
    """Abstract interface for local / open-weight LLM providers.

    Intended for execution on local hardware or Google Colab L4 instances
    running inference engines like vLLM, Ollama, or Hugging Face Transformers.
    STRICTLY NO PAID APIs, NO CLOUD BILLING, AND NO SECRET KEYS REQUIRED.
    """

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Name of the open-weight model (e.g. 'Qwen/Qwen2.5-Coder-7B-Instruct')."""
        ...

    @property
    @abc.abstractmethod
    def is_deterministic(self) -> bool:
        """Whether greedy decoding (temperature=0.0) is enforced."""
        ...

    @abc.abstractmethod
    def propose_keywords(
        self,
        current_positive: tuple[str, ...],
        current_negative: tuple[str, ...],
        task_prompt: str,
        n_proposals: int = 4,
        seed: int = 42,
    ) -> list[tuple[str, str]]:
        """Propose keyword changes: list of (action, keyword), e.g. ('add_positive', 'verifiable')."""
        ...


class OpenWeightColabL4Stub(OpenWeightLLMProvider):
    """Colab L4 / local open-weight adapter stub.

    This class is a boundary marker, NOT a working model integration. It exists to
    make an accidental silent GPU/network call impossible: any attempt to use it
    raises an informative error. No LLM inference has been implemented or run.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cuda:0"):
        self._model_name = model_name
        self._device = device

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_deterministic(self) -> bool:
        return True

    def propose_keywords(
        self,
        current_positive: tuple[str, ...],
        current_negative: tuple[str, ...],
        task_prompt: str,
        n_proposals: int = 4,
        seed: int = 42,
    ) -> list[tuple[str, str]]:
        raise NotImplementedError(
            f"OpenWeightColabL4Stub requires an active GPU environment (e.g. Colab L4) with local "
            f"weights loaded for '{self._model_name}'. This stub performs no inference. In this milestone, "
            f"use DeterministicMutationGenerator (method_label: HARNESS_BASELINE_NOT_LLM) or supply "
            f"proposals through JsonFileProposalProvider. No paid APIs are used."
        )
