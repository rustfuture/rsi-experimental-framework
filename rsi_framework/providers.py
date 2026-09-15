"""Provider interface and candidate generation abstractions.

This module defines the boundaries for policy candidate generation.
The primary baseline is deterministic mutation (no LLM, no network, no API cost).
An abstract interface and stub are provided for future local or open-weight models
(e.g. running on Colab L4 or local GPU via vLLM / HuggingFace Transformers).
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Iterable, Protocol

if TYPE_CHECKING:
    from .core import Candidate, Policy


class CandidateGenerator(Protocol):
    """Protocol implemented by candidate generators."""

    def generate(self, current: Candidate, generation: int, seed: int) -> list[Candidate]:
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
    def generate(self, current: Candidate, generation: int, seed: int) -> list[Candidate]:
        """Generate candidate mutations from the current candidate."""
        ...


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

    Validates that no network or unconfigured local GPU calls occur silently.
    When instantiated in test or offline environments, it raises informative errors
    explaining how to mount a local open-weight model without paid API dependencies.
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
            f"weights loaded for '{self._model_name}'. In this milestone, use DeterministicMutationGenerator "
            f"(method_label: HARNESS_BASELINE_NOT_LLM). No paid APIs are used."
        )
