"""
LLM configuration and reproducibility for infomux.

This module provides structures for:
- Recording model names, versions, and parameters
- Generating and recording random seeds
- Ensuring reproducible LLM outputs

Design principles:
- All randomness must be explicit and recorded
- Model versions must be captured for reproducibility
- Parameters must be serializable for job envelope storage
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ModelInfo:
    """
    Information about a model used in a step.

    Attributes:
        name: Model identifier (e.g., "qwen2.5:14b-instruct", "ggml-base.en")
        provider: Model provider (e.g., "ollama", "whisper.cpp")
        version: Model version if known
        path: Local path to model file if applicable
    """

    name: str
    provider: str
    version: str | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelInfo:
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class GenerationParams:
    """
    Parameters for LLM text generation.

    These parameters affect output and must be recorded for reproducibility.

    Attributes:
        seed: Random seed for sampling. If None, one will be generated.
        temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        top_p: Nucleus sampling probability threshold
        top_k: Top-k sampling limit
        max_tokens: Maximum tokens to generate
        stop: Stop sequences
        system_prompt: System prompt used (for auditing)
    """

    seed: int | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    stop: list[str] = field(default_factory=list)
    system_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary, omitting None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationParams:
        """Deserialize from dictionary."""
        return cls(**data)

    def with_seed(self) -> GenerationParams:
        """
        Return a copy with a seed, generating one if needed.

        Returns:
            New GenerationParams with seed set.
        """
        if self.seed is not None:
            return self

        # Generate a reproducible seed
        new_seed = generate_seed()
        return GenerationParams(
            seed=new_seed,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            stop=self.stop.copy(),
            system_prompt=self.system_prompt,
        )


@dataclass
class StepModelRecord:
    """
    Record of model usage in a step for reproducibility.

    Stored in StepRecord.model_info for steps that use models.

    Attributes:
        model: Model information
        params: Generation parameters used
        input_hash: Hash of input text (for verification)
        output_tokens: Number of tokens generated
    """

    model: ModelInfo
    params: GenerationParams
    input_hash: str | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "model": self.model.to_dict(),
            "params": self.params.to_dict(),
        }
        if self.input_hash:
            result["input_hash"] = self.input_hash
        if self.output_tokens:
            result["output_tokens"] = self.output_tokens
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepModelRecord:
        """Deserialize from dictionary."""
        return cls(
            model=ModelInfo.from_dict(data["model"]),
            params=GenerationParams.from_dict(data["params"]),
            input_hash=data.get("input_hash"),
            output_tokens=data.get("output_tokens"),
        )


def generate_seed() -> int:
    """
    Generate a random seed for reproducibility.

    Uses system entropy to generate a seed that can be recorded
    and reused for reproducible runs.

    Returns:
        Random integer seed.
    """
    # Use system random for initial seed generation
    # This ensures unpredictability for the initial run
    return random.SystemRandom().randint(0, 2**31 - 1)


def hash_text(text: str) -> str:
    """
    Generate a hash of text content for verification.

    Args:
        text: Text to hash.

    Returns:
        Hex digest of SHA-256 hash.
    """
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


# Default generation parameters for different use cases
DEFAULT_SUMMARIZE_PARAMS = GenerationParams(
    temperature=0.3,  # Lower for more focused summaries
    top_p=0.9,
    max_tokens=1024,
)

DEFAULT_CHAT_PARAMS = GenerationParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048,
)

DEFAULT_DETERMINISTIC_PARAMS = GenerationParams(
    temperature=0.0,  # Fully deterministic
    top_p=1.0,
    top_k=1,
    max_tokens=2048,
)
