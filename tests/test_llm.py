"""
Tests for the LLM configuration and reproducibility module.
"""

from __future__ import annotations

from infomux.llm import (
    DEFAULT_SUMMARIZE_PARAMS,
    GenerationParams,
    ModelInfo,
    StepModelRecord,
    generate_seed,
    hash_text,
)


class TestGenerateSeed:
    """Tests for seed generation."""

    def test_generates_integer(self) -> None:
        """Seed is an integer."""
        seed = generate_seed()
        assert isinstance(seed, int)

    def test_generates_positive(self) -> None:
        """Seed is positive."""
        seed = generate_seed()
        assert seed >= 0

    def test_generates_unique(self) -> None:
        """Seeds are unique (with high probability)."""
        seeds = [generate_seed() for _ in range(100)]
        assert len(set(seeds)) == 100


class TestHashText:
    """Tests for text hashing."""

    def test_consistent_hash(self) -> None:
        """Same text produces same hash."""
        text = "hello world"
        hash1 = hash_text(text)
        hash2 = hash_text(text)
        assert hash1 == hash2

    def test_different_text_different_hash(self) -> None:
        """Different text produces different hash."""
        hash1 = hash_text("hello")
        hash2 = hash_text("world")
        assert hash1 != hash2

    def test_hash_format(self) -> None:
        """Hash is hex string of expected length."""
        h = hash_text("test")
        assert len(h) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)


class TestModelInfo:
    """Tests for ModelInfo."""

    def test_basic_model_info(self) -> None:
        """ModelInfo stores basic attributes."""
        info = ModelInfo(name="gpt-4", provider="openai")
        assert info.name == "gpt-4"
        assert info.provider == "openai"
        assert info.version is None
        assert info.path is None

    def test_full_model_info(self) -> None:
        """ModelInfo stores all attributes."""
        info = ModelInfo(
            name="ggml-base.en.bin",
            provider="whisper.cpp",
            version="1.5.0",
            path="/path/to/model",
        )
        assert info.version == "1.5.0"
        assert info.path == "/path/to/model"

    def test_serialization(self) -> None:
        """ModelInfo round-trips through dict."""
        info = ModelInfo(
            name="qwen2.5:7b",
            provider="ollama",
            version="latest",
        )
        data = info.to_dict()
        restored = ModelInfo.from_dict(data)

        assert restored.name == info.name
        assert restored.provider == info.provider
        assert restored.version == info.version

    def test_to_dict_omits_none(self) -> None:
        """to_dict excludes None values."""
        info = ModelInfo(name="test", provider="test")
        data = info.to_dict()
        assert "version" not in data
        assert "path" not in data


class TestGenerationParams:
    """Tests for GenerationParams."""

    def test_default_params(self) -> None:
        """Default params have expected values."""
        params = GenerationParams()
        assert params.seed is None
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.max_tokens == 2048

    def test_with_seed_generates(self) -> None:
        """with_seed generates seed if None."""
        params = GenerationParams()
        assert params.seed is None

        with_seed = params.with_seed()
        assert with_seed.seed is not None
        assert isinstance(with_seed.seed, int)

    def test_with_seed_preserves_existing(self) -> None:
        """with_seed preserves existing seed."""
        params = GenerationParams(seed=42)
        with_seed = params.with_seed()
        assert with_seed.seed == 42

    def test_with_seed_preserves_other_params(self) -> None:
        """with_seed preserves other parameters."""
        params = GenerationParams(temperature=0.5, max_tokens=100)
        with_seed = params.with_seed()
        assert with_seed.temperature == 0.5
        assert with_seed.max_tokens == 100

    def test_serialization(self) -> None:
        """GenerationParams round-trips through dict."""
        params = GenerationParams(
            seed=12345,
            temperature=0.3,
            top_p=0.95,
            stop=["END"],
        )
        data = params.to_dict()
        restored = GenerationParams.from_dict(data)

        assert restored.seed == params.seed
        assert restored.temperature == params.temperature
        assert restored.stop == params.stop


class TestStepModelRecord:
    """Tests for StepModelRecord."""

    def test_basic_record(self) -> None:
        """StepModelRecord stores model and params."""
        model = ModelInfo(name="test", provider="test")
        params = GenerationParams(seed=42)
        record = StepModelRecord(model=model, params=params)

        assert record.model.name == "test"
        assert record.params.seed == 42

    def test_full_record(self) -> None:
        """StepModelRecord stores all fields."""
        model = ModelInfo(name="test", provider="test")
        params = GenerationParams(seed=42)
        record = StepModelRecord(
            model=model,
            params=params,
            input_hash="abc123",
            output_tokens=500,
        )

        assert record.input_hash == "abc123"
        assert record.output_tokens == 500

    def test_serialization(self) -> None:
        """StepModelRecord round-trips through dict."""
        model = ModelInfo(name="gpt-4", provider="openai")
        params = GenerationParams(seed=42, temperature=0.5)
        record = StepModelRecord(
            model=model,
            params=params,
            input_hash="abc",
            output_tokens=100,
        )

        data = record.to_dict()
        restored = StepModelRecord.from_dict(data)

        assert restored.model.name == record.model.name
        assert restored.params.seed == record.params.seed
        assert restored.input_hash == record.input_hash
        assert restored.output_tokens == record.output_tokens


class TestDefaultParams:
    """Tests for default parameter presets."""

    def test_summarize_params(self) -> None:
        """Summarize params have lower temperature."""
        assert DEFAULT_SUMMARIZE_PARAMS.temperature < 0.5
