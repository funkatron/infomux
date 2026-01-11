"""
Tests for pipeline definition and execution.
"""

from __future__ import annotations

import pytest

from infomux.pipeline_def import (
    DEFAULT_PIPELINE,
    PipelineDef,
    StepDef,
    get_pipeline,
    list_pipelines,
)


class TestStepDef:
    """Tests for StepDef."""

    def test_basic_step(self) -> None:
        """StepDef holds step configuration."""
        step = StepDef(name="test_step")
        assert step.name == "test_step"
        assert step.input_from is None
        assert step.config == {}

    def test_step_with_input(self) -> None:
        """StepDef can specify input source."""
        step = StepDef(name="transcribe", input_from="extract_audio")
        assert step.input_from == "extract_audio"

    def test_step_serialization(self) -> None:
        """StepDef round-trips through dict."""
        step = StepDef(
            name="test",
            input_from="previous",
            config={"key": "value"},
        )
        data = step.to_dict()
        restored = StepDef.from_dict(data)

        assert restored.name == step.name
        assert restored.input_from == step.input_from
        assert restored.config == step.config


class TestPipelineDef:
    """Tests for PipelineDef."""

    def test_basic_pipeline(self) -> None:
        """PipelineDef holds pipeline configuration."""
        pipeline = PipelineDef(
            name="test",
            description="Test pipeline",
            steps=[
                StepDef(name="step1"),
                StepDef(name="step2", input_from="step1"),
            ],
        )
        assert pipeline.name == "test"
        assert len(pipeline.steps) == 2

    def test_step_names(self) -> None:
        """step_names returns ordered list."""
        pipeline = PipelineDef(
            name="test",
            description="",
            steps=[
                StepDef(name="a"),
                StepDef(name="b"),
                StepDef(name="c"),
            ],
        )
        assert pipeline.step_names() == ["a", "b", "c"]

    def test_get_step(self) -> None:
        """get_step finds step by name."""
        step = StepDef(name="target", config={"x": 1})
        pipeline = PipelineDef(
            name="test",
            description="",
            steps=[StepDef(name="other"), step],
        )

        found = pipeline.get_step("target")
        assert found is not None
        assert found.config == {"x": 1}

        assert pipeline.get_step("nonexistent") is None

    def test_serialization(self) -> None:
        """PipelineDef round-trips through dict."""
        pipeline = PipelineDef(
            name="test",
            description="A test",
            steps=[
                StepDef(name="s1"),
                StepDef(name="s2", input_from="s1"),
            ],
        )
        data = pipeline.to_dict()
        restored = PipelineDef.from_dict(data)

        assert restored.name == pipeline.name
        assert restored.description == pipeline.description
        assert len(restored.steps) == 2
        assert restored.steps[1].input_from == "s1"


class TestDefaultPipeline:
    """Tests for the default pipeline."""

    def test_default_exists(self) -> None:
        """DEFAULT_PIPELINE is defined."""
        assert DEFAULT_PIPELINE is not None
        assert DEFAULT_PIPELINE.name == "transcribe"

    def test_default_steps(self) -> None:
        """Default pipeline has expected steps."""
        steps = DEFAULT_PIPELINE.step_names()
        assert steps == ["extract_audio", "transcribe"]

    def test_step_dependencies(self) -> None:
        """Steps have correct input dependencies."""
        extract = DEFAULT_PIPELINE.get_step("extract_audio")
        transcribe = DEFAULT_PIPELINE.get_step("transcribe")

        assert extract is not None
        assert extract.input_from is None  # Uses original input

        assert transcribe is not None
        assert transcribe.input_from == "extract_audio"


class TestPipelineRegistry:
    """Tests for pipeline registry functions."""

    def test_list_pipelines(self) -> None:
        """list_pipelines returns available pipelines."""
        names = list_pipelines()
        assert "transcribe" in names

    def test_get_pipeline_default(self) -> None:
        """get_pipeline(None) returns default."""
        pipeline = get_pipeline(None)
        assert pipeline is DEFAULT_PIPELINE

    def test_get_pipeline_by_name(self) -> None:
        """get_pipeline finds pipeline by name."""
        pipeline = get_pipeline("transcribe")
        assert pipeline.name == "transcribe"

    def test_get_pipeline_unknown(self) -> None:
        """get_pipeline raises for unknown name."""
        with pytest.raises(ValueError, match="Unknown pipeline"):
            get_pipeline("nonexistent")
