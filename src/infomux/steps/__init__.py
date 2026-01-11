"""
Pipeline steps for infomux.

Each step is a module that:
- Receives an input (file path or artifact from previous step)
- Produces one or more output artifacts
- Records its execution in the job envelope

Steps are designed to be:
- Small and focused on one task
- Testable in isolation
- Deterministic (same inputs → same outputs)

To create a new step:
1. Create a new file in src/infomux/steps/
2. Define a class with @register_step decorator and a `name` attribute
3. Define a `run(input_path, output_dir, **config) -> StepResult` function
4. Optionally define OUTPUT_FILENAME for input resolution

Steps are auto-discovered on import.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class StepProtocol(Protocol):
    """
    Protocol defining the interface for pipeline steps.

    All steps must implement this interface to be usable in the pipeline.
    """

    name: str

    def execute(self, input_path: Path, output_dir: Path) -> list[Path]:
        """
        Execute the step.

        Args:
            input_path: Path to the input file or artifact.
            output_dir: Directory where output artifacts should be written.

        Returns:
            List of paths to output artifacts produced by this step.

        Raises:
            StepError: If the step fails.
        """
        ...


@dataclass
class StepResult:
    """
    Result of executing a pipeline step.

    Attributes:
        name: Name of the step.
        success: Whether the step completed successfully.
        outputs: List of output artifact paths.
        duration_seconds: How long the step took.
        error: Error message if the step failed.
        model_info: Model and generation parameters (for LLM steps).
    """

    name: str
    success: bool
    outputs: list[Path]
    duration_seconds: float
    error: str | None = None
    model_info: dict | None = None


class StepError(Exception):
    """
    Exception raised when a pipeline step fails.

    Attributes:
        step_name: Name of the step that failed.
        message: Error message.
    """

    def __init__(self, step_name: str, message: str) -> None:
        self.step_name = step_name
        self.message = message
        super().__init__(f"Step '{step_name}' failed: {message}")


@dataclass
class StepInfo:
    """
    Complete info about a registered step.

    Attributes:
        name: Step identifier.
        step_class: The step class.
        run_func: The run() function for executing the step.
        output_filename: Primary output filename (for input resolution).
        module: The module the step was loaded from.
    """

    name: str
    step_class: type
    run_func: Callable[..., StepResult] | None = None
    output_filename: str | None = None
    module: str | None = None
    config_schema: dict[str, Any] = field(default_factory=dict)


# Registry of available steps (populated by auto-discovery)
STEP_REGISTRY: dict[str, StepInfo] = {}

# Flag to track if discovery has run
_discovered = False


def register_step(step_class: type) -> type:
    """
    Decorator to register a step class in the registry.

    Usage:
        @register_step
        class MyStep:
            name = "my_step"
            ...

    Args:
        step_class: The step class to register.

    Returns:
        The step class (unchanged).
    """
    if hasattr(step_class, "name"):
        name = step_class.name
        if isinstance(name, property):
            # Handle dataclass default value
            name = step_class.__dataclass_fields__["name"].default
        STEP_REGISTRY[name] = StepInfo(
            name=name,
            step_class=step_class,
        )
    return step_class


def _discover_steps() -> None:
    """
    Auto-discover and import all step modules.

    This imports every .py file in the steps/ directory (except __init__.py),
    triggering their @register_step decorators and capturing their run()
    functions and OUTPUT_FILENAME constants.
    """
    global _discovered
    if _discovered:
        return

    # Import all submodules in this package
    package_path = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_path)]):
        if module_info.name.startswith("_"):
            continue

        module_name = f"infomux.steps.{module_info.name}"
        try:
            module = importlib.import_module(module_name)

            # Find the step that was registered from this module
            for step_info in STEP_REGISTRY.values():
                if step_info.module is None:
                    # This step was just registered, update its info
                    step_info.module = module_name

                    # Look for run() function
                    if hasattr(module, "run"):
                        step_info.run_func = module.run

                    # Look for output filename constant
                    for attr in dir(module):
                        if attr.endswith("_FILENAME") and not attr.startswith("_"):
                            step_info.output_filename = getattr(module, attr)
                            break

        except ImportError as e:
            # Log but don't fail - step just won't be available
            import sys
            print(f"Warning: failed to import step {module_name}: {e}", file=sys.stderr)

    _discovered = True


def get_step(name: str) -> StepInfo | None:
    """
    Get step info by name.

    Args:
        name: The step name.

    Returns:
        StepInfo, or None if not found.
    """
    _discover_steps()
    return STEP_REGISTRY.get(name)


def get_step_class(name: str) -> type | None:
    """
    Get a step class by name (backwards compatible).

    Args:
        name: The step name.

    Returns:
        The step class, or None if not found.
    """
    info = get_step(name)
    return info.step_class if info else None


def list_steps() -> list[str]:
    """
    List all registered step names.

    Returns:
        List of step names.
    """
    _discover_steps()
    return list(STEP_REGISTRY.keys())


def run_step(
    name: str,
    input_path: Path,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> StepResult:
    """
    Run a step by name.

    Args:
        name: Step name.
        input_path: Input file for this step.
        output_dir: Directory for output artifacts.
        config: Step-specific configuration.

    Returns:
        StepResult with execution details.
    """
    _discover_steps()

    step_info = STEP_REGISTRY.get(name)
    if not step_info:
        return StepResult(
            name=name,
            success=False,
            outputs=[],
            duration_seconds=0,
            error=f"unknown step: {name}",
        )

    if not step_info.run_func:
        return StepResult(
            name=name,
            success=False,
            outputs=[],
            duration_seconds=0,
            error=f"step '{name}' has no run() function",
        )

    # Call the step's run function with config kwargs
    if config:
        return step_info.run_func(input_path, output_dir, **config)
    return step_info.run_func(input_path, output_dir)


def get_step_output(name: str) -> str | None:
    """
    Get the primary output filename for a step.

    Args:
        name: Step name.

    Returns:
        Output filename, or None if not defined.
    """
    _discover_steps()
    step_info = STEP_REGISTRY.get(name)
    return step_info.output_filename if step_info else None
