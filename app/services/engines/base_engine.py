from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.models.schemas import OutputFormat

ReconstructionProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ReconstructionResult:
    engine_name: str
    requested_output_format: OutputFormat
    model_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


class ReconstructionEngine(ABC):
    name = "base"
    is_implemented = True

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def reconstruct(
        self,
        project_id: str,
        images_dir: Path,
        output_dir: Path,
        output_format: OutputFormat,
        progress_callback: ReconstructionProgressCallback | None = None,
    ) -> ReconstructionResult:
        raise NotImplementedError