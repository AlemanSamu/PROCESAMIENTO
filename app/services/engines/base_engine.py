from abc import ABC, abstractmethod
from pathlib import Path

from app.models.schemas import OutputFormat


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
    ) -> Path:
        raise NotImplementedError
