import shutil
from pathlib import Path

from app.core.errors import ProcessingError
from app.models.schemas import OutputFormat
from app.services.engines.base_engine import ReconstructionEngine


class ColmapReconstructionEngine(ReconstructionEngine):
    name = "colmap"
    # Marcado como False para dejar explicito que este archivo es solo un adaptador inicial.
    is_implemented = False

    def __init__(self, colmap_binary: str = "colmap") -> None:
        self.colmap_binary = colmap_binary

    def is_available(self) -> bool:
        return shutil.which(self.colmap_binary) is not None

    def reconstruct(
        self,
        project_id: str,
        images_dir: Path,
        output_dir: Path,
        output_format: OutputFormat,
    ) -> Path:
        raise ProcessingError(
            "El adaptador COLMAP aun no ejecuta reconstruccion real. "
            "Implementa aqui el pipeline con tus comandos de COLMAP."
        )
