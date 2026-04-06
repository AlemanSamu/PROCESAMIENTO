from __future__ import annotations

import time
from pathlib import Path

from app.algorithms.reconstruction_pipeline import ReconstructionPipeline
from app.models.schemas import OutputFormat
from app.services.engines.base_engine import ReconstructionEngine


class MockReconstructionEngine(ReconstructionEngine):
    name = 'mock'
    is_implemented = True

    def __init__(
        self,
        delay_seconds: int = 5,
        pipeline: ReconstructionPipeline | None = None,
    ) -> None:
        self.delay_seconds = max(delay_seconds, 0)
        self.pipeline = pipeline or ReconstructionPipeline()

    def is_available(self) -> bool:
        return True

    def reconstruct(
        self,
        project_id: str,
        images_dir: Path,
        output_dir: Path,
        output_format: OutputFormat,
    ) -> Path:
        # Simula tiempo de procesamiento mientras la tuberia produce artefactos reales.
        time.sleep(self.delay_seconds)
        pipeline_result = self.pipeline.execute(
            project_id=project_id,
            images_dir=images_dir,
            output_dir=output_dir,
            output_format=output_format,
        )
        return pipeline_result.model_path
