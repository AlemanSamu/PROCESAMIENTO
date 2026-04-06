from __future__ import annotations

import time
from pathlib import Path

from app.algorithms.reconstruction_pipeline import ReconstructionPipeline
from app.models.schemas import OutputFormat
from app.services.engines.base_engine import (
    ReconstructionEngine,
    ReconstructionProgressCallback,
    ReconstructionResult,
)


class MockReconstructionEngine(ReconstructionEngine):
    name = "mock"
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
        progress_callback: ReconstructionProgressCallback | None = None,
    ) -> ReconstructionResult:
        started_at = time.perf_counter()

        if progress_callback is not None:
            progress_callback(
                {
                    "engine": self.name,
                    "current_stage": "mock_pipeline",
                    "progress": 0.1,
                    "status_message": "Ejecutando pipeline mock.",
                }
            )

        # Simula tiempo de procesamiento mientras la tuberia produce artefactos reales.
        time.sleep(self.delay_seconds)
        pipeline_result = self.pipeline.execute(
            project_id=project_id,
            images_dir=images_dir,
            output_dir=output_dir,
            output_format=output_format,
        )
        elapsed_seconds = round(time.perf_counter() - started_at, 3)
        metrics = {
            "total_processing_seconds": elapsed_seconds,
            "image_count_processed": pipeline_result.image_count,
            "feature_count": pipeline_result.feature_count,
            "match_count": pipeline_result.match_count,
            "point_count": pipeline_result.point_count,
            "face_count": pipeline_result.face_count,
        }

        metadata = {
            "engine": self.name,
            "processing_seconds": elapsed_seconds,
            "output_path": str(pipeline_result.model_path),
            "report_path": str(pipeline_result.report_path),
            "reconstruction_type": "synthetic_pipeline",
            "image_count_processed": pipeline_result.image_count,
            "feature_count": pipeline_result.feature_count,
            "match_count": pipeline_result.match_count,
            "point_count": pipeline_result.point_count,
            "face_count": pipeline_result.face_count,
            "requested_output_format": output_format.value,
            "actual_output_format": pipeline_result.model_path.suffix.lower().lstrip("."),
            "current_stage": "completed",
            "progress": 1.0,
            "status_message": "Pipeline mock completado.",
            "metrics": metrics,
            "fallback": {
                "used": False,
                "from_engine": None,
                "reason": None,
            },
        }

        if progress_callback is not None:
            progress_callback(
                {
                    "engine": self.name,
                    "current_stage": "completed",
                    "progress": 1.0,
                    "status_message": "Pipeline mock completado.",
                    "metrics": metrics,
                }
            )

        return ReconstructionResult(
            engine_name=self.name,
            requested_output_format=output_format,
            model_path=pipeline_result.model_path,
            metadata=metadata,
        )