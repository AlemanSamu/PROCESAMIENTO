import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.errors import InvalidProjectStateError
from app.models.schemas import OutputFormat
from app.services.engines.base_engine import ReconstructionProgressCallback, ReconstructionResult
from app.services.engines.factory import build_reconstruction_engines
from app.services.project_service import ProjectService
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(
        self,
        project_service: ProjectService,
        storage_service: StorageService,
        settings,
    ) -> None:
        self.project_service = project_service
        self.storage_service = storage_service
        self.settings = settings
        self._engine, self._fallback_engine = build_reconstruction_engines(settings)
        self._requested_engine_mode = str(getattr(settings, "processing_engine", "colmap")).lower().strip() or "colmap"
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reconstruction")
        self._jobs: dict[str, Future] = {}
        self._jobs_lock = Lock()

    @property
    def engine_name(self) -> str:
        return self._engine.name

    def start_processing(self, project_id: str, output_format: OutputFormat) -> str:
        self._ensure_not_running(project_id)
        project_metadata = self.project_service.get_project(project_id)
        initial_metadata = self._build_initial_processing_metadata(output_format, project_metadata.image_count)
        self.project_service.mark_processing(
            project_id,
            output_format,
            processing_metadata=initial_metadata,
        )

        future = self._executor.submit(self._run_reconstruction_job, project_id, output_format)
        with self._jobs_lock:
            self._jobs[project_id] = future
        future.add_done_callback(lambda _: self._cleanup_job(project_id))
        return self._engine.name

    def _ensure_not_running(self, project_id: str) -> None:
        with self._jobs_lock:
            existing = self._jobs.get(project_id)
        if existing and not existing.done():
            raise InvalidProjectStateError("El proyecto ya tiene un proceso en ejecucion.")

    def _run_reconstruction_job(self, project_id: str, output_format: OutputFormat) -> None:
        job_started_at = time.perf_counter()
        progress_state: dict[str, Any] | None = None

        try:
            images_dir = self.storage_service.get_images_dir(project_id)
            output_dir = self.storage_service.get_output_dir(project_id)
            image_count = self._count_images(images_dir)
            progress_state = self._build_initial_processing_metadata(output_format, image_count)
            progress_state["current_stage"] = "starting"
            progress_state["progress"] = 0.02
            progress_state["status_message"] = "Inicializando procesamiento con el motor configurado."
            self.project_service.update_processing_metadata(project_id, dict(progress_state))

            logger.info(
                "Starting reconstruction job for project %s with engine=%s output_format=%s image_count=%s",
                project_id,
                self._engine.name,
                output_format.value,
                image_count,
            )

            self.storage_service.clear_output_files(project_id)
            progress_callback = self._build_progress_callback(project_id, progress_state)
            result = self._reconstruct_with_fallback(
                project_id,
                images_dir,
                output_dir,
                output_format,
                progress_callback,
            )
            metadata = self._merge_metadata(progress_state, result.metadata)
            total_seconds = round(time.perf_counter() - job_started_at, 3)
            metrics = dict(metadata.get("metrics") or {})
            metrics["total_processing_seconds"] = total_seconds
            metrics.setdefault("image_count_processed", image_count)
            metadata["metrics"] = metrics
            metadata["engine"] = result.engine_name
            metadata["engine_requested"] = self._requested_engine_mode
            metadata["requested_output_format"] = output_format.value
            metadata["current_stage"] = "completed"
            metadata["progress"] = 1.0
            metadata["status_message"] = "Reconstruccion completada."

            self.project_service.mark_completed(
                project_id,
                output_format,
                result.model_path.name,
                processing_metadata=metadata,
            )
            logger.info(
                "Reconstruction job completed for project %s using engine=%s in %.3fs",
                project_id,
                result.engine_name,
                total_seconds,
            )
        except Exception as exc:
            elapsed_seconds = round(time.perf_counter() - job_started_at, 3)
            logger.exception("Reconstruction job failed for project %s", project_id, exc_info=exc)
            self.project_service.mark_failed(
                project_id,
                str(exc),
                processing_metadata=self._build_failure_metadata(
                    project_id,
                    output_format,
                    str(exc),
                    elapsed_seconds,
                    progress_state,
                ),
            )

    def _cleanup_job(self, project_id: str) -> None:
        with self._jobs_lock:
            self._jobs.pop(project_id, None)

    def _reconstruct_with_fallback(
        self,
        project_id: str,
        images_dir: Path,
        output_dir: Path,
        output_format: OutputFormat,
        progress_callback: ReconstructionProgressCallback | None,
    ) -> ReconstructionResult:
        try:
            return self._engine.reconstruct(
                project_id,
                images_dir,
                output_dir,
                output_format,
                progress_callback=progress_callback,
            )
        except Exception as primary_exc:
            if not self._should_use_fallback():
                raise

            logger.warning(
                "Primary reconstruction engine '%s' failed for project %s. Falling back to '%s'.",
                self._engine.name,
                project_id,
                self._fallback_engine.name,
                exc_info=primary_exc,
            )
            try:
                fallback_result = self._fallback_engine.reconstruct(
                    project_id,
                    images_dir,
                    output_dir,
                    output_format,
                    progress_callback=progress_callback,
                )
            except Exception as fallback_exc:
                raise RuntimeError(
                    "La reconstruccion fallo con COLMAP y tambien con el motor de respaldo. "
                    f"Error primario: {primary_exc}. Error fallback: {fallback_exc}."
                ) from fallback_exc

            metadata = dict(fallback_result.metadata)
            warnings = list(metadata.get("warnings", []))
            warnings.append(
                f"Se uso fallback a '{fallback_result.engine_name}' porque '{self._engine.name}' fallo: {primary_exc}"
            )
            metadata["warnings"] = warnings
            metadata["fallback"] = {
                "used": True,
                "from_engine": self._engine.name,
                "reason": str(primary_exc),
            }
            metadata["engine_requested"] = self._requested_engine_mode
            metadata["engine"] = fallback_result.engine_name

            return ReconstructionResult(
                engine_name=fallback_result.engine_name,
                requested_output_format=fallback_result.requested_output_format,
                model_path=fallback_result.model_path,
                metadata=metadata,
            )

    def _build_initial_processing_metadata(
        self,
        output_format: OutputFormat,
        image_count: int,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        if image_count:
            metrics["image_count_processed"] = image_count

        return {
            "engine": self._engine.name,
            "engine_requested": self._requested_engine_mode,
            "requested_output_format": output_format.value,
            "current_stage": "queued",
            "progress": 0.0,
            "status_message": "Procesamiento en cola.",
            "stage_sequence": [
                "starting",
                "feature_extractor",
                "exhaustive_matcher",
                "mapper",
                "export",
                "completed",
            ],
            "metrics": metrics,
            "fallback": {
                "used": False,
                "from_engine": None,
                "reason": None,
            },
        }

    def _build_progress_callback(
        self,
        project_id: str,
        initial_state: dict[str, Any],
    ) -> ReconstructionProgressCallback:
        state = dict(initial_state)

        def callback(update: dict[str, Any]) -> None:
            merged = self._merge_metadata(state, update)
            progress_value = merged.get("progress")
            if isinstance(progress_value, (int, float)):
                merged["progress"] = max(0.0, min(1.0, float(progress_value)))
            self.project_service.update_processing_metadata(project_id, merged)
            state.clear()
            state.update(merged)

        return callback

    def _build_failure_metadata(
        self,
        project_id: str,
        output_format: OutputFormat,
        reason: str,
        elapsed_seconds: float,
        progress_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        existing_metadata: dict[str, Any] = dict(progress_state or {})
        try:
            persisted = self.project_service.get_project(project_id).processing_metadata or {}
            existing_metadata = self._merge_metadata(existing_metadata, persisted)
        except Exception:
            pass

        failed_stage = str(existing_metadata.get("current_stage") or "starting")
        metrics = dict(existing_metadata.get("metrics") or {})
        metrics["total_processing_seconds"] = elapsed_seconds

        existing_metadata["engine"] = existing_metadata.get("engine") or self._engine.name
        existing_metadata["engine_requested"] = existing_metadata.get("engine_requested") or self._requested_engine_mode
        existing_metadata["requested_output_format"] = existing_metadata.get("requested_output_format") or output_format.value
        existing_metadata["failed_stage"] = failed_stage
        existing_metadata["current_stage"] = "failed"
        existing_metadata["progress"] = float(existing_metadata.get("progress") or 0.0)
        existing_metadata["status_message"] = "Procesamiento fallido."
        existing_metadata["failure_reason"] = reason
        existing_metadata["metrics"] = metrics
        existing_metadata["fallback"] = existing_metadata.get("fallback") or {
            "used": False,
            "from_engine": None,
            "reason": None,
        }
        return existing_metadata

    @staticmethod
    def _merge_metadata(base: dict[str, Any] | None, update: dict[str, Any] | None) -> dict[str, Any]:
        merged: dict[str, Any] = dict(base or {})
        for key, value in (update or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                nested = dict(merged[key])
                nested.update(value)
                merged[key] = nested
                continue
            merged[key] = value
        return merged

    @staticmethod
    def _count_images(images_dir: Path) -> int:
        if not images_dir.exists():
            return 0
        return len([path for path in images_dir.iterdir() if path.is_file()])

    def _should_use_fallback(self) -> bool:
        if self._requested_engine_mode != "auto":
            return False
        if self._fallback_engine is None:
            return False
        if self._fallback_engine.name == self._engine.name:
            return False
        return bool(getattr(self.settings, "colmap_fallback_to_mock", False))