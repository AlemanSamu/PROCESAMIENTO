import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from app.core.errors import InvalidProjectStateError
from app.models.schemas import OutputFormat
from app.services.engines.base_engine import ReconstructionResult
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
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reconstruction")
        self._jobs: dict[str, Future] = {}
        self._jobs_lock = Lock()

    @property
    def engine_name(self) -> str:
        return self._engine.name

    def start_processing(self, project_id: str, output_format: OutputFormat) -> str:
        self._ensure_not_running(project_id)
        self.project_service.mark_processing(project_id, output_format)

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
        try:
            images_dir = self.storage_service.get_images_dir(project_id)
            output_dir = self.storage_service.get_output_dir(project_id)
            self.storage_service.clear_output_files(project_id)
            result = self._reconstruct_with_fallback(project_id, images_dir, output_dir, output_format)
            self.project_service.mark_completed(
                project_id,
                output_format,
                result.model_path.name,
                processing_metadata=result.metadata,
            )
        except Exception as exc:
            logger.exception("Reconstruction job failed for project %s", project_id, exc_info=exc)
            self.project_service.mark_failed(
                project_id,
                str(exc),
                processing_metadata={
                    "engine": self._engine.name,
                    "requested_output_format": output_format.value,
                    "failure_reason": str(exc),
                },
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
    ) -> ReconstructionResult:
        try:
            return self._engine.reconstruct(project_id, images_dir, output_dir, output_format)
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
            metadata["engine_requested"] = self._engine.name
            metadata["engine"] = fallback_result.engine_name

            return ReconstructionResult(
                engine_name=fallback_result.engine_name,
                requested_output_format=fallback_result.requested_output_format,
                model_path=fallback_result.model_path,
                metadata=metadata,
            )

    def _should_use_fallback(self) -> bool:
        if self._fallback_engine is None:
            return False
        if self._fallback_engine.name == self._engine.name:
            return False
        return bool(getattr(self.settings, "colmap_fallback_to_mock", True))