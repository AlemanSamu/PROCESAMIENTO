import importlib
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.errors import InvalidProjectStateError, ProcessingError
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
            metadata = self._standardize_final_metadata(project_id, metadata, result.model_path, output_format)
            final_stage = str(metadata.get("current_stage") or "completed")
            if final_stage not in {"completed", "completed_with_fallback"}:
                final_stage = "completed"
            metadata["current_stage"] = final_stage
            metadata["progress"] = 1.0
            metadata["status_message"] = (
                "Reconstruccion completada con fallback sparse."
                if final_stage == "completed_with_fallback"
                else "Reconstruccion completada."
            )

            self._validate_final_result_artifact(project_id, result, output_format, metadata)

            self.project_service.mark_completed(
                project_id,
                output_format,
                result.model_path.name,
                processing_metadata=metadata,
            )
            logger.info(
                "Reconstruction job completed for project %s using engine=%s in %.3fs current_stage=%s fallback_used=%s method_used=%s model_path=%s",
                project_id,
                result.engine_name,
                total_seconds,
                metadata.get("current_stage"),
                metadata.get("fallback_used"),
                metadata.get("method_used"),
                metadata.get("final_model_path"),
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
                    error_context=self._extract_processing_error_context(exc),
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
            if isinstance(primary_exc, ProcessingError) and not getattr(primary_exc, "allow_fallback", True):
                logger.info(
                    "Primary reconstruction engine failure for project %s is not eligible for fallback. reason_code=%s current_stage=%s",
                    project_id,
                    getattr(primary_exc, "reason_code", None),
                    getattr(primary_exc, "current_stage", None),
                )
                raise
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
                "mapper_failed_insufficient_registered_images",
                "model_converter_txt",
                "model_converter_sparse_ply",
                "sparse_mesh_fallback",
                "image_undistorter",
                "patch_match_stereo",
                "stereo_fusion",
                "poisson_mesher",
                "export",
                "completed_with_fallback",
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
        error_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing_metadata: dict[str, Any] = dict(progress_state or {})
        try:
            persisted = self.project_service.get_project(project_id).processing_metadata or {}
            existing_metadata = self._merge_metadata(existing_metadata, persisted)
        except Exception:
            pass

        if error_context:
            existing_metadata = self._merge_metadata(existing_metadata, error_context)

        explicit_failure_stage = str((error_context or {}).get("current_stage") or "").strip()
        previous_stage = str(existing_metadata.get("current_stage") or "starting").strip() or "starting"
        failed_stage = explicit_failure_stage or previous_stage
        metrics = dict(existing_metadata.get("metrics") or {})
        metrics["total_processing_seconds"] = elapsed_seconds

        if existing_metadata.get("registered_image_count") is not None:
            metrics["registered_image_count"] = existing_metadata.get("registered_image_count")

        existing_metadata["engine"] = existing_metadata.get("engine") or self._engine.name
        existing_metadata["engine_requested"] = existing_metadata.get("engine_requested") or self._requested_engine_mode
        existing_metadata["requested_output_format"] = existing_metadata.get("requested_output_format") or output_format.value
        existing_metadata["failed_stage"] = failed_stage
        existing_metadata["current_stage"] = explicit_failure_stage or "failed"
        existing_metadata["progress"] = float(existing_metadata.get("progress") or 0.0)
        existing_metadata["status_message"] = (
            existing_metadata.get("status_message")
            if explicit_failure_stage
            else "Procesamiento fallido."
        )
        existing_metadata["failure_reason"] = reason
        existing_metadata["metrics"] = metrics
        existing_metadata["fallback"] = existing_metadata.get("fallback") or {
            "used": False,
            "from_engine": None,
            "reason": None,
        }
        return existing_metadata


    @staticmethod
    def _extract_processing_error_context(exc: Exception) -> dict[str, Any]:
        if not isinstance(exc, ProcessingError):
            return {}

        context: dict[str, Any] = {}
        current_stage = getattr(exc, "current_stage", None)
        if current_stage:
            context["current_stage"] = current_stage

        reason_code = getattr(exc, "reason_code", None)
        if reason_code:
            context["reason_code"] = reason_code

        metadata = getattr(exc, "metadata", None)
        if isinstance(metadata, dict):
            context = ProcessingService._merge_metadata(context, metadata)

        return context


    def _standardize_final_metadata(
        self,
        project_id: str,
        metadata: dict[str, Any],
        model_path: Path,
        output_format: OutputFormat,
    ) -> dict[str, Any]:
        normalized = dict(metadata)
        artifacts = dict(normalized.get("artifacts") or {})
        artifacts["model_path"] = str(model_path)
        normalized["artifacts"] = artifacts
        normalized["final_model_path"] = str(model_path)
        normalized["final_model_type"] = model_path.suffix.lower().lstrip(".") or None
        normalized["fallback_used"] = bool(
            normalized.get("fallback_used")
            or (normalized.get("fallback") or {}).get("used")
            or (normalized.get("sparse_fallback") or {}).get("used")
        )
        normalized["method_used"] = (
            normalized.get("method_used")
            or (normalized.get("sparse_fallback") or {}).get("mesh_method")
            or normalized.get("meshing_method")
        )
        normalized["requested_output_format"] = output_format.value
        logger.info(
            "Project %s final metadata standardized. current_stage=%s fallback_used=%s method_used=%s final_model_type=%s final_model_path=%s",
            project_id,
            normalized.get("current_stage"),
            normalized.get("fallback_used"),
            normalized.get("method_used"),
            normalized.get("final_model_type"),
            normalized.get("final_model_path"),
        )
        return normalized

    def _validate_final_result_artifact(
        self,
        project_id: str,
        result: ReconstructionResult,
        output_format: OutputFormat,
        metadata: dict[str, Any],
    ) -> None:
        model_path = result.model_path
        if not model_path.exists() or not model_path.is_file() or model_path.stat().st_size <= 0:
            raise RuntimeError(f"El artefacto final no existe o esta vacio: {model_path}")

        expected_suffix = f".{output_format.value}"
        if model_path.suffix.lower() != expected_suffix:
            raise RuntimeError(
                "El procesamiento genero un artefacto final inconsistente. "
                f"Esperado: {expected_suffix}. Encontrado: {model_path.suffix.lower() or 'sin extension'}."
            )

        if output_format != OutputFormat.GLB:
            return

        trimesh_module = self._import_trimesh_validator()
        summary = self._inspect_glb_mesh(trimesh_module, model_path)
        metrics = dict(metadata.get("metrics") or {})
        metrics["mesh_vertex_count"] = summary["vertex_count"]
        metrics["mesh_face_count"] = summary["face_count"]
        metadata["metrics"] = metrics
        logger.info(
            "Validated final GLB for project %s. vertices=%s faces=%s model_path=%s",
            project_id,
            summary["vertex_count"],
            summary["face_count"],
            model_path,
        )

    @staticmethod
    def _import_trimesh_validator() -> object:
        try:
            return importlib.import_module("trimesh")
        except ImportError as exc:
            raise RuntimeError(
                "No se pudo validar el GLB final porque falta la dependencia 'trimesh'."
            ) from exc

    @classmethod
    def _inspect_glb_mesh(cls, trimesh_module: object, model_path: Path) -> dict[str, int]:
        try:
            mesh_asset = trimesh_module.load(str(model_path), file_type="glb", force="scene")
        except Exception as exc:
            raise RuntimeError(f"No se pudo abrir el GLB final para validarlo: {exc}") from exc

        summary = cls._extract_mesh_counts(mesh_asset)
        if summary["vertex_count"] <= 0:
            raise RuntimeError(
                "El GLB final no contiene vertices utiles. No se puede marcar el proyecto como completado."
            )
        if summary["face_count"] <= 0:
            raise RuntimeError(
                "El GLB final no contiene caras. El visor estaria recibiendo una nube de puntos, no una malla valida."
            )
        return summary

    @classmethod
    def _extract_mesh_counts(cls, mesh_asset: object) -> dict[str, int]:
        geometry = getattr(mesh_asset, "geometry", None)
        if isinstance(geometry, dict):
            vertex_count = sum(cls._safe_len(getattr(item, "vertices", ())) for item in geometry.values())
            face_count = sum(cls._safe_len(getattr(item, "faces", ())) for item in geometry.values())
            return {
                "vertex_count": vertex_count,
                "face_count": face_count,
            }
        return {
            "vertex_count": cls._safe_len(getattr(mesh_asset, "vertices", ())),
            "face_count": cls._safe_len(getattr(mesh_asset, "faces", ())),
        }

    @staticmethod
    def _safe_len(values: object) -> int:
        try:
            return len(values)
        except TypeError:
            return 0

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