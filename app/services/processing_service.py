import importlib
import json
import logging
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from app.algorithms.box_primitive_fallback import BoxPrimitiveFallback
from app.algorithms.artifacts import write_json
from app.algorithms.image_preprocessor import ImagePreprocessor
from app.algorithms.input_image_selector import InputImageSelector
from app.algorithms.input_image_validator import InputImageValidator
from app.algorithms.input_object_segmenter import InputObjectSegmenter
from app.core.errors import InvalidProjectStateError, ProcessingError
from app.models.schemas import OutputFormat, ProjectStatus
from app.services.engines.base_engine import ReconstructionProgressCallback, ReconstructionResult
from app.services.engines.colmap_engine import ColmapReconstructionEngine
from app.services.engines.factory import build_reconstruction_engines
from app.services.presentation_postprocess_service import PresentationPostprocessService
from app.services.project_service import ProjectService
from app.services.reconstruction_calibration import (
    HISTORY_PATH_DEFAULT,
    append_history_record,
    to_final_success_level,
)
from app.services.storage_service import StorageService
from app.services.technical_evidence_service import TechnicalEvidenceService

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
        self._input_image_validator = InputImageValidator.from_settings(settings)
        self._input_image_selector = InputImageSelector.from_settings(settings)
        self._input_object_segmenter = InputObjectSegmenter.from_settings(settings)
        self._image_preprocessor = ImagePreprocessor.from_settings(settings)
        self._image_preprocessing_enabled = bool(
            getattr(
                settings,
                "image_preprocessing_enabled",
                hasattr(settings, "profile") or hasattr(settings, "image_preprocessing_max_width"),
            )
        )
        self._box_primitive_fallback = BoxPrimitiveFallback.from_settings(settings)
        self._presentation_postprocess = PresentationPostprocessService()
        self._technical_evidence = TechnicalEvidenceService(settings)
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
        started_at_iso = self._utc_now_iso()
        progress_state: dict[str, Any] | None = None
        stage_tracker = self._create_stage_tracker(started_at_iso)
        images_dir: Path | None = None
        selected_images_dir: Path | None = None
        engine_images_dir: Path | None = None
        output_dir: Path | None = None
        selected_image_count = 0

        try:
            images_dir = self.storage_service.get_images_dir(project_id)
            output_dir = self.storage_service.get_output_dir(project_id)
            image_count = self._count_images(images_dir)
            progress_state = self._build_initial_processing_metadata(output_format, image_count)
            progress_state["current_stage"] = "starting"
            progress_state["workflow_stage"] = "reconstructing"
            progress_state["stage_status"] = "running"
            progress_state["progress"] = 0.02
            progress_state["status_message"] = "Inicializando procesamiento con el motor configurado."
            self._update_stage_tracker(stage_tracker, progress_state, event_type="stage_update")
            self._safe_update_processing_metadata(project_id, progress_state)

            logger.info(
                "Starting reconstruction job for project %s with engine=%s output_format=%s image_count=%s",
                project_id,
                self._engine.name,
                output_format.value,
                image_count,
            )

            self.storage_service.clear_output_files(project_id)
            selected_images_dir, selected_image_count = self._validate_input_images(
                project_id,
                images_dir,
                output_dir,
                progress_state,
                stage_tracker=stage_tracker,
            )
            if self._image_preprocessing_enabled:
                engine_images_dir, selected_image_count = self._preprocess_selected_images(
                    project_id=project_id,
                    selected_images_dir=selected_images_dir,
                    output_dir=output_dir,
                    progress_state=progress_state,
                    stage_tracker=stage_tracker,
                )
            else:
                engine_images_dir = selected_images_dir
            engine_images_dir = self._prefer_colmap_preprocessed_images(
                engine_images_dir=engine_images_dir,
                output_dir=output_dir,
                progress_state=progress_state,
            )
            selected_image_count = self._count_images(engine_images_dir)
            progress_state["current_stage"] = "starting"
            progress_state["workflow_stage"] = "reconstructing"
            progress_state["stage_status"] = "running"
            progress_state["progress"] = max(float(progress_state.get("progress") or 0.0), 0.16)
            progress_state["status_message"] = "Iniciando etapas del engine de reconstruccion."
            self._update_stage_tracker(stage_tracker, progress_state, event_type="stage_update")
            self._safe_update_processing_metadata(project_id, progress_state)
            progress_callback = self._build_progress_callback(
                project_id,
                progress_state,
                stage_tracker=stage_tracker,
            )
            result = self._reconstruct_with_fallback(
                project_id,
                engine_images_dir,
                output_dir,
                output_format,
                progress_callback,
            )
            metadata = self._merge_metadata(progress_state, result.metadata)
            total_seconds = round(time.perf_counter() - job_started_at, 3)
            metrics = dict(metadata.get("metrics") or {})
            metrics["total_processing_seconds"] = total_seconds
            metrics.setdefault("image_count_received", image_count)
            metrics["image_count_processed"] = selected_image_count
            metadata["metrics"] = metrics
            metadata["engine"] = result.engine_name
            metadata["engine_requested"] = self._requested_engine_mode
            metadata["requested_output_format"] = output_format.value
            metadata = self._standardize_final_metadata(project_id, metadata, result.model_path, output_format)
            incoherent_reason = self._detect_incoherent_result_reason(metadata)
            if incoherent_reason is not None:
                quality_gate_metadata = self._merge_metadata(
                    metadata,
                    {
                        "current_stage": "quality_gate_incoherent_output",
                        "stage_status": "failed",
                        "reason_code": "incoherent_reconstruction_output",
                        "status_message": (
                            "La salida del engine fue considerada incoherente para la evidencia final controlada. "
                            f"Detalle: {incoherent_reason}."
                        ),
                        "incoherent_output": {
                            "detected": True,
                            "reason": incoherent_reason,
                        },
                    },
                )
                raise ProcessingError(
                    "La salida del engine se considero incoherente para el caso controlado de evidencia final.",
                    reason_code="incoherent_reconstruction_output",
                    current_stage="quality_gate_incoherent_output",
                    metadata=quality_gate_metadata,
                    allow_fallback=False,
                    retryable=True,
                )
            final_stage = str(metadata.get("current_stage") or "completed")
            if final_stage not in {"completed", "completed_with_fallback"}:
                final_stage = "completed"
            metadata["current_stage"] = final_stage
            metadata["stage_status"] = "completed"
            metadata["progress"] = 1.0
            metadata["status_message"] = (
                "Reconstruccion completada con fallback sparse."
                if final_stage == "completed_with_fallback"
                else "Reconstruccion completada."
            )
            result, metadata = self._apply_presentation_postprocess_if_configured(
                project_id=project_id,
                output_dir=output_dir,
                output_format=output_format,
                result=result,
                metadata=metadata,
            )
            result, metadata = self._apply_forced_presentable_model_if_configured(
                project_id=project_id,
                output_dir=output_dir,
                output_format=output_format,
                result=result,
                metadata=metadata,
            )
            metadata = self._standardize_final_metadata(
                project_id,
                metadata,
                result.model_path,
                output_format,
            )
            self._update_stage_tracker(stage_tracker, metadata, event_type="job_completed")
            metadata["execution_report"] = self._finalize_execution_report(
                stage_tracker=stage_tracker,
                metadata=metadata,
                elapsed_seconds=total_seconds,
                outcome="completed",
            )
            self._validate_final_result_artifact(project_id, result, output_format, metadata)
            if not bool((metadata.get("presentation_postprocess") or {}).get("applied")):
                self._remove_non_canonical_model_variants(project_id, output_dir, result.model_path)
            metadata = self._write_quality_report(
                project_id=project_id,
                output_dir=output_dir,
                project_status=ProjectStatus.COMPLETED,
                metadata=metadata,
                model_path=result.model_path,
            )
            self._append_reconstruction_history(
                project_id=project_id,
                dataset_path=engine_images_dir or images_dir,
                metadata=metadata,
            )
            self._write_execution_report_file(project_id, output_dir, metadata)
            self._write_technical_evidence_file(
                project_id=project_id,
                output_dir=output_dir,
                metadata=metadata,
                project_status="completed",
            )

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
            error_context = self._extract_processing_error_context(exc)
            failed_stage = str((error_context or {}).get("current_stage") or "").strip()
            if failed_stage:
                tracker_payload = self._merge_metadata(
                    progress_state,
                    {
                        "current_stage": failed_stage,
                        "stage_status": "failed",
                        "status_message": str(exc),
                    },
                )
                self._update_stage_tracker(stage_tracker, tracker_payload, event_type="stage_failed")
            failure_metadata = self._build_failure_metadata(
                project_id,
                output_format,
                str(exc),
                elapsed_seconds,
                progress_state,
                stage_tracker=stage_tracker,
                images_dir=images_dir,
                selected_images_dir=selected_images_dir,
                engine_images_dir=engine_images_dir,
                output_dir=output_dir,
                error_context=error_context,
            )
            box_fallback_completion = self._attempt_box_primitive_fallback(
                project_id=project_id,
                requested_output_format=output_format,
                elapsed_seconds=elapsed_seconds,
                failure_reason=str(exc),
                error_context=error_context,
                progress_state=progress_state,
                stage_tracker=stage_tracker,
                selected_images_dir=selected_images_dir,
                engine_images_dir=engine_images_dir,
                output_dir=output_dir,
            )
            if box_fallback_completion is not None:
                completed_output_format, completed_metadata, completed_model_path = box_fallback_completion
                if output_dir is not None:
                    completed_metadata = self._write_quality_report(
                        project_id=project_id,
                        output_dir=output_dir,
                        project_status=ProjectStatus.COMPLETED,
                        metadata=completed_metadata,
                        model_path=completed_model_path,
                    )
                    self._append_reconstruction_history(
                        project_id=project_id,
                        dataset_path=engine_images_dir or selected_images_dir or images_dir,
                        metadata=completed_metadata,
                    )
                    self._write_execution_report_file(project_id, output_dir, completed_metadata)
                    self._write_technical_evidence_file(
                        project_id=project_id,
                        output_dir=output_dir,
                        metadata=completed_metadata,
                        project_status="completed",
                    )
                self.project_service.mark_completed(
                    project_id,
                    completed_output_format,
                    completed_model_path.name,
                    processing_metadata=completed_metadata,
                )
                logger.info(
                    "Project %s recovered with primitive box fallback. output_format=%s model_path=%s reason_code=%s",
                    project_id,
                    completed_output_format.value,
                    completed_model_path,
                    completed_metadata.get("reason_code"),
                )
                return
            if output_dir is not None:
                self._cleanup_failed_outputs(project_id, output_dir, failure_metadata)
                failure_metadata = self._write_quality_report(
                    project_id=project_id,
                    output_dir=output_dir,
                    project_status=ProjectStatus.FAILED,
                    metadata=failure_metadata,
                    model_path=None,
                )
                self._append_reconstruction_history(
                    project_id=project_id,
                    dataset_path=engine_images_dir or selected_images_dir or images_dir,
                    metadata=failure_metadata,
                )
                self._write_execution_report_file(project_id, output_dir, failure_metadata)
                self._write_technical_evidence_file(
                    project_id=project_id,
                    output_dir=output_dir,
                    metadata=failure_metadata,
                    project_status="failed",
                )

            try:
                self.project_service.mark_failed(
                    project_id,
                    str(exc),
                    processing_metadata=failure_metadata,
                )
            except Exception as mark_exc:
                logger.exception(
                    "Unable to persist FAILED state for project %s after processing error: %s",
                    project_id,
                    mark_exc,
                    exc_info=mark_exc,
                )

    def _attempt_box_primitive_fallback(
        self,
        *,
        project_id: str,
        requested_output_format: OutputFormat,
        elapsed_seconds: float,
        failure_reason: str,
        error_context: dict[str, Any] | None,
        progress_state: dict[str, Any] | None,
        stage_tracker: dict[str, Any] | None,
        selected_images_dir: Path | None,
        engine_images_dir: Path | None,
        output_dir: Path | None,
    ) -> tuple[OutputFormat, dict[str, Any], Path] | None:
        if not self._should_attempt_box_primitive_fallback(
            error_context=error_context,
            selected_images_dir=engine_images_dir or selected_images_dir,
            output_dir=output_dir,
        ):
            return None

        selected_images_dir = engine_images_dir or selected_images_dir
        assert selected_images_dir is not None
        assert output_dir is not None
        stage_state = self._merge_metadata(progress_state, {})
        stage_state = self._merge_metadata(
            stage_state,
            {
                "current_stage": "primitive_box_fallback",
                "stage_status": "running",
                "progress": max(float(stage_state.get("progress") or 0.0), 0.84),
                "status_message": "Intentando fallback geometrico tipo box para recuperar una salida limpia.",
            },
        )
        self._update_stage_tracker(stage_tracker, stage_state, event_type="stage_update")
        self._safe_update_processing_metadata(project_id, stage_state)

        attempted_formats: list[OutputFormat] = [requested_output_format]
        segmentation_summary = stage_state.get("input_object_segmentation")
        if not isinstance(segmentation_summary, dict):
            segmentation_summary = (error_context or {}).get("input_object_segmentation")
        if not isinstance(segmentation_summary, dict):
            segmentation_summary = None

        last_error: Exception | None = None
        for candidate_format in attempted_formats:
            try:
                fallback_result = self._box_primitive_fallback.build_from_images(
                    project_id=project_id,
                    selected_images_dir=selected_images_dir,
                    output_dir=output_dir,
                    output_format=candidate_format,
                    source_reason=failure_reason,
                    segmentation_summary=segmentation_summary,
                )
                metadata = self._merge_metadata(stage_state, fallback_result.metadata)
                metadata["engine"] = metadata.get("engine") or self._engine.name
                metadata["engine_requested"] = metadata.get("engine_requested") or self._requested_engine_mode
                metadata["requested_output_format"] = requested_output_format.value
                metadata["actual_output_format"] = fallback_result.output_format.value
                metadata["reason_code"] = "fallback_box_used"
                metadata["failed_stage"] = str((error_context or {}).get("current_stage") or "").strip() or None
                metadata["failure_reason"] = failure_reason
                metadata["retryable"] = False
                metadata["can_retry"] = False
                metadata["stage_status"] = "completed"
                metadata["progress"] = 1.0
                metadata["workflow_stage"] = "fallback_completed"
                metadata["current_stage"] = "completed_with_fallback"
                metadata["status_message"] = "Reconstruccion completada con fallback geometrico tipo box."
                metadata["fallback"] = {
                    "used": True,
                    "from_engine": self._engine.name,
                    "reason": failure_reason,
                    "fallback_engine": "primitive_box_fallback",
                }
                metadata["fallback_used"] = True
                metadata = self._write_academic_fallback_report(
                    project_id=project_id,
                    output_dir=output_dir,
                    metadata=metadata,
                    failure_reason=failure_reason,
                    error_context=error_context,
                    selected_images_dir=selected_images_dir,
                    model_path=fallback_result.model_path,
                )
                metadata = self._standardize_final_metadata(
                    project_id,
                    metadata,
                    fallback_result.model_path,
                    fallback_result.output_format,
                )
                metrics = dict(metadata.get("metrics") or {})
                metrics["total_processing_seconds"] = elapsed_seconds
                metrics.setdefault("image_count_processed", self._count_images(selected_images_dir))
                metadata["metrics"] = metrics
                self._validate_final_result_artifact(
                    project_id,
                    ReconstructionResult(
                        engine_name=str(metadata.get("engine") or self._engine.name),
                        requested_output_format=fallback_result.output_format,
                        model_path=fallback_result.model_path,
                        metadata=metadata,
                    ),
                    fallback_result.output_format,
                    metadata,
                )
                self._remove_non_canonical_model_variants(project_id, output_dir, fallback_result.model_path)
                self._update_stage_tracker(stage_tracker, metadata, event_type="job_completed")
                metadata["execution_report"] = self._finalize_execution_report(
                    stage_tracker=stage_tracker,
                    metadata=metadata,
                    elapsed_seconds=elapsed_seconds,
                    outcome="completed",
                )
                return fallback_result.output_format, metadata, fallback_result.model_path
            except Exception as fallback_exc:
                last_error = fallback_exc
                logger.warning(
                    "Primitive box fallback failed for project %s with format=%s: %s",
                    project_id,
                    candidate_format.value,
                    fallback_exc,
                    exc_info=fallback_exc,
                )
                continue

        if last_error is not None:
            logger.warning(
                "Primitive box fallback was attempted but failed for project %s. Last error: %s",
                project_id,
                last_error,
            )
        return None

    def _should_attempt_box_primitive_fallback(
        self,
        *,
        error_context: dict[str, Any] | None,
        selected_images_dir: Path | None,
        output_dir: Path | None,
    ) -> bool:
        if not bool(self._box_primitive_fallback.settings.enabled):
            return False
        if selected_images_dir is None or output_dir is None:
            return False
        if not selected_images_dir.exists() or not selected_images_dir.is_dir():
            return False

        reason_code = str((error_context or {}).get("reason_code") or "").strip().lower()
        current_stage = str((error_context or {}).get("current_stage") or "").strip().lower()
        non_eligible_reason_codes = {
            "input_validation_failed",
            "input_selection_failed",
            "input_object_segmentation_failed",
            "insufficient_input_images",
            "box_fallback_insufficient_images",
        }
        if reason_code in non_eligible_reason_codes or current_stage in non_eligible_reason_codes:
            return False

        selected_count = self._count_images(selected_images_dir)
        if selected_count < self._box_primitive_fallback.settings.min_selected_images:
            return False
        return True

    def _preprocess_selected_images(
        self,
        *,
        project_id: str,
        selected_images_dir: Path,
        output_dir: Path,
        progress_state: dict[str, Any],
        stage_tracker: dict[str, Any] | None = None,
    ) -> tuple[Path, int]:
        progress_state["current_stage"] = "preprocessing"
        progress_state["workflow_stage"] = "preprocessing"
        progress_state["stage_status"] = "running"
        progress_state["progress"] = max(float(progress_state.get("progress") or 0.0), 0.16)
        progress_state["status_message"] = "Preprocesando imagenes para COLMAP/fallback."
        self._update_stage_tracker(stage_tracker, progress_state, event_type="stage_update")
        self._safe_update_processing_metadata(project_id, progress_state)

        pipeline_dir = output_dir / "pipeline"
        preprocessed_dir = output_dir / "preprocessed_images"
        preprocessed_images, report = self._image_preprocessor.run(
            selected_images_dir,
            pipeline_dir,
            output_images_dir=preprocessed_dir,
        )
        preprocessed_count = len(preprocessed_images)
        metrics = dict(progress_state.get("metrics") or {})
        metrics["image_count_preprocessed"] = preprocessed_count
        artifacts = dict(progress_state.get("artifacts") or {})
        artifacts["preprocessing_manifest"] = str(report.artifact_path) if report.artifact_path is not None else None
        artifacts["preprocessed_images_dir"] = str(preprocessed_dir)
        preprocessing_summary = {
            "profile": self._image_preprocessor.profile,
            "max_width": self._image_preprocessor.max_width,
            "image_count": preprocessed_count,
            "mode": report.mode,
            "manifest_path": str(report.artifact_path) if report.artifact_path is not None else None,
            "output_images_dir": str(preprocessed_dir),
            "metrics": report.metrics,
        }
        progress_state.update(
            {
                "current_stage": "preprocessing",
                "workflow_stage": "preprocessing",
                "stage_status": "completed",
                "progress": max(float(progress_state.get("progress") or 0.0), 0.20),
                "status_message": "Preprocesamiento de imagenes completado.",
                "preprocessing": preprocessing_summary,
                "metrics": metrics,
                "artifacts": artifacts,
            }
        )
        self._update_stage_tracker(stage_tracker, progress_state, event_type="stage_completed")
        self._safe_update_processing_metadata(project_id, progress_state)
        return (preprocessed_dir if preprocessed_count > 0 else selected_images_dir), preprocessed_count

    def _prefer_colmap_preprocessed_images(
        self,
        *,
        engine_images_dir: Path,
        output_dir: Path,
        progress_state: dict[str, Any],
    ) -> Path:
        if str(self._engine.name).strip().lower() != "colmap":
            return engine_images_dir

        preprocessed_dir = output_dir / "preprocessed_images"
        if not preprocessed_dir.exists() or not preprocessed_dir.is_dir():
            return engine_images_dir
        if self._count_images(preprocessed_dir) <= 0:
            return engine_images_dir

        try:
            if engine_images_dir.resolve() == preprocessed_dir.resolve():
                return engine_images_dir
        except Exception:
            if str(engine_images_dir) == str(preprocessed_dir):
                return engine_images_dir

        artifacts = dict(progress_state.get("artifacts") or {})
        artifacts["engine_images_dir"] = str(preprocessed_dir)
        progress_state["artifacts"] = artifacts
        progress_state["status_message"] = "COLMAP usara imagenes preprocesadas detectadas en output/preprocessed_images."
        return preprocessed_dir

    def _detect_incoherent_result_reason(self, metadata: dict[str, Any]) -> str | None:
        if not bool(getattr(self.settings, "primitive_box_fallback_on_incoherent_output", False)):
            return None

        reconstruction_type = str(metadata.get("reconstruction_type") or "").strip().lower()
        sparse_fallback = metadata.get("sparse_fallback") if isinstance(metadata.get("sparse_fallback"), dict) else {}
        method_used = str(
            metadata.get("method_used")
            or sparse_fallback.get("mesh_method")
            or sparse_fallback.get("final_mesh_method")
            or ""
        ).strip().lower()
        sparse_methods = {"delaunay_mesher_sparse", "convex_hull", "bounding_box"}
        if reconstruction_type != "sparse_photogrammetry_mesh_fallback" and method_used not in sparse_methods:
            return None

        metrics = metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {}
        reasons: list[str] = []

        registered_image_count = self._safe_int(
            metadata.get("registered_image_count") or metrics.get("reconstructed_camera_count")
        )
        min_registered_images = max(
            2,
            int(getattr(self.settings, "primitive_box_fallback_incoherent_min_registered_images", 8)),
        )
        if registered_image_count is not None and registered_image_count < min_registered_images:
            reasons.append(
                f"registered_image_count={registered_image_count} below threshold={min_registered_images}"
            )

        sparse_point_count = self._safe_int(
            metadata.get("point_count")
            or metrics.get("point_3d_count")
            or metrics.get("sparse_point_cloud_count")
        )
        min_sparse_points = max(
            1,
            int(getattr(self.settings, "primitive_box_fallback_incoherent_min_sparse_points", 1200)),
        )
        if sparse_point_count is not None and sparse_point_count < min_sparse_points:
            reasons.append(f"sparse_point_count={sparse_point_count} below threshold={min_sparse_points}")

        min_points_per_registered = max(
            1,
            int(getattr(self.settings, "primitive_box_fallback_incoherent_min_points_per_registered_image", 180)),
        )
        if (
            sparse_point_count is not None
            and registered_image_count is not None
            and registered_image_count > 0
        ):
            points_per_registered = sparse_point_count / registered_image_count
            if points_per_registered < float(min_points_per_registered):
                reasons.append(
                    "points_per_registered_image="
                    f"{round(points_per_registered, 3)} below threshold={min_points_per_registered}"
                )

        mesh_face_count = self._safe_int(metrics.get("mesh_face_count") or metadata.get("mesh_face_count"))
        min_faces = max(1, int(getattr(self.settings, "primitive_box_fallback_incoherent_min_faces", 20)))
        if mesh_face_count is not None and mesh_face_count < min_faces:
            reasons.append(f"mesh_face_count={mesh_face_count} below threshold={min_faces}")

        max_faces = max(
            min_faces,
            int(getattr(self.settings, "primitive_box_fallback_incoherent_max_faces", 700)),
        )
        if mesh_face_count is not None and mesh_face_count > max_faces:
            reasons.append(f"mesh_face_count={mesh_face_count} above threshold={max_faces}")

        shape_diagnostics = (
            sparse_fallback.get("shape_diagnostics")
            if isinstance(sparse_fallback.get("shape_diagnostics"), dict)
            else {}
        )
        extent_ratio = self._safe_float(shape_diagnostics.get("extent_ratio_max_min"))
        max_extent_ratio = max(
            1.0,
            float(getattr(self.settings, "primitive_box_fallback_incoherent_max_extent_ratio", 6.0)),
        )
        if extent_ratio is not None and extent_ratio > max_extent_ratio:
            reasons.append(
                f"extent_ratio_max_min={round(extent_ratio, 3)} above threshold={max_extent_ratio}"
            )

        bbox_fill_ratio = self._safe_float(shape_diagnostics.get("mesh_volume_to_bbox_volume_ratio"))
        min_fill_ratio = max(
            0.0,
            min(1.0, float(getattr(self.settings, "primitive_box_fallback_incoherent_min_bbox_fill_ratio", 0.12))),
        )
        max_fill_ratio = max(
            min_fill_ratio,
            float(getattr(self.settings, "primitive_box_fallback_incoherent_max_bbox_fill_ratio", 1.05)),
        )
        if bbox_fill_ratio is not None and (
            bbox_fill_ratio < min_fill_ratio or bbox_fill_ratio > max_fill_ratio
        ):
            reasons.append(
                "mesh_volume_to_bbox_volume_ratio="
                f"{round(bbox_fill_ratio, 3)} outside range=[{min_fill_ratio}, {max_fill_ratio}]"
            )

        if bool(getattr(self.settings, "primitive_box_fallback_replace_sparse_bounding_box", False)):
            if method_used == "bounding_box":
                reasons.append("engine_completed_with_sparse_bounding_box")

        if not reasons:
            return None
        return "; ".join(reasons)

    @staticmethod
    def _remove_non_canonical_model_variants(project_id: str, output_dir: Path, canonical_path: Path) -> None:
        candidates = [
            output_dir / f"{project_id}_model.glb",
            output_dir / f"{project_id}_model.obj",
        ]
        for path in candidates:
            if path == canonical_path:
                continue
            if path.exists() and path.is_file():
                path.unlink(missing_ok=True)

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
                primary_context = self._extract_processing_error_context(primary_exc)
                fallback_context = self._extract_processing_error_context(fallback_exc)
                failure_stage = (
                    str(fallback_context.get("current_stage") or "").strip()
                    or str(primary_context.get("current_stage") or "").strip()
                    or "fallback_failed"
                )
                fallback_retryable = fallback_context.get("retryable")
                primary_retryable = primary_context.get("retryable")
                retryable = (
                    bool(fallback_retryable)
                    if isinstance(fallback_retryable, bool)
                    else bool(primary_retryable) if isinstance(primary_retryable, bool)
                    else True
                )
                metadata = self._merge_metadata(
                    {
                        "current_stage": failure_stage,
                        "reason_code": "primary_and_fallback_failed",
                        "status_message": "Fallo del motor primario y del motor de respaldo.",
                        "primary_error": {
                            "engine": self._engine.name,
                            "message": str(primary_exc),
                            "reason_code": primary_context.get("reason_code"),
                            "current_stage": primary_context.get("current_stage"),
                        },
                        "fallback_error": {
                            "engine": self._fallback_engine.name if self._fallback_engine is not None else None,
                            "message": str(fallback_exc),
                            "reason_code": fallback_context.get("reason_code"),
                            "current_stage": fallback_context.get("current_stage"),
                        },
                    },
                    fallback_context,
                )
                raise ProcessingError(
                    "La reconstruccion fallo con COLMAP y tambien con el motor de respaldo. "
                    f"Error primario: {primary_exc}. Error fallback: {fallback_exc}.",
                    reason_code="primary_and_fallback_failed",
                    current_stage=failure_stage,
                    metadata=metadata,
                    allow_fallback=False,
                    retryable=retryable,
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
            metrics["image_count_received"] = image_count

        return {
            "engine": self._engine.name,
            "engine_requested": self._requested_engine_mode,
            "profile": str(getattr(self.settings, "profile", "balanced")).strip().lower() or "balanced",
            "requested_output_format": output_format.value,
            "current_stage": "queued",
            "workflow_stage": "queued",
            "stage_status": "queued",
            "progress": 0.0,
            "status_message": "Procesamiento en cola.",
            "stage_sequence": [
                "input_validation",
                "input_validation_failed",
                "input_selection",
                "input_selection_failed",
                "input_object_segmentation",
                "input_object_segmentation_failed",
                "preprocessing",
                "reconstructing",
                "texturing",
                "fallback_completed",
                "starting",
                "feature_extractor",
                "exhaustive_matcher",
                "mapper",
                "mapper_failed_insufficient_registered_images",
                "dense_reconstruction_unavailable",
                "model_converter_txt",
                "model_converter_sparse_ply",
                "sparse_mesh_fallback",
                "delaunay_mesher_sparse",
                "quality_gate_incoherent_output",
                "primitive_box_fallback",
                "image_undistorter",
                "patch_match_stereo",
                "stereo_fusion",
                "poisson_mesher",
                "export",
                "completed_with_fallback",
                "completed",
                "failed",
            ],
            "metrics": metrics,
            "fallback": {
                "used": False,
                "from_engine": None,
                "reason": None,
            },
        }

    def _validate_input_images(
        self,
        project_id: str,
        images_dir: Path,
        output_dir: Path,
        progress_state: dict[str, Any],
        stage_tracker: dict[str, Any] | None = None,
    ) -> tuple[Path, int]:
        if not self._input_image_validator.validation_settings.enabled:
            metrics = dict(progress_state.get("metrics") or {})
            metrics["image_count_received"] = self._count_images(images_dir)
            metrics["image_count_accepted"] = metrics["image_count_received"]
            progress_state["metrics"] = metrics
            progress_state["current_stage"] = "input_validation"
            progress_state["workflow_stage"] = "preprocessing"
            progress_state["status_message"] = "Validacion de imagenes deshabilitada por configuracion."
            progress_state["stage_status"] = "skipped"
            self._update_stage_tracker(stage_tracker, progress_state, event_type="stage_skipped")
            self._safe_update_processing_metadata(project_id, progress_state)
            return images_dir, int(metrics["image_count_received"])

        progress_state["current_stage"] = "input_validation"
        progress_state["workflow_stage"] = "preprocessing"
        progress_state["stage_status"] = "running"
        progress_state["progress"] = 0.06
        progress_state["status_message"] = "Validando imagenes antes de reconstruir."
        self._update_stage_tracker(stage_tracker, progress_state, event_type="stage_update")
        self._safe_update_processing_metadata(project_id, progress_state)

        validation_dir = output_dir / "validation"
        validation_result = self._input_image_validator.validate_batch(
            images_dir,
            report_dir=validation_dir,
        )
        staged_accepted_images = self._input_image_validator.stage_accepted_images(
            validation_result.accepted_images,
            validation_dir / "accepted_images",
        )
        progress_state["current_stage"] = "input_selection"
        progress_state["workflow_stage"] = "preprocessing"
        progress_state["stage_status"] = "running"
        progress_state["progress"] = 0.11
        progress_state["status_message"] = "Seleccionando subconjunto optimo de imagenes."
        self._update_stage_tracker(stage_tracker, progress_state, event_type="stage_update")
        self._safe_update_processing_metadata(project_id, progress_state)

        selection_result = self._input_image_selector.select_images(
            validation_result.summary,
            validation_result.accepted_images,
            report_dir=validation_dir,
        )
        staged_selected_images = self._input_image_selector.stage_selected_images(
            selection_result.selected_images,
            validation_dir / "selected_images",
        )

        summary = dict(validation_result.summary)
        summary["staged_images_dir"] = str(validation_dir / "accepted_images")
        summary["staged_image_count"] = len(staged_accepted_images)

        selection_summary = dict(selection_result.summary)
        selection_summary["staged_images_dir"] = str(validation_dir / "selected_images")
        selection_summary["staged_image_count"] = len(staged_selected_images)

        metrics = dict(progress_state.get("metrics") or {})
        metrics["image_count_received"] = int(summary.get("total_images", 0))
        metrics["image_count_accepted"] = int(summary.get("accepted_images", len(staged_accepted_images)))
        metrics["image_count_rejected"] = int(summary.get("rejected_images", 0))
        metrics["image_count_warned"] = int(summary.get("warning_images", 0))
        metrics["image_count_selected"] = int(selection_summary.get("selected_images", len(staged_selected_images)))
        metrics["image_count_discarded_selection"] = int(selection_summary.get("discarded_images", 0))

        validation_metadata: dict[str, Any] = {
            "input_validation": summary,
            "input_selection": selection_summary,
            "metrics": metrics,
            "artifacts": {
                "input_validation_report": str(validation_result.report_path) if validation_result.report_path else None,
                "validated_images_dir": str(validation_dir / "accepted_images"),
                "input_selection_report": str(selection_result.report_path) if selection_result.report_path else None,
                "selected_images_dir": str(validation_dir / "selected_images"),
            },
        }

        if not validation_result.allow_processing:
            blocking_reasons = list(summary.get("blocking_reasons") or [])
            reason_message = (
                "La validacion de imagenes bloqueo el procesamiento. "
                f"Motivos: {', '.join(blocking_reasons) if blocking_reasons else 'sin detalle'}."
            )
            error_metadata = self._merge_metadata(
                self._merge_metadata(progress_state, validation_metadata),
                {
                    "current_stage": "input_validation_failed",
                    "stage_status": "failed",
                    "reason_code": "input_validation_failed",
                    "status_message": reason_message,
                },
            )
            raise ProcessingError(
                reason_message,
                reason_code="input_validation_failed",
                current_stage="input_validation_failed",
                metadata=error_metadata,
                allow_fallback=False,
                retryable=False,
            )

        if not selection_result.allow_processing:
            blocking_reasons = list(selection_summary.get("blocking_reasons") or [])
            reason_message = (
                "La seleccion de imagenes bloqueo el procesamiento. "
                f"Motivos: {', '.join(blocking_reasons) if blocking_reasons else 'sin detalle'}."
            )
            error_metadata = self._merge_metadata(
                self._merge_metadata(progress_state, validation_metadata),
                {
                    "current_stage": "input_selection_failed",
                    "stage_status": "failed",
                    "reason_code": "input_selection_failed",
                    "status_message": reason_message,
                },
            )
            raise ProcessingError(
                reason_message,
                reason_code="input_selection_failed",
                current_stage="input_selection_failed",
                metadata=error_metadata,
                allow_fallback=False,
                retryable=False,
            )

        progress_state["current_stage"] = "input_object_segmentation"
        progress_state["workflow_stage"] = "preprocessing"
        progress_state["stage_status"] = "running"
        progress_state["progress"] = 0.14
        progress_state["status_message"] = "Segmentando automaticamente el objeto principal."
        self._update_stage_tracker(stage_tracker, progress_state, event_type="stage_update")
        self._safe_update_processing_metadata(project_id, self._merge_metadata(progress_state, validation_metadata))

        segmentation_dir = output_dir / "segmentation"
        segmentation_result = self._input_object_segmenter.segment_images(
            staged_selected_images,
            report_dir=segmentation_dir,
        )
        segmentation_summary = dict(segmentation_result.summary)
        metrics["image_count_segmentation_candidates"] = int(
            segmentation_summary.get("candidate_images", len(staged_selected_images))
        )
        metrics["image_count_segmented"] = int(segmentation_summary.get("segmented_images", 0))
        metrics["image_count_segmentation_fallback"] = int(
            segmentation_summary.get("fallback_original_images", len(staged_selected_images))
        )

        artifacts = dict(validation_metadata.get("artifacts") or {})
        artifacts["input_object_segmentation_report"] = (
            str(segmentation_result.report_path)
            if segmentation_result.report_path is not None
            else None
        )
        artifacts["segmentation_report"] = artifacts.get("input_object_segmentation_report")
        artifacts["segmented_images_dir"] = (
            str(segmentation_result.processed_images_dir)
            if segmentation_result.processed_images_dir is not None
            else str(segmentation_dir / "segmented_images")
        )
        artifacts["segmentation_masks_dir"] = (
            str(segmentation_result.masks_dir)
            if segmentation_result.masks_dir is not None
            else None
        )

        validation_metadata["input_object_segmentation"] = segmentation_summary
        validation_metadata["metrics"] = metrics
        validation_metadata["artifacts"] = artifacts

        merged_state = self._merge_metadata(progress_state, validation_metadata)
        segmented_dir = (
            segmentation_result.processed_images_dir
            if segmentation_result.processed_images_dir is not None
            else validation_dir / "selected_images"
        )
        segmented_count = len(segmentation_result.processed_images) or len(staged_selected_images)
        staged_output_count = self._count_images(segmented_dir) if segmented_dir.exists() else 0
        extreme_failure_reasons: list[str] = []
        if staged_output_count <= 0:
            extreme_failure_reasons.append("no_staged_output_images")
        if segmented_count <= 0:
            extreme_failure_reasons.append("no_processed_images")

        policy_decision = {
            "mode": "assistive_non_blocking",
            "segmenter_allow_processing": bool(segmentation_result.allow_processing),
            "blocking_reasons": list(segmentation_summary.get("blocking_reasons") or []),
            "extreme_failure_detected": bool(extreme_failure_reasons),
            "extreme_failure_reasons": extreme_failure_reasons,
            "staged_output_images": staged_output_count,
            "input_selected_images": len(staged_selected_images),
            "fallback_to_original_enabled": True,
            "processing_blocked": bool(extreme_failure_reasons),
        }
        segmentation_summary["policy_decision"] = policy_decision
        validation_metadata["input_object_segmentation"] = segmentation_summary
        merged_state = self._merge_metadata(progress_state, validation_metadata)

        if not segmentation_result.allow_processing and not extreme_failure_reasons:
            merged_state["status_message"] = (
                "Segmentacion con confianza parcial; se conserva fallback por imagen para mantener estabilidad."
            )

        if extreme_failure_reasons:
            blocking_reasons = list(segmentation_summary.get("blocking_reasons") or [])
            reason_message = (
                "La segmentacion automatica no produjo salida utilizable para continuar. "
                "Motivos: "
                f"{', '.join(blocking_reasons + extreme_failure_reasons) if (blocking_reasons or extreme_failure_reasons) else 'sin detalle'}."
            )
            error_metadata = self._merge_metadata(
                merged_state,
                {
                    "current_stage": "input_object_segmentation_failed",
                    "stage_status": "failed",
                    "reason_code": "input_object_segmentation_failed",
                    "status_message": reason_message,
                },
            )
            raise ProcessingError(
                reason_message,
                reason_code="input_object_segmentation_failed",
                current_stage="input_object_segmentation_failed",
                metadata=error_metadata,
                allow_fallback=False,
                retryable=False,
            )

        merged_state["progress"] = 0.15
        merged_state["stage_status"] = "completed"
        merged_state["status_message"] = "Validacion, seleccion y segmentacion completadas."
        self._update_stage_tracker(stage_tracker, merged_state, event_type="stage_completed")
        self._safe_update_processing_metadata(project_id, merged_state)
        progress_state.clear()
        progress_state.update(merged_state)

        return segmented_dir, segmented_count

    def _build_progress_callback(
        self,
        project_id: str,
        initial_state: dict[str, Any],
        stage_tracker: dict[str, Any] | None = None,
    ) -> ReconstructionProgressCallback:
        state = dict(initial_state)

        def callback(update: dict[str, Any]) -> None:
            merged = self._merge_metadata(state, update)
            current_stage = str(merged.get("current_stage") or "").strip().lower()
            workflow_stage = self._derive_workflow_stage(current_stage)
            if workflow_stage is not None:
                merged["workflow_stage"] = workflow_stage
            progress_value = merged.get("progress")
            if isinstance(progress_value, (int, float)):
                merged["progress"] = max(0.0, min(1.0, float(progress_value)))
            stage_status = str(merged.get("stage_status") or "").strip().lower()
            if not stage_status:
                if current_stage in {"completed", "completed_with_fallback"}:
                    merged["stage_status"] = "completed"
                else:
                    merged["stage_status"] = "running"

            event_type = "stage_update"
            if str(merged.get("stage_status") or "").lower() == "completed":
                event_type = "stage_completed"
            self._update_stage_tracker(stage_tracker, merged, event_type=event_type)
            self._safe_update_processing_metadata(project_id, merged)
            state.clear()
            state.update(merged)

        return callback

    @staticmethod
    def _derive_workflow_stage(current_stage: str) -> str | None:
        if current_stage in {
            "input_validation",
            "input_selection",
            "input_object_segmentation",
            "input_validation_failed",
            "input_selection_failed",
            "input_object_segmentation_failed",
        }:
            return "preprocessing"
        if current_stage in {
            "starting",
            "feature_extractor",
            "exhaustive_matcher",
            "mapper",
            "mapper_failed_insufficient_registered_images",
            "dense_reconstruction_unavailable",
            "model_converter_txt",
            "model_converter_sparse_ply",
            "sparse_mesh_fallback",
            "delaunay_mesher_sparse",
            "image_undistorter",
            "patch_match_stereo",
            "stereo_fusion",
            "poisson_mesher",
            "export",
        }:
            return "reconstructing"
        if current_stage in {"primitive_box_fallback", "completed_with_fallback"}:
            return "fallback_completed"
        if current_stage == "texturing":
            return "texturing"
        if current_stage in {"completed", "failed"}:
            return current_stage
        return None

    def _build_failure_metadata(
        self,
        project_id: str,
        output_format: OutputFormat,
        reason: str,
        elapsed_seconds: float,
        progress_state: dict[str, Any] | None,
        stage_tracker: dict[str, Any] | None = None,
        images_dir: Path | None = None,
        selected_images_dir: Path | None = None,
        engine_images_dir: Path | None = None,
        output_dir: Path | None = None,
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
        reason_code = str(
            (error_context or {}).get("reason_code")
            or existing_metadata.get("reason_code")
            or "processing_failed"
        ).strip() or "processing_failed"
        retryable_flag = (error_context or {}).get("retryable")
        if not isinstance(retryable_flag, bool):
            retryable_flag = self._infer_retryable(reason_code, reason)

        metrics = dict(existing_metadata.get("metrics") or {})
        metrics["total_processing_seconds"] = elapsed_seconds

        if existing_metadata.get("registered_image_count") is not None:
            metrics["registered_image_count"] = existing_metadata.get("registered_image_count")

        existing_metadata["engine"] = existing_metadata.get("engine") or self._engine.name
        existing_metadata["engine_requested"] = existing_metadata.get("engine_requested") or self._requested_engine_mode
        existing_metadata["requested_output_format"] = existing_metadata.get("requested_output_format") or output_format.value
        existing_metadata["reason_code"] = reason_code
        existing_metadata["failed_stage"] = failed_stage
        existing_metadata["current_stage"] = explicit_failure_stage or "failed"
        existing_metadata["stage_status"] = "failed"
        existing_metadata["progress"] = float(existing_metadata.get("progress") or 0.0)
        existing_metadata["status_message"] = (
            existing_metadata.get("status_message")
            if explicit_failure_stage
            else "Procesamiento fallido."
        )
        existing_metadata["failure_reason"] = reason
        existing_metadata["retryable"] = bool(retryable_flag)
        existing_metadata["can_retry"] = bool(retryable_flag)
        existing_metadata["metrics"] = metrics
        existing_metadata["fallback"] = existing_metadata.get("fallback") or {
            "used": False,
            "from_engine": None,
            "reason": None,
        }

        resources = self._collect_failure_resources(
            images_dir=images_dir,
            selected_images_dir=selected_images_dir,
            engine_images_dir=engine_images_dir,
            output_dir=output_dir,
            metadata=existing_metadata,
        )
        error_block = dict(existing_metadata.get("error") or {})
        error_block.update(
            {
                "code": reason_code,
                "message": reason,
                "stage": existing_metadata["current_stage"],
                "retryable": bool(retryable_flag),
            }
        )
        if resources:
            error_block["resources"] = resources
        existing_metadata["error"] = error_block

        existing_metadata["execution_report"] = self._finalize_execution_report(
            stage_tracker=stage_tracker,
            metadata=existing_metadata,
            elapsed_seconds=elapsed_seconds,
            outcome="failed",
            failure_reason=reason,
            reason_code=reason_code,
            retryable=bool(retryable_flag),
        )
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

        retryable = getattr(exc, "retryable", None)
        if isinstance(retryable, bool):
            context["retryable"] = retryable

        metadata = getattr(exc, "metadata", None)
        if isinstance(metadata, dict):
            context = ProcessingService._merge_metadata(context, metadata)

        return context

    def _safe_update_processing_metadata(self, project_id: str, metadata: dict[str, Any]) -> None:
        try:
            self.project_service.update_processing_metadata(project_id, dict(metadata))
        except Exception as exc:
            logger.warning(
                "Unable to persist processing metadata for project %s at stage=%s: %s",
                project_id,
                metadata.get("current_stage"),
                exc,
                exc_info=exc,
            )

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _create_stage_tracker(self, started_at_iso: str) -> dict[str, Any]:
        return {
            "started_at": started_at_iso,
            "finished_at": None,
            "outcome": "running",
            "active_stage": None,
            "stages": {},
            "timeline": [],
        }

    def _update_stage_tracker(
        self,
        stage_tracker: dict[str, Any] | None,
        metadata: dict[str, Any],
        *,
        event_type: str,
    ) -> None:
        if not isinstance(stage_tracker, dict):
            return

        stage = str(metadata.get("current_stage") or "").strip()
        if not stage:
            return

        now_iso = self._utc_now_iso()
        progress_value = metadata.get("progress")
        progress = None
        if isinstance(progress_value, (int, float)):
            progress = round(max(0.0, min(1.0, float(progress_value))), 4)

        message = str(metadata.get("status_message") or "").strip() or None
        stage_status = str(metadata.get("stage_status") or "running").strip().lower()
        if stage_status not in {"queued", "running", "completed", "failed", "skipped"}:
            stage_status = "running"

        timeline_limit = max(10, int(getattr(self.settings, "processing_execution_timeline_limit", 200)))
        timeline = stage_tracker.setdefault("timeline", [])
        if isinstance(timeline, list):
            timeline.append(
                {
                    "timestamp": now_iso,
                    "event_type": event_type,
                    "stage": stage,
                    "stage_status": stage_status,
                    "progress": progress,
                    "message": message,
                }
            )
            if len(timeline) > timeline_limit:
                del timeline[: len(timeline) - timeline_limit]

        stages = stage_tracker.setdefault("stages", {})
        if not isinstance(stages, dict):
            return

        stage_info = stages.get(stage)
        if not isinstance(stage_info, dict):
            stage_info = {
                "stage": stage,
                "status": stage_status,
                "first_seen_at": now_iso,
                "last_update_at": now_iso,
                "completed_at": None,
                "latest_progress": progress,
                "latest_message": message,
                "event_count": 0,
            }
            stages[stage] = stage_info

        previous_stage = str(stage_tracker.get("active_stage") or "").strip()
        if previous_stage and previous_stage != stage:
            previous_info = stages.get(previous_stage)
            if isinstance(previous_info, dict) and previous_info.get("status") in {"queued", "running"}:
                previous_info["status"] = "completed"
                previous_info["completed_at"] = now_iso

        stage_info["status"] = stage_status
        stage_info["last_update_at"] = now_iso
        stage_info["latest_progress"] = progress
        stage_info["latest_message"] = message
        stage_info["event_count"] = int(stage_info.get("event_count") or 0) + 1
        if stage_status in {"completed", "failed", "skipped"}:
            stage_info["completed_at"] = now_iso
        stage_tracker["active_stage"] = stage

    def _finalize_execution_report(
        self,
        *,
        stage_tracker: dict[str, Any] | None,
        metadata: dict[str, Any],
        elapsed_seconds: float,
        outcome: str,
        failure_reason: str | None = None,
        reason_code: str | None = None,
        retryable: bool | None = None,
    ) -> dict[str, Any]:
        started_at = str((stage_tracker or {}).get("started_at") or self._utc_now_iso())
        finished_at = self._utc_now_iso()
        current_stage = str(metadata.get("current_stage") or "").strip() or None
        failed_stage = str(metadata.get("failed_stage") or "").strip() or None

        raw_stages = (stage_tracker or {}).get("stages") or {}
        stage_items: list[dict[str, Any]] = []
        if isinstance(raw_stages, dict):
            for item in raw_stages.values():
                if not isinstance(item, dict):
                    continue
                normalized = {
                    "stage": str(item.get("stage") or "").strip(),
                    "status": str(item.get("status") or "running").strip().lower(),
                    "first_seen_at": item.get("first_seen_at"),
                    "last_update_at": item.get("last_update_at"),
                    "completed_at": item.get("completed_at"),
                    "latest_progress": item.get("latest_progress"),
                    "latest_message": item.get("latest_message"),
                    "event_count": int(item.get("event_count") or 0),
                }
                if normalized["stage"]:
                    stage_items.append(normalized)
        stage_items.sort(key=lambda item: str(item.get("first_seen_at") or ""))

        if current_stage and not any(item["stage"] == current_stage for item in stage_items):
            stage_items.append(
                {
                    "stage": current_stage,
                    "status": "completed" if outcome == "completed" else "failed",
                    "first_seen_at": finished_at,
                    "last_update_at": finished_at,
                    "completed_at": finished_at,
                    "latest_progress": metadata.get("progress"),
                    "latest_message": metadata.get("status_message"),
                    "event_count": 1,
                }
            )

        if outcome == "completed":
            for item in stage_items:
                if item["stage"] == current_stage:
                    item["status"] = "completed"
                    item["completed_at"] = finished_at
                    break
        elif failed_stage:
            for item in stage_items:
                if item["stage"] == failed_stage:
                    item["status"] = "failed"
                    item["completed_at"] = finished_at
                    break

        stage_sequence = [
            str(value).strip()
            for value in list(metadata.get("stage_sequence") or [])
            if str(value).strip()
        ]
        observed_names = {item["stage"] for item in stage_items}
        pending_stages = [name for name in stage_sequence if name not in observed_names]

        timeline = (stage_tracker or {}).get("timeline") or []
        if not isinstance(timeline, list):
            timeline = []

        report = {
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": round(float(elapsed_seconds), 3),
            "outcome": outcome,
            "current_stage": current_stage,
            "failed_stage": failed_stage,
            "reason_code": reason_code or metadata.get("reason_code"),
            "retryable": retryable if outcome == "failed" else None,
            "failure_reason": failure_reason if outcome == "failed" else None,
            "stages": stage_items,
            "pending_stages": pending_stages,
            "timeline": timeline,
        }
        return report

    def _write_quality_report(
        self,
        *,
        project_id: str,
        output_dir: Path,
        project_status: ProjectStatus,
        metadata: dict[str, Any],
        model_path: Path | None,
    ) -> dict[str, Any]:
        pipeline_dir = output_dir / "pipeline"
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        classification = self._classify_quality_result(project_status=project_status, metadata=metadata)
        quality_metrics = self._build_quality_metrics(metadata=metadata, model_path=model_path)
        geometry_source = self._infer_geometry_source(metadata=metadata, quality_metrics=quality_metrics)
        texture_source = self._infer_texture_source(metadata=metadata)
        sparse_density_level = self._compute_sparse_density_level(int(quality_metrics.get("points_3d_count") or 0))
        sparse_quality_score = self._compute_sparse_quality_score(
            points_3d=int(quality_metrics.get("points_3d_count") or 0),
            cameras_reconstructed=int(quality_metrics.get("cameras_reconstructed") or 0),
        )
        visualization_type = self._infer_visualization_type(metadata=metadata, geometry_source=geometry_source)
        mesh_quality_score = self._compute_mesh_quality_score(
            quality_metrics=quality_metrics,
            geometry_source=geometry_source,
        )
        texture_quality_score = self._compute_texture_quality_score(
            quality_metrics=quality_metrics,
            texture_source=texture_source,
        )
        final_visual_score = self._compute_final_visual_score(
            mesh_quality_score=mesh_quality_score,
            texture_quality_score=texture_quality_score,
            classification=classification,
        )
        final_texture_quality = self._compute_final_texture_quality(
            texture_score=texture_quality_score,
            texture_source=texture_source,
        )
        presentable_state = self._compute_presentable_state(
            geometry_source=geometry_source,
            texture_source=texture_source,
            texture_score=texture_quality_score,
            classification=classification,
        )
        is_academic_fallback = classification == "fallback_completed"
        warnings = sorted({str(item) for item in (metadata.get("warnings") or []) if str(item).strip()})

        limitations_by_classification = {
            "success_real": [
                "La calidad depende del dataset de captura y puede variar entre sesiones.",
            ],
            "success_approx_surface": [
                "La malla proviene de reconstruccion de superficie sobre nube sparse, no de dense real.",
                "Puede perder detalle fino y cavidades complejas.",
            ],
            "success_sparse_only": [
                "No hay malla densa final; la geometria proviene de sparse/fallback de mesh.",
                "Puede requerir recaptura para mejorar detalle superficial.",
            ],
            "fallback_completed": [
                "No representa una reconstruccion SfM real completa.",
                "Debe reportarse como salida academica de recuperacion controlada.",
            ],
            "failed": [
                "No se obtuvo un modelo final utilizable para visualizacion.",
            ],
        }
        academic_interpretation_by_classification = {
            "success_real": (
                "La ejecucion produjo evidencia de reconstruccion real con COLMAP y puede usarse como resultado "
                "principal en la sustentacion."
            ),
            "success_approx_surface": (
                "La ejecucion produjo una superficie aproximada defendible basada en SfM sparse real, "
                "adecuada para visualizacion academica sin declararla como malla densa real."
            ),
            "success_sparse_only": (
                "La ejecucion produjo evidencia tecnica valida de SfM sparse, pero sin reconstruccion densa completa. "
                "Es evidencia intermedia defendible con limitaciones."
            ),
            "fallback_completed": (
                "La ejecucion preservo continuidad academica mediante fallback controlado, demostrando robustez del "
                "pipeline ante fallos de reconstruccion real."
            ),
            "failed": (
                "La ejecucion no alcanzo una salida usable. El valor academico principal es el diagnostico tecnico y "
                "las recomendaciones de recaptura/configuracion."
            ),
        }
        next_action_by_classification = {
            "success_real": "Usar este resultado como evidencia principal y anexar reportes JSON al informe.",
            "success_approx_surface": "Reportar la malla como aproximada y planear recaptura para intentar dense real.",
            "success_sparse_only": "Mejorar captura (overlap, textura, iluminacion) para intentar reconstruccion densa.",
            "fallback_completed": "Usar fallback como evidencia de robustez y repetir captura para buscar SfM real.",
            "failed": "Revisar logs de COLMAP, validar setup y repetir captura con mejor calidad de imagen.",
        }

        report_payload = {
            "project_id": project_id,
            "profile": str(metadata.get("profile") or getattr(self.settings, "profile", "balanced")).strip().lower() or "balanced",
            "engine": str(metadata.get("engine") or self._engine.name),
            "status": str(project_status.value if isinstance(project_status, ProjectStatus) else project_status),
            "quality_classification": classification,
            "geometry_source": geometry_source,
            "texture_source": texture_source,
            "visualization_type": visualization_type,
            "is_real_sfm": geometry_source in {"colmap_dense", "colmap_sparse_point_cloud", "colmap_sparse", "surface_from_sparse", "geometric_prior"},
            "is_dense_mesh": geometry_source == "colmap_dense",
            "is_textured_mesh": texture_source in {"real_projection", "best_image_projection"},
            "cameras_reconstructed": quality_metrics.get("cameras_reconstructed", 0),
            "images_registered": quality_metrics.get("images_registered", 0),
            "points3D_count": quality_metrics.get("points_3d_count", 0),
            "dense_available": quality_metrics.get("dense_exists", False),
            "sparse_quality_score": sparse_quality_score,
            "sparse_density_level": sparse_density_level,
            "mesh_quality_score": mesh_quality_score,
            "texture_quality_score": texture_quality_score,
            "is_academic_fallback": is_academic_fallback,
            "metrics": quality_metrics,
            "warnings": warnings,
            "limitations": limitations_by_classification[classification],
            "academic_interpretation": academic_interpretation_by_classification[classification],
            "recommended_next_action": next_action_by_classification[classification],
            "recommended_capture_fix": self._recommended_capture_fix(sparse_density_level),
            "surface_reconstruction": metadata.get("surface_reconstruction"),
            "segmentation_report": metadata.get("input_object_segmentation"),
            "geometric_prior_report": metadata.get("geometric_prior"),
            "texture_report": metadata.get("texture_projection"),
            "surface_attempted": bool(metadata.get("surface_attempted")),
            "surface_success": bool(metadata.get("surface_success")),
            "surface_failure_reason": metadata.get("surface_failure_reason"),
            "final_visual_is_mesh": geometry_source in {"colmap_dense", "surface_from_sparse", "geometric_prior"},
            "final_visual_is_point_cloud": geometry_source == "colmap_sparse_point_cloud",
            "min_faces_required_for_surface": 500,
            "visual_quality_level": self._compute_visual_quality_level(
                geometry_source=geometry_source,
                mesh_faces=int(quality_metrics.get("mesh_face_count") or 0),
                points_3d=int(quality_metrics.get("points_3d_count") or 0),
            ),
            "real_geometry_metrics": {
                "dense_faces_count_real": int(quality_metrics.get("dense_faces_count_real") or 0),
                "dense_vertices_count_real": int(quality_metrics.get("dense_vertices_count_real") or 0),
                "surface_faces_count_real": int(quality_metrics.get("surface_faces_count_real") or 0),
                "surface_vertices_count_real": int(quality_metrics.get("surface_vertices_count_real") or 0),
                "real_mesh_available": bool(
                    int(quality_metrics.get("dense_faces_count_real") or 0) >= 500
                    or int(quality_metrics.get("surface_faces_count_real") or 0) >= 500
                ),
                "real_mesh_source": (
                    "colmap_dense"
                    if int(quality_metrics.get("dense_faces_count_real") or 0) >= 500
                    else "surface_from_sparse"
                    if int(quality_metrics.get("surface_faces_count_real") or 0) >= 500
                    else "none"
                ),
            },
            "visualization_metrics": {
                "visual_faces_count": int(quality_metrics.get("visual_faces_count") or 0),
                "visual_vertices_count": int(quality_metrics.get("visual_vertices_count") or 0),
                "visual_geometry_type": str(quality_metrics.get("visual_geometry_type") or visualization_type),
                "visual_faces_are_reconstruction": not bool(quality_metrics.get("mesh_face_count_is_visual_only")),
            },
            "final_model_strategy": {
                "geometry_source": geometry_source,
                "texture_source": texture_source,
                "classification": classification,
            },
            "final_model_texture_quality": final_texture_quality,
            "final_model_visual_score": final_visual_score,
            "final_model_is_presentable": presentable_state,
            "final_model_limitations": limitations_by_classification[classification],
            "recommended_next_capture_action": next_action_by_classification[classification],
        }
        report_path = write_json(pipeline_dir / "quality_report.json", report_payload)

        artifacts = dict(metadata.get("artifacts") or {})
        artifacts["quality_report"] = str(report_path)
        metadata["artifacts"] = artifacts
        metadata["quality_report"] = report_payload
        metadata["quality_classification"] = classification
        metadata["geometry_source"] = geometry_source
        metadata["visualization_type"] = visualization_type
        metadata["is_real_sfm"] = report_payload["is_real_sfm"]
        metadata["is_dense_mesh"] = report_payload["is_dense_mesh"]
        metadata["is_textured_mesh"] = report_payload["is_textured_mesh"]

        if (
            classification == "fallback_completed"
            and report_payload["is_real_sfm"]
            and geometry_source == "colmap_sparse_point_cloud"
        ):
            fallback_payload = {
                "project_id": project_id,
                "fallback_type": "sparse_real_academic_fallback",
                "reason_message": (
                    "Se obtuvo reconstruccion sparse real, pero la densidad de puntos no fue suficiente para "
                    "generar una malla densa utilizable."
                ),
                "cameras_reconstructed": report_payload["cameras_reconstructed"],
                "points3D_count": report_payload["points3D_count"],
                "mesh_faces_count": int(quality_metrics.get("mesh_face_count") or 0),
                "sparse_density_level": sparse_density_level,
                "visualization_type": visualization_type,
            }
            fallback_path = write_json(pipeline_dir / "fallback_report.json", fallback_payload)
            artifacts["fallback_report"] = str(fallback_path)
            metadata["artifacts"] = artifacts
            metadata["fallback_report"] = fallback_payload
        return metadata

    def _classify_quality_result(
        self,
        *,
        project_status: ProjectStatus,
        metadata: dict[str, Any],
    ) -> str:
        if project_status == ProjectStatus.FAILED:
            return "failed"

        metrics = metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {}
        reconstruction_type = str(metadata.get("reconstruction_type") or "").strip().lower()
        cameras = self._safe_int(
            metadata.get("registered_image_count")
            or metadata.get("camera_count")
            or metrics.get("reconstructed_camera_count")
        ) or 0
        images_registered = self._safe_int(metadata.get("registered_image_count")) or 0
        points = self._safe_int(
            metadata.get("point_count")
            or metrics.get("point_3d_count")
            or metrics.get("sparse_point_cloud_count")
        ) or 0
        mesh_faces = self._safe_int(metrics.get("mesh_face_count") or metadata.get("mesh_face_count")) or 0
        dense_faces_real = self._safe_int(metrics.get("dense_faces_count_real")) or 0
        dense_vertices_real = self._safe_int(metrics.get("dense_vertices_count_real")) or 0
        surface_faces_real = self._safe_int(metrics.get("surface_faces_count_real")) or 0
        surface_vertices_real = self._safe_int(metrics.get("surface_vertices_count_real")) or 0
        mesh_face_count_is_visual_only = bool(metrics.get("mesh_face_count_is_visual_only"))
        surface_success = bool(metadata.get("surface_success"))
        surface_attempted = bool(metadata.get("surface_attempted"))
        artifacts = metadata.get("artifacts") if isinstance(metadata.get("artifacts"), dict) else {}
        dense_exists = any(
            self._path_exists(artifacts.get(key))
            for key in ("fused_ply_path", "poisson_mesh_ply")
        )
        sparse_exists = any(
            self._path_exists(artifacts.get(key))
            for key in ("sparse_txt_dir", "raw_sparse_ply", "sparse_delaunay_mesh_ply")
        )

        if reconstruction_type == "approximate_box_primitive_fallback":
            return "fallback_completed"
        if (
            reconstruction_type == "dense_photogrammetry_mesh"
            and dense_exists
            and dense_faces_real >= 500
            and dense_vertices_real >= 100
            and not mesh_face_count_is_visual_only
        ):
            return "success_real"
        if (
            reconstruction_type == "sparse_surface_reconstruction"
            and surface_success
            and surface_faces_real >= 500
            and surface_vertices_real >= 100
        ):
            return "success_approx_surface"
        if reconstruction_type == "sparse_geometric_prior_reconstruction":
            if surface_faces_real >= 500 and surface_vertices_real >= 100:
                return "success_approx_surface"
            if mesh_faces >= 500:
                return "success_approx_surface"
            return "fallback_completed"
        if reconstruction_type == "sparse_surface_reconstruction" and surface_attempted and surface_faces_real < 500:
            return "fallback_completed"
        visualization_type = self._infer_visualization_type(metadata=metadata, geometry_source=self._infer_geometry_source(metadata=metadata, quality_metrics={"mesh_face_count": mesh_faces, "dense_exists": dense_exists, "sparse_exists": sparse_exists}))
        if mesh_faces == 0 and (sparse_exists or "point_cloud_fallback" in reconstruction_type):
            if cameras >= 8 and points >= 1500 and visualization_type == "point_spheres":
                return "success_sparse_only"
            return "fallback_completed"
        if (
            sparse_exists
            and cameras >= 8
            and points >= 1500
            and visualization_type in {"point_spheres", "point_cloud"}
            and not surface_success
            and (surface_attempted or "point_cloud_fallback" in reconstruction_type)
        ):
            return "success_sparse_only"
        if sparse_exists and project_status == ProjectStatus.COMPLETED:
            return "fallback_completed"
        return "failed"

    def _build_quality_metrics(
        self,
        *,
        metadata: dict[str, Any],
        model_path: Path | None,
    ) -> dict[str, Any]:
        metrics = metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {}
        artifacts = metadata.get("artifacts") if isinstance(metadata.get("artifacts"), dict) else {}

        image_count_processed = self._safe_int(
            metrics.get("image_count_processed")
            or metrics.get("image_count_preprocessed")
            or metrics.get("image_count_selected")
            or metadata.get("image_count_processed")
        ) or 0
        cameras_reconstructed = self._safe_int(
            metadata.get("registered_image_count")
            or metadata.get("camera_count")
            or metrics.get("reconstructed_camera_count")
        ) or 0
        images_registered = self._safe_int(metadata.get("registered_image_count")) or cameras_reconstructed
        points_3d = self._safe_int(
            metadata.get("point_count")
            or metrics.get("point_3d_count")
            or metrics.get("sparse_point_cloud_count")
        ) or 0
        sparse_exists = any(
            self._path_exists(artifacts.get(key))
            for key in ("sparse_txt_dir", "raw_sparse_ply", "sparse_delaunay_mesh_ply")
        )
        dense_exists = any(
            self._path_exists(artifacts.get(key))
            for key in ("fused_ply_path", "poisson_mesh_ply")
        )
        resolved_model_path = model_path or self._resolve_optional_path(metadata.get("final_model_path") or artifacts.get("model_path"))
        model_size_bytes = resolved_model_path.stat().st_size if resolved_model_path is not None and resolved_model_path.exists() else 0
        fallback_used = bool(
            metadata.get("fallback_used")
            or (metadata.get("fallback") or {}).get("used")
            or (metadata.get("sparse_fallback") or {}).get("used")
        )

        return {
            "image_count_processed": image_count_processed,
            "cameras_reconstructed": cameras_reconstructed,
            "images_registered": images_registered,
            "points_3d_count": points_3d,
            "mesh_face_count": self._safe_int(metrics.get("mesh_face_count") or metadata.get("mesh_face_count")) or 0,
            "dense_faces_count_real": self._safe_int(metrics.get("dense_faces_count_real")) or 0,
            "dense_vertices_count_real": self._safe_int(metrics.get("dense_vertices_count_real")) or 0,
            "surface_faces_count_real": self._safe_int(metrics.get("surface_faces_count_real")) or 0,
            "surface_vertices_count_real": self._safe_int(metrics.get("surface_vertices_count_real")) or 0,
            "visual_faces_count": self._safe_int(metrics.get("visual_faces_count")) or 0,
            "visual_vertices_count": self._safe_int(metrics.get("visual_vertices_count")) or 0,
            "visual_geometry_type": str(metrics.get("visual_geometry_type") or ""),
            "mesh_face_count_is_visual_only": bool(metrics.get("mesh_face_count_is_visual_only")),
            "model_size_bytes": int(model_size_bytes),
            "sparse_exists": sparse_exists,
            "dense_exists": dense_exists,
            "dense_stages_enabled": bool(metadata.get("dense_stages_enabled")),
            "fallback_used": fallback_used,
            "total_processing_seconds": self._safe_float(metrics.get("total_processing_seconds")),
        }

    def _infer_geometry_source(self, *, metadata: dict[str, Any], quality_metrics: dict[str, Any]) -> str:
        reconstruction_type = str(metadata.get("reconstruction_type") or "").strip().lower()
        mesh_faces = int(quality_metrics.get("mesh_face_count") or 0)
        if reconstruction_type == "sparse_geometric_prior_reconstruction":
            return "geometric_prior"
        if reconstruction_type == "approximate_box_primitive_fallback":
            return "primitive_box"
        if reconstruction_type == "dense_photogrammetry_mesh" and bool(quality_metrics.get("dense_exists")) and mesh_faces >= 100:
            return "colmap_dense"
        if reconstruction_type == "sparse_surface_reconstruction":
            return "surface_from_sparse"
        if "point_cloud_fallback" in reconstruction_type:
            return "colmap_sparse_point_cloud"
        if reconstruction_type == "sparse_photogrammetry_mesh_fallback" or bool(quality_metrics.get("sparse_exists")):
            sparse_fallback = metadata.get("sparse_fallback") if isinstance(metadata.get("sparse_fallback"), dict) else {}
            final_method = str(sparse_fallback.get("final_mesh_method") or sparse_fallback.get("mesh_method") or "").strip().lower()
            if final_method == "bounding_box":
                return "fallback"
            return "colmap_sparse"
        return "fallback"

    @staticmethod
    def _compute_sparse_density_level(points_3d: int) -> str:
        if points_3d < 1500:
            return "low"
        if points_3d <= 5000:
            return "medium"
        return "high"

    @staticmethod
    def _compute_sparse_quality_score(*, points_3d: int, cameras_reconstructed: int) -> float:
        points_component = min(0.75, points_3d / 8000.0 * 0.75)
        cameras_component = min(0.25, cameras_reconstructed / 24.0 * 0.25)
        return round(max(0.0, min(1.0, points_component + cameras_component)), 3)

    @staticmethod
    def _recommended_capture_fix(sparse_density_level: str) -> list[str]:
        return [
            "tomar mas fotos",
            "mejorar textura del objeto",
            "mejorar iluminacion",
            "variar altura y angulo",
            "evitar superficies lisas/brillantes",
            "aumentar overlap",
        ]

    @staticmethod
    def _infer_visualization_type(*, metadata: dict[str, Any], geometry_source: str) -> str:
        if geometry_source == "primitive_box":
            return "sparse_bbox"
        if geometry_source == "geometric_prior":
            return "approximated_surface"
        if geometry_source == "colmap_dense":
            return "dense_mesh"
        if geometry_source == "surface_from_sparse":
            return "reconstructed_surface"
        sparse_fallback = metadata.get("sparse_fallback") if isinstance(metadata.get("sparse_fallback"), dict) else {}
        vis = str(sparse_fallback.get("visualization_type") or "").strip().lower()
        if vis in {"point_cloud", "point_spheres", "sparse_bbox", "reconstructed_surface", "approximated_surface"}:
            return vis
        if geometry_source == "colmap_sparse_point_cloud":
            return "point_spheres"
        return "point_cloud"

    @staticmethod
    def _compute_visual_quality_level(*, geometry_source: str, mesh_faces: int, points_3d: int) -> str:
        if geometry_source == "colmap_dense" and mesh_faces >= 500:
            return "high"
        if geometry_source == "surface_from_sparse":
            return "medium" if mesh_faces >= 500 else "low"
        if geometry_source == "colmap_sparse_point_cloud":
            return "medium" if points_3d >= 3000 else "low"
        if mesh_faces < 500:
            return "low"
        return "none"

    @staticmethod
    def _infer_texture_source(*, metadata: dict[str, Any]) -> str:
        texture_projection = metadata.get("texture_projection") if isinstance(metadata.get("texture_projection"), dict) else {}
        if texture_projection:
            source = str(texture_projection.get("texture_source") or "").strip().lower()
            if source:
                return source
        surface = metadata.get("surface_reconstruction") if isinstance(metadata.get("surface_reconstruction"), dict) else {}
        color_strategy = str(surface.get("color_strategy") or "").strip().lower()
        if color_strategy == "vertex_colors_from_colmap":
            return "vertex_colors_from_colmap"
        if color_strategy == "average_image_color":
            return "average_image_color"
        captured_texture = (metadata.get("approximate_geometry_fallback") or {}).get("captured_texture")
        if not isinstance(captured_texture, dict):
            return "none"
        if bool(captured_texture.get("applied")):
            return "best_image_projection"
        return "none"

    @staticmethod
    def _compute_mesh_quality_score(*, quality_metrics: dict[str, Any], geometry_source: str) -> float:
        points = int(quality_metrics.get("points_3d_count") or 0)
        cameras = int(quality_metrics.get("cameras_reconstructed") or 0)
        dense = bool(quality_metrics.get("dense_exists"))
        base = 0.15
        if geometry_source == "colmap_dense" and dense:
            base = 0.85
        elif geometry_source == "colmap_sparse":
            base = 0.55
        elif geometry_source == "colmap_sparse_point_cloud":
            base = 0.5
        elif geometry_source == "geometric_prior":
            base = 0.45
        elif geometry_source in {"primitive_box", "fallback"}:
            base = 0.25
        boost = min(0.15, (points / 5000.0) * 0.1 + (cameras / 40.0) * 0.05)
        return round(max(0.0, min(1.0, base + boost)), 3)

    @staticmethod
    def _compute_texture_quality_score(*, quality_metrics: dict[str, Any], texture_source: str) -> float:
        if texture_source == "real_projection":
            return 0.8
        if texture_source == "best_image_projection":
            return 0.45
        if texture_source == "vertex_colors_from_colmap":
            return 0.55
        if texture_source == "average_image_color":
            return 0.25
        if bool(quality_metrics.get("dense_exists")):
            return 0.3
        return 0.0

    @staticmethod
    def _compute_final_visual_score(*, mesh_quality_score: float, texture_quality_score: float, classification: str) -> float:
        class_bonus = 0.1 if classification == "success_real" else 0.05 if classification == "success_approx_surface" else 0.0
        score = mesh_quality_score * 0.65 + texture_quality_score * 0.35 + class_bonus
        return round(max(0.0, min(1.0, score)), 3)

    @staticmethod
    def _compute_final_texture_quality(*, texture_score: float, texture_source: str) -> str:
        if texture_source in {"best_image_projection", "real_projection"} and texture_score >= 0.45:
            return "good"
        if texture_source in {"vertex_colors_from_colmap", "average_image_color"} and texture_score >= 0.25:
            return "acceptable"
        if texture_score > 0.0:
            return "basic"
        return "none"

    @staticmethod
    def _compute_presentable_state(
        *,
        geometry_source: str,
        texture_source: str,
        texture_score: float,
        classification: str,
    ) -> str:
        geometry_ok = geometry_source in {"colmap_dense", "surface_from_sparse", "geometric_prior"}
        if not geometry_ok:
            return "false"
        if texture_source == "average_image_color":
            return "partial"
        if texture_score >= 0.45 and classification in {"success_real", "success_approx_surface", "fallback_completed"}:
            return "true"
        if texture_score > 0.0:
            return "partial"
        return "false"

    @staticmethod
    def _path_exists(raw_path: object) -> bool:
        if raw_path is None:
            return False
        try:
            return Path(str(raw_path)).exists()
        except Exception:
            return False

    def _write_execution_report_file(
        self,
        project_id: str,
        output_dir: Path,
        metadata: dict[str, Any],
    ) -> None:
        try:
            report_dir = output_dir / "pipeline"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"{project_id}_execution_report.json"
            payload = {
                "project_id": project_id,
                "generated_at": self._utc_now_iso(),
                "current_stage": metadata.get("current_stage"),
                "stage_status": metadata.get("stage_status"),
                "status_message": metadata.get("status_message"),
                "reason_code": metadata.get("reason_code"),
                "error": metadata.get("error"),
                "metrics": metadata.get("metrics"),
                "execution_report": metadata.get("execution_report"),
                "artifacts": metadata.get("artifacts"),
            }
            report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

            artifacts = dict(metadata.get("artifacts") or {})
            artifacts["execution_report"] = str(report_path)
            metadata["artifacts"] = artifacts
            metadata["execution_report_path"] = str(report_path)
        except Exception as exc:
            logger.warning(
                "Unable to write execution report for project %s: %s",
                project_id,
                exc,
                exc_info=exc,
            )

    def _write_technical_evidence_file(
        self,
        *,
        project_id: str,
        output_dir: Path,
        metadata: dict[str, Any],
        project_status: str,
    ) -> None:
        try:
            report_path = self._technical_evidence.write_run_evidence(
                project_id=project_id,
                output_dir=output_dir,
                processing_metadata=metadata,
                project_status=project_status,
            )
            if report_path is None:
                return

            artifacts = dict(metadata.get("artifacts") or {})
            artifacts["technical_evidence_report"] = str(report_path)
            metadata["artifacts"] = artifacts
            metadata["technical_evidence_report_path"] = str(report_path)
        except Exception as exc:
            logger.warning(
                "Unable to write technical evidence report for project %s: %s",
                project_id,
                exc,
                exc_info=exc,
            )

    def _append_reconstruction_history(
        self,
        *,
        project_id: str,
        dataset_path: Path | None,
        metadata: dict[str, Any],
    ) -> None:
        try:
            quality = dict(metadata.get("quality_report") or {})
            metrics = dict(quality.get("metrics") or metadata.get("metrics") or {})
            real_metrics = dict(quality.get("real_geometry_metrics") or {})
            duplicate_ratio = float(
                metrics.get("possible_duplicates_ratio")
                or quality.get("duplicate_ratio")
                or 0.0
            )
            record = {
                "project_id": project_id,
                "dataset_path": str(dataset_path) if dataset_path is not None else None,
                "image_count": int(metrics.get("image_count_processed") or metrics.get("image_count_received") or 0),
                "mesh_readiness_score": float(metrics.get("mesh_readiness_score") or 0.0),
                "angular_coverage_score": float(metrics.get("angular_coverage_score") or 0.0),
                "visual_variety_score": float(metrics.get("visual_variety_score") or 0.0),
                "average_feature_points": float(metrics.get("average_feature_points") or 0.0),
                "high_confidence_ratio": float(metrics.get("high_confidence_ratio") or 0.0),
                "average_sharpness": float(metrics.get("average_sharpness") or 0.0),
                "average_brightness": float(metrics.get("average_brightness") or 0.0),
                "duplicate_ratio": duplicate_ratio,
                "profile": str(quality.get("profile") or getattr(self.settings, "profile", "balanced")),
                "colmap_gpu_used": bool(
                    ((metadata.get("colmap_runtime") or {}).get("gpu_probe") or {}).get("enabled")
                    or getattr(self.settings, "colmap_use_gpu", False)
                ),
                "cameras_reconstructed": int(quality.get("cameras_reconstructed") or 0),
                "points3D_count": int(quality.get("points3D_count") or 0),
                "surface_faces_count_real": int(real_metrics.get("surface_faces_count_real") or 0),
                "dense_faces_count_real": int(real_metrics.get("dense_faces_count_real") or 0),
                "quality_classification": str(quality.get("quality_classification") or "failed"),
                "geometry_source": str(quality.get("geometry_source") or "unknown"),
                "visualization_type": str(quality.get("visualization_type") or "unknown"),
                "total_processing_time": float(metrics.get("total_processing_seconds") or 0.0),
                "final_success_level": to_final_success_level(str(quality.get("quality_classification") or "failed")),
            }
            append_history_record(HISTORY_PATH_DEFAULT, record)
        except Exception as exc:
            logger.warning(
                "Unable to append reconstruction history for project %s: %s",
                project_id,
                exc,
                exc_info=exc,
            )

    def _cleanup_failed_outputs(
        self,
        project_id: str,
        output_dir: Path,
        metadata: dict[str, Any],
    ) -> None:
        if not bool(getattr(self.settings, "processing_cleanup_workspace_on_failure", True)):
            return
        if not output_dir.exists():
            return

        removed_paths: list[str] = []
        removable = [
            output_dir / "workspace",
            output_dir / f"{project_id}_model.glb",
            output_dir / f"{project_id}_model.obj",
            output_dir / f"{project_id}_sparse.ply",
        ]
        for path in removable:
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                    removed_paths.append(str(path))
                elif path.exists():
                    path.unlink(missing_ok=True)
                    removed_paths.append(str(path))
            except Exception as exc:
                logger.warning(
                    "Failed to cleanup path after processing error. project=%s path=%s error=%s",
                    project_id,
                    path,
                    exc,
                )

        if removed_paths:
            metadata["cleanup"] = {
                "performed": True,
                "removed_paths": removed_paths,
                "preserved_paths": [
                    str(output_dir / "logs"),
                    str(output_dir / "validation"),
                    str(output_dir / "pipeline"),
                ],
            }

    @staticmethod
    def _collect_failure_resources(
        *,
        images_dir: Path | None,
        selected_images_dir: Path | None,
        output_dir: Path | None,
        engine_images_dir: Path | None = None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        resources: dict[str, Any] = {}
        if images_dir is not None:
            resources["images_dir"] = str(images_dir)
        if selected_images_dir is not None:
            resources["selected_images_dir"] = str(selected_images_dir)
        if engine_images_dir is not None:
            resources["engine_images_dir"] = str(engine_images_dir)
        if output_dir is not None:
            resources["output_dir"] = str(output_dir)

        failed_command = metadata.get("failed_command")
        if failed_command:
            resources["failed_command"] = failed_command

        logs = metadata.get("logs") if isinstance(metadata.get("logs"), dict) else {}
        if logs.get("directory"):
            resources["logs_dir"] = logs["directory"]
        if logs.get("stdout_path"):
            resources["stdout_log"] = logs["stdout_path"]
        if logs.get("stderr_path"):
            resources["stderr_log"] = logs["stderr_path"]

        artifacts = metadata.get("artifacts") if isinstance(metadata.get("artifacts"), dict) else {}
        for key in (
            "input_validation_report",
            "input_selection_report",
            "input_object_segmentation_report",
            "execution_report",
            "model_path",
            "selected_images_dir",
            "validated_images_dir",
            "segmented_images_dir",
            "segmentation_masks_dir",
            "preprocessing_manifest",
            "preprocessed_images_dir",
            "fallback_report",
        ):
            value = artifacts.get(key)
            if value:
                resources[key] = value
        return resources

    def _write_academic_fallback_report(
        self,
        *,
        project_id: str,
        output_dir: Path,
        metadata: dict[str, Any],
        failure_reason: str,
        error_context: dict[str, Any] | None,
        selected_images_dir: Path,
        model_path: Path,
    ) -> dict[str, Any]:
        pipeline_dir = output_dir / "pipeline"
        reason_code = str((error_context or {}).get("reason_code") or metadata.get("reason_code") or "processing_failed")
        report_payload = {
            "project_id": project_id,
            "reason_code": reason_code,
            "reason_message": failure_reason,
            "profile": str(getattr(self.settings, "profile", "balanced")).strip().lower() or "balanced",
            "engine_attempted": self._engine.name,
            "engine_requested": self._requested_engine_mode,
            "colmap_available": self._safe_engine_available(),
            "fallback_type": "primitive_box_academic_fallback",
            "images_used": [
                str(path)
                for path in sorted(selected_images_dir.iterdir(), key=lambda item: item.name.lower())
                if path.is_file()
            ] if selected_images_dir.exists() else [],
            "limitations": [
                "No representa una reconstruccion fotogrametrica exacta.",
                "La geometria se aproxima con una primitiva tipo caja estimada desde las imagenes.",
                "La textura puede provenir de una vista representativa, no de un unwrap completo multi-vista.",
            ],
            "academic_explanation": (
                "El sistema conserva trazabilidad academica cuando COLMAP no produce una reconstruccion usable. "
                "El fallback entrega un modelo minimo demostrable y registra la causa tecnica para que la sustentacion "
                "distinga entre SfM real y recuperacion geometrica aproximada."
            ),
            "generated_model_path": str(model_path),
        }
        report_path = write_json(pipeline_dir / "fallback_report.json", report_payload)
        colmap_report_path = ColmapReconstructionEngine.write_failure_report(
            project_id=project_id,
            output_dir=output_dir,
            colmap_binary=str(getattr(self._engine, "detected_binary", None) or getattr(self._engine, "colmap_binary", "colmap")),
            profile=str(getattr(self.settings, "profile", "balanced")),
            failure_reason=failure_reason,
            error_context=error_context,
            fallback_used=True,
        )
        artifacts = dict(metadata.get("artifacts") or {})
        artifacts["fallback_report"] = str(report_path)
        artifacts["colmap_report"] = str(colmap_report_path)
        metadata["artifacts"] = artifacts
        metadata["fallback_report"] = report_payload
        metadata["colmap_report_path"] = str(colmap_report_path)
        return metadata

    def _safe_engine_available(self) -> bool:
        try:
            return bool(self._engine.is_available())
        except Exception:
            return False

    @staticmethod
    def _infer_retryable(reason_code: str | None, reason: str | None) -> bool:
        normalized_code = str(reason_code or "").strip().lower()
        non_retryable_codes = {
            "input_validation_failed",
            "input_selection_failed",
            "input_object_segmentation_failed",
            "insufficient_registered_images",
            "dense_reconstruction_unavailable",
            "colmap_unavailable",
            "colmap_binary_not_found",
            "unsupported_output_format",
            "insufficient_input_images",
        }
        if normalized_code in non_retryable_codes:
            return False
        if normalized_code in {"colmap_command_timeout"}:
            return True

        normalized_reason = str(reason or "").strip().lower()
        if "timeout" in normalized_reason or "tiempo de espera" in normalized_reason:
            return True
        if "no esta disponible" in normalized_reason and "colmap" in normalized_reason:
            return False
        if "falta la dependencia" in normalized_reason:
            return False
        return True


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

        forced_model_applied = bool((metadata.get("forced_presentable_model") or {}).get("applied"))
        try:
            trimesh_module = self._import_trimesh_validator()
        except RuntimeError:
            if not forced_model_applied:
                raise
            metrics = dict(metadata.get("metrics") or {})
            metrics["mesh_validation_skipped"] = True
            metrics["mesh_validation_reason"] = "missing_trimesh_dependency"
            metadata["metrics"] = metrics
            metadata["final_model_validation"] = self._merge_metadata(
                metadata.get("final_model_validation") if isinstance(metadata.get("final_model_validation"), dict) else None,
                {
                    "performed": False,
                    "status": "skipped",
                    "reason": "missing_trimesh_dependency",
                    "fallback_check": "file_exists_non_empty",
                    "forced_presentable_model": True,
                },
            )
            logger.warning(
                "Skipping GLB mesh validation because trimesh is unavailable and forced presentable model is applied for project %s. model_path=%s",
                project_id,
                model_path,
            )
            return

        reconstruction_type = str(metadata.get("reconstruction_type") or "").strip().lower()
        allow_point_cloud = reconstruction_type == "sparse_photogrammetry_point_cloud_fallback"
        summary = self._inspect_glb_mesh(trimesh_module, model_path, allow_point_cloud=allow_point_cloud)
        metrics = dict(metadata.get("metrics") or {})
        if allow_point_cloud:
            metrics["visual_vertices_count"] = summary["vertex_count"]
            metrics["visual_faces_count"] = summary["face_count"]
            metrics["mesh_face_count_is_visual_only"] = True
        else:
            metrics["mesh_vertex_count"] = summary["vertex_count"]
            metrics["mesh_face_count"] = summary["face_count"]
            metrics["mesh_face_count_is_visual_only"] = False
        metadata["metrics"] = metrics
        metadata["final_model_validation"] = self._merge_metadata(
            metadata.get("final_model_validation") if isinstance(metadata.get("final_model_validation"), dict) else None,
            {
                "performed": True,
                "status": "passed",
                "method": "trimesh_scene_mesh_count",
                "vertex_count": summary["vertex_count"],
                "face_count": summary["face_count"],
                "point_cloud_only": bool(allow_point_cloud and summary["face_count"] <= 0),
                "forced_presentable_model": forced_model_applied,
            },
        )
        logger.info(
            "Validated final GLB for project %s. vertices=%s faces=%s model_path=%s",
            project_id,
            summary["vertex_count"],
            summary["face_count"],
            model_path,
        )

    def _apply_presentation_postprocess_if_configured(
        self,
        *,
        project_id: str,
        output_dir: Path,
        output_format: OutputFormat,
        result: ReconstructionResult,
        metadata: dict[str, Any],
    ) -> tuple[ReconstructionResult, dict[str, Any]]:
        project_dir = self.storage_service.get_project_dir(project_id)
        decision = self._presentation_postprocess.should_apply(project_dir, project_id)
        if not decision.apply:
            return result, metadata

        try:
            postprocess_result = self._presentation_postprocess.apply(
                project_id=project_id,
                output_dir=output_dir,
                output_format=output_format,
                current_model_path=result.model_path,
                profile=decision.profile,
                profile_path=decision.profile_path,
            )
        except Exception as exc:
            logger.warning(
                "Presentation postprocess failed for project %s. Preserving original model. error=%s",
                project_id,
                exc,
                exc_info=exc,
            )
            failed_details = {
                "applied": False,
                "error": str(exc),
                "profile_path": str(decision.profile_path) if decision.profile_path is not None else None,
            }
            updated_metadata = self._merge_metadata(
                metadata,
                {
                    "presentation_postprocess": failed_details,
                },
            )
            return result, updated_metadata

        updated_metadata = self._merge_metadata(
            metadata,
            {
                "presentation_postprocess": postprocess_result.details,
            },
        )
        updated_metrics = dict(updated_metadata.get("metrics") or {})
        updated_metrics["mesh_vertex_count"] = int(postprocess_result.details.get("result_vertex_count") or 0)
        updated_metrics["mesh_face_count"] = int(postprocess_result.details.get("result_face_count") or 0)
        updated_metadata["metrics"] = updated_metrics
        updated_metadata["method_used"] = str(postprocess_result.details.get("method") or updated_metadata.get("method_used") or "")
        updated_metadata["status_message"] = "Reconstruccion completada con ajuste de presentacion sobre malla sparse."
        updated_metadata = self._standardize_final_metadata(
            project_id,
            updated_metadata,
            postprocess_result.model_path,
            output_format,
        )
        updated_result = ReconstructionResult(
            engine_name=result.engine_name,
            requested_output_format=result.requested_output_format,
            model_path=postprocess_result.model_path,
            metadata=result.metadata,
        )
        logger.info(
            "Presentation postprocess applied for project %s. model_path=%s vertices=%s faces=%s",
            project_id,
            postprocess_result.model_path,
            postprocess_result.details.get("result_vertex_count"),
            postprocess_result.details.get("result_face_count"),
        )
        return updated_result, updated_metadata

    def _apply_forced_presentable_model_if_configured(
        self,
        *,
        project_id: str,
        output_dir: Path,
        output_format: OutputFormat,
        result: ReconstructionResult,
        metadata: dict[str, Any],
    ) -> tuple[ReconstructionResult, dict[str, Any]]:
        if not bool(getattr(self.settings, "force_presentable_model_enabled", False)):
            return result, metadata

        if self._has_captured_texture(metadata):
            preserved_metadata = self._merge_metadata(
                metadata,
                {
                    "forced_presentable_model": {
                        "enabled": True,
                        "applied": False,
                        "reason": "preserved_captured_texture_from_images",
                    }
                },
            )
            logger.info(
                "Skipping forced presentable model for project %s because captured texture from input images is available.",
                project_id,
            )
            return result, preserved_metadata

        glb_source = self._resolve_optional_path(getattr(self.settings, "force_presentable_model_glb", None))
        obj_source = self._resolve_optional_path(getattr(self.settings, "force_presentable_model_obj", None))
        requested_source = glb_source if output_format == OutputFormat.GLB else obj_source
        if requested_source is None:
            logger.warning(
                "Forced presentable model is enabled but source path for output_format=%s is missing. project_id=%s",
                output_format.value,
                project_id,
            )
            updated_metadata = self._merge_metadata(
                metadata,
                {
                    "forced_presentable_model": {
                        "enabled": True,
                        "applied": False,
                        "reason": f"missing_source_for_{output_format.value}",
                    }
                },
            )
            return result, updated_metadata

        if not requested_source.exists() or not requested_source.is_file():
            logger.warning(
                "Forced presentable model source does not exist. project_id=%s path=%s",
                project_id,
                requested_source,
            )
            updated_metadata = self._merge_metadata(
                metadata,
                {
                    "forced_presentable_model": {
                        "enabled": True,
                        "applied": False,
                        "reason": "source_not_found",
                        "source_path": str(requested_source),
                    }
                },
            )
            return result, updated_metadata

        destination_glb = output_dir / f"{project_id}_model.glb"
        destination_obj = output_dir / f"{project_id}_model.obj"
        copied_paths: list[str] = []
        try:
            if glb_source is not None and glb_source.exists() and glb_source.is_file():
                shutil.copyfile(glb_source, destination_glb)
                copied_paths.append(str(destination_glb))
            if obj_source is not None and obj_source.exists() and obj_source.is_file():
                shutil.copyfile(obj_source, destination_obj)
                copied_paths.append(str(destination_obj))
        except Exception as exc:
            logger.warning(
                "Unable to apply forced presentable model. project_id=%s error=%s",
                project_id,
                exc,
                exc_info=exc,
            )
            updated_metadata = self._merge_metadata(
                metadata,
                {
                    "forced_presentable_model": {
                        "enabled": True,
                        "applied": False,
                        "reason": f"copy_failed: {exc}",
                    }
                },
            )
            return result, updated_metadata

        final_model_path = destination_glb if output_format == OutputFormat.GLB else destination_obj
        if not final_model_path.exists() or not final_model_path.is_file():
            logger.warning(
                "Forced presentable model did not produce final artifact. project_id=%s expected=%s",
                project_id,
                final_model_path,
            )
            updated_metadata = self._merge_metadata(
                metadata,
                {
                    "forced_presentable_model": {
                        "enabled": True,
                        "applied": False,
                        "reason": "final_artifact_missing_after_copy",
                    }
                },
            )
            return result, updated_metadata

        forced_metadata = self._merge_metadata(
            metadata,
            {
                "forced_presentable_model": {
                    "enabled": True,
                    "applied": True,
                    "source_glb": str(glb_source) if glb_source is not None else None,
                    "source_obj": str(obj_source) if obj_source is not None else None,
                    "copied_paths": copied_paths,
                    "output_model_path": str(final_model_path),
                },
            },
        )
        forced_metadata["status_message"] = (
            "Reconstruccion completada con modelo canonico de presentacion (caja de medicamento)."
        )
        forced_metadata["method_used"] = "forced_presentable_model"
        forced_metadata = self._standardize_final_metadata(
            project_id,
            forced_metadata,
            final_model_path,
            output_format,
        )
        forced_result = ReconstructionResult(
            engine_name=result.engine_name,
            requested_output_format=result.requested_output_format,
            model_path=final_model_path,
            metadata=result.metadata,
        )
        logger.info(
            "Forced presentable model applied for project %s. output_format=%s final_model_path=%s",
            project_id,
            output_format.value,
            final_model_path,
        )
        return forced_result, forced_metadata

    @staticmethod
    def _has_captured_texture(metadata: dict[str, Any]) -> bool:
        approximate = metadata.get("approximate_geometry_fallback")
        if not isinstance(approximate, dict):
            return False
        captured = approximate.get("captured_texture")
        if not isinstance(captured, dict):
            return False
        return bool(captured.get("applied"))

    @staticmethod
    def _resolve_optional_path(raw_path: object) -> Path | None:
        if raw_path is None:
            return None
        normalized = str(raw_path).strip()
        if not normalized:
            return None
        candidate = Path(normalized)
        if candidate.is_absolute():
            return candidate
        cwd_candidate = Path.cwd() / candidate
        if cwd_candidate.exists():
            return cwd_candidate
        project_root = Path(__file__).resolve().parents[2]
        return project_root / candidate

    @staticmethod
    def _import_trimesh_validator() -> object:
        try:
            return importlib.import_module("trimesh")
        except ImportError as exc:
            raise RuntimeError(
                "No se pudo validar el GLB final porque falta la dependencia 'trimesh'."
            ) from exc

    @classmethod
    def _inspect_glb_mesh(
        cls,
        trimesh_module: object,
        model_path: Path,
        *,
        allow_point_cloud: bool = False,
    ) -> dict[str, int]:
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
            if allow_point_cloud:
                return summary
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
    def _safe_int(value: object) -> int | None:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value: object) -> float | None:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if numeric != numeric:
            return None
        if numeric in {float("inf"), float("-inf")}:
            return None
        return numeric

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
        return sum(1 for path in images_dir.iterdir() if path.is_file())

    def _should_use_fallback(self) -> bool:
        if self._requested_engine_mode != "auto":
            return False
        if self._fallback_engine is None:
            return False
        if self._fallback_engine.name == self._engine.name:
            return False
        return bool(getattr(self.settings, "colmap_fallback_to_mock", False))
