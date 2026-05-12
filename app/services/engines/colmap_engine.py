from __future__ import annotations

import importlib
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from app.algorithms.surface_reconstruction import SurfaceReconstruction
from app.algorithms.geometric_priors import GeometricPriorDetector
from app.algorithms.texture_projection import TextureProjection
from app.core.errors import ProcessingError
from app.models.schemas import OutputFormat
from app.services.engines.base_engine import (
    ReconstructionEngine,
    ReconstructionProgressCallback,
    ReconstructionResult,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


@dataclass(frozen=True)
class SparsePoint:
    x: float
    y: float
    z: float
    r: int
    g: int
    b: int
    error: float


@dataclass(frozen=True)
class ColmapCommandTrace:
    name: str
    command: str
    duration_seconds: float
    return_code: int
    stdout_tail: str
    stderr_tail: str
    stdout_path: str | None = None
    stderr_path: str | None = None


@dataclass(frozen=True)
class MeshArtifactSummary:
    vertex_count: int
    face_count: int

class ColmapReconstructionEngine(ReconstructionEngine):
    name = "colmap"
    is_implemented = True
    _MAX_LOG_TAIL = 4000
    _PROBE_TIMEOUT_SECONDS = 10
    _DENSE_RECONSTRUCTION_UNAVAILABLE_REASON_CODE = "dense_reconstruction_unavailable"
    _DENSE_RECONSTRUCTION_UNAVAILABLE_STAGE = "dense_reconstruction_unavailable"
    _INSUFFICIENT_REGISTERED_IMAGES_REASON_CODE = "insufficient_registered_images"
    _INSUFFICIENT_REGISTERED_IMAGES_STAGE = "mapper_failed_insufficient_registered_images"
    _INSUFFICIENT_REGISTERED_IMAGES_MESSAGE = (
        "COLMAP no logro registrar suficientes imagenes para reconstruir el modelo. "
        "Intenta capturar mas fotos con mejor traslape, buena iluminacion y mas textura visual."
    )
    _MAPPER_INSUFFICIENT_REGISTERED_IMAGES_PATTERN = re.compile(
        r"At least two images must be registered for global bundle-adjustment",
        re.IGNORECASE,
    )
    _MAPPER_NUM_IMAGES_PATTERN = re.compile(
        r"NumImages\(\)\s*>?=\s*2\s*\((\d+)\s+vs\.\s*2\)",
        re.IGNORECASE,
    )
    _MAPPER_REGISTERED_FRAME_PATTERN = re.compile(r"num_reg_frames=(\d+)", re.IGNORECASE)
    _MODEL_ANALYZER_REGISTERED_IMAGES_PATTERN = re.compile(r"Registered images:\s*(\d+)", re.IGNORECASE)
    _MODEL_ANALYZER_POINTS_PATTERN = re.compile(r"Points:\s*(\d+)", re.IGNORECASE)

    def __init__(
        self,
        colmap_binary: str = "colmap",
        timeout_seconds: int = 1800,
        use_gpu: bool = False,
        profile: str = "balanced",
        gpu_mode: str = "disabled",
        gpu_probe_reason: str = "not_provided",
        enable_dense_stages: bool = True,
        camera_model: str = "SIMPLE_RADIAL",
        single_camera: bool = True,
        require_dense_reconstruction: bool = False,
    ) -> None:
        self.colmap_binary = (colmap_binary or "colmap").strip() or "colmap"
        self.timeout_seconds = max(timeout_seconds, 30)
        self.use_gpu = use_gpu
        self.profile = self._normalize_profile(profile)
        self.gpu_mode = str(gpu_mode or "disabled").strip().lower() or "disabled"
        self.gpu_probe_reason = str(gpu_probe_reason or "not_provided").strip() or "not_provided"
        self.enable_dense_stages = enable_dense_stages
        self.camera_model = camera_model
        self.single_camera = single_camera
        self.require_dense_reconstruction = require_dense_reconstruction
        self._detected_binary: str | None = None
        self._detection_attempted = False
        self._feature_extraction_gpu_option: str | None = None
        self._feature_matching_gpu_option: str | None = None
        self._dense_reconstruction_supported: bool | None = None
        self._colmap_version: str | None = None
        self._surface_reconstruction = SurfaceReconstruction(min_surface_points=1200)
        self._geometric_priors = GeometricPriorDetector()
        self._texture_projection = TextureProjection()

    @property
    def detected_binary(self) -> str | None:
        return self._detected_binary

    @classmethod
    def profile_options(cls, profile: str, *, gpu_available: bool = True) -> dict[str, object]:
        normalized = cls._normalize_profile(profile)
        if normalized == "conservative":
            return {
                "profile": "conservative",
                "SiftExtraction.use_gpu": 0,
                "SiftMatching.use_gpu": 0,
                "dense_enabled": False,
                "recommended_for": "pruebas rapidas y hardware limitado",
                "timeout_seconds": 900,
            }
        if normalized == "quality":
            return {
                "profile": "quality",
                "SiftExtraction.use_gpu": 1 if gpu_available else 0,
                "SiftMatching.use_gpu": 1 if gpu_available else 0,
                "dense_enabled": False,
                "dense_optional": True,
                "recommended_for": "evidencia de mayor calidad con mas tiempo disponible",
                "timeout_seconds": 3600,
            }
        return {
            "profile": "balanced",
            "SiftExtraction.use_gpu": 1 if gpu_available else 0,
            "SiftMatching.use_gpu": 1 if gpu_available else 0,
            "dense_enabled": False,
            "recommended_for": "sustentacion estable con RTX disponible",
            "timeout_seconds": 1800,
        }

    @staticmethod
    def _normalize_profile(profile: str | None) -> str:
        normalized = str(profile or "balanced").strip().lower()
        return normalized if normalized in {"conservative", "balanced", "quality"} else "balanced"

    def is_available(self) -> bool:
        return self.detect_binary() is not None

    def get_colmap_version(self, binary: str | None = None) -> str | None:
        resolved_binary = binary or self.detect_binary()
        if not resolved_binary:
            return None
        if self._colmap_version is not None:
            return self._colmap_version

        for args in (["--version"], ["version"]):
            try:
                completed = subprocess.run(
                    [resolved_binary, *args],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self._PROBE_TIMEOUT_SECONDS,
                    check=False,
                )
            except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
                continue
            output = f"{completed.stdout}\n{completed.stderr}".strip()
            if completed.returncode == 0 and output:
                self._colmap_version = self._tail(output)
                return self._colmap_version

        help_text = self._probe_binary_help(resolved_binary)
        version_match = re.search(r"COLMAP\s+([0-9][^\s]*)", help_text, re.IGNORECASE)
        self._colmap_version = version_match.group(0) if version_match else None
        return self._colmap_version

    @staticmethod
    def detect_nvidia_gpu(timeout_seconds: int = 3) -> dict[str, object]:
        try:
            completed = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(timeout_seconds, 1),
                check=False,
            )
        except FileNotFoundError:
            return {"available": False, "reason": "nvidia_smi_not_found", "raw_output": ""}
        except subprocess.TimeoutExpired:
            return {"available": False, "reason": "nvidia_smi_timeout", "raw_output": ""}
        except OSError as exc:
            return {"available": False, "reason": f"nvidia_smi_os_error: {exc}", "raw_output": ""}

        output = f"{completed.stdout}\n{completed.stderr}".strip()
        lower_output = output.lower()
        names = re.findall(r"GPU\s+\d+:\s+(.+?)\s+\(", output)
        return {
            "available": completed.returncode == 0 and "gpu" in lower_output,
            "reason": "nvidia_smi_detected_gpu" if completed.returncode == 0 and "gpu" in lower_output else f"nvidia_smi_returncode_{completed.returncode}",
            "rtx_4050_detected": "rtx 4050" in lower_output,
            "names": names,
            "raw_output": output,
        }

    def detect_binary(self, force_refresh: bool = False) -> str | None:
        if self._detection_attempted and not force_refresh:
            return self._detected_binary

        self._detection_attempted = True
        self._detected_binary = None
        candidates = self._build_binary_candidates()

        logger.info(
            "Checking COLMAP availability. configured_path=%s candidates=%s",
            self.colmap_binary,
            candidates,
        )

        for candidate in candidates:
            if self._probe_binary(candidate):
                self._detected_binary = candidate
                logger.info("COLMAP detected successfully using candidate=%s", candidate)
                return candidate

        logger.warning(
            "COLMAP detection failed. None of the candidates responded to '-h'. candidates=%s",
            candidates,
        )
        return None

    def reconstruct(
        self,
        project_id: str,
        images_dir: Path,
        output_dir: Path,
        output_format: OutputFormat,
        progress_callback: ReconstructionProgressCallback | None = None,
    ) -> ReconstructionResult:
        started_at = time.perf_counter()
        resolved_binary = self.detect_binary()
        if resolved_binary is None:
            raise ProcessingError(
                "COLMAP no esta disponible. Configura LOCAL3D_COLMAP_PATH o LOCAL3D_COLMAP_BINARY con la ruta "
                "del ejecutable, o agrega 'colmap', 'colmap.exe' o 'colmap.bat' al PATH del sistema.",
                reason_code="colmap_unavailable",
                current_stage="starting",
                metadata={
                    "current_stage": "starting",
                    "reason_code": "colmap_unavailable",
                    "status_message": "COLMAP no disponible en el entorno del backend.",
                },
                retryable=False,
            )

        image_paths = self._collect_images(images_dir)
        if len(image_paths) < 2:
            raise ProcessingError(
                "COLMAP requiere al menos 2 imagenes para intentar la reconstruccion sparse.",
                reason_code="insufficient_input_images",
                current_stage="starting",
                metadata={
                    "current_stage": "starting",
                    "reason_code": "insufficient_input_images",
                    "status_message": "No hay imagenes suficientes para iniciar reconstruccion.",
                    "image_count": len(image_paths),
                },
                retryable=False,
            )
        if output_format not in {OutputFormat.GLB, OutputFormat.OBJ}:
            raise ProcessingError(
                f"Formato de salida no soportado para COLMAP: {output_format.value}.",
                reason_code="unsupported_output_format",
                current_stage="starting",
                metadata={
                    "current_stage": "starting",
                    "reason_code": "unsupported_output_format",
                    "status_message": "Formato de salida no soportado por el engine COLMAP.",
                    "requested_output_format": output_format.value,
                },
                retryable=False,
            )

        trimesh_module = self._import_trimesh()
        feature_extraction_gpu_option = self._get_feature_extraction_gpu_option(resolved_binary)
        feature_matching_gpu_option = self._get_feature_matching_gpu_option(resolved_binary)
        dense_stages_enabled = bool(self.enable_dense_stages)
        dense_reconstruction_supported = (
            self._supports_dense_reconstruction(resolved_binary)
            if dense_stages_enabled
            else False
        )
        if self.require_dense_reconstruction and not dense_reconstruction_supported:
            if dense_stages_enabled:
                error_message = (
                    "El backend exige reconstruccion densa de COLMAP, pero el binario detectado no tiene soporte CUDA. "
                    "Desactiva LOCAL3D_COLMAP_REQUIRE_DENSE_RECONSTRUCTION para permitir fallback sparse o ejecuta en "
                    "un equipo con NVIDIA CUDA."
                )
                status_message = "No hay soporte CUDA para reconstruccion densa en el binario de COLMAP configurado."
            else:
                error_message = (
                    "El backend exige reconstruccion densa de COLMAP, pero LOCAL3D_COLMAP_ENABLE_DENSE_STAGES=false. "
                    "Activa etapas densas o desactiva LOCAL3D_COLMAP_REQUIRE_DENSE_RECONSTRUCTION."
                )
                status_message = "Las etapas densas de COLMAP estan deshabilitadas por configuracion."
            raise ProcessingError(
                error_message,
                reason_code=self._DENSE_RECONSTRUCTION_UNAVAILABLE_REASON_CODE,
                current_stage=self._DENSE_RECONSTRUCTION_UNAVAILABLE_STAGE,
                metadata={
                    "current_stage": self._DENSE_RECONSTRUCTION_UNAVAILABLE_STAGE,
                    "reason_code": self._DENSE_RECONSTRUCTION_UNAVAILABLE_REASON_CODE,
                    "status_message": status_message,
                    "dense_reconstruction_supported": dense_reconstruction_supported,
                    "dense_reconstruction_required": self.require_dense_reconstruction,
                    "dense_stages_enabled": dense_stages_enabled,
                    "gpu_mode": self.gpu_mode,
                    "gpu_probe_reason": self.gpu_probe_reason,
                    "gpu_requested": self.use_gpu,
                    "colmap_binary": resolved_binary,
                },
                allow_fallback=False,
                retryable=False,
            )
        gpu_requested = bool(self.use_gpu)
        effective_use_gpu = gpu_requested
        gpu_used = False
        gpu_fallback_to_cpu = False
        gpu_error_message: str | None = None
        allow_gpu_cpu_fallback = self._profile_allows_gpu_cpu_fallback()
        gpu_runtime_fallback_applied = False
        gpu_runtime_fallback_stage: str | None = None
        gpu_runtime_fallback_reason: str | None = None
        warnings: list[str] = []
        if not dense_stages_enabled:
            warning = (
                "Las etapas densas de COLMAP estan deshabilitadas por configuracion. "
                "Se generara mesh aproximado desde la reconstruccion sparse en CPU."
            )
            warnings.append(warning)
            logger.warning("Project %s | %s", project_id, warning)
        elif gpu_requested and not dense_reconstruction_supported and dense_stages_enabled:
            warning = (
                "Se solicito uso de GPU en COLMAP, pero no hay soporte CUDA para etapas densas. "
                "Se intentara GPU en SIFT y, si falla, se aplicara la politica de fallback del perfil."
            )
            warnings.append(warning)
            logger.warning("Project %s | %s", project_id, warning)

        output_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir = output_dir / "workspace"
        sparse_dir = workspace_dir / "sparse"
        dense_dir = workspace_dir / "dense"
        logs_dir = output_dir / "logs" / "colmap"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        dense_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        database_path = workspace_dir / "database.db"
        txt_model_dir = output_dir / "colmap_sparse_txt"
        txt_model_dir.mkdir(parents=True, exist_ok=True)
        raw_ply_path = output_dir / f"{project_id}_sparse.ply"
        fused_ply_path = dense_dir / "fused.ply"
        poisson_mesh_path = dense_dir / "meshed-poisson.ply"
        obj_model_path = output_dir / f"{project_id}_model.obj"
        glb_model_path = output_dir / f"{project_id}_model.glb"
        metadata_path = output_dir / f"{project_id}_colmap_metadata.json"

        logger.info(
            "Project %s | Starting COLMAP dense reconstruction with %s images. output_dir=%s",
            project_id,
            len(image_paths),
            output_dir,
        )
        self._notify_progress(
            progress_callback,
            {
                "engine": self.name,
                "current_stage": "starting",
                "progress": 0.02,
                "status_message": "Iniciando procesamiento con COLMAP.",
                "workspace": {
                    "root": str(workspace_dir),
                    "database_path": str(database_path),
                    "sparse_dir": str(sparse_dir),
                    "dense_dir": str(dense_dir),
                },
                "logs": {
                    "directory": str(logs_dir),
                },
                "runtime": {
                    "dense_stages_enabled": dense_stages_enabled,
                    "dense_reconstruction_supported": dense_reconstruction_supported,
                    "dense_reconstruction_required": self.require_dense_reconstruction,
                    "gpu_mode": self.gpu_mode,
                    "gpu_probe_reason": self.gpu_probe_reason,
                    "gpu_requested": gpu_requested,
                    "gpu_effective": effective_use_gpu,
                    "gpu_used": gpu_used,
                    "gpu_fallback_to_cpu": gpu_fallback_to_cpu,
                    "gpu_runtime_fallback_applied": gpu_runtime_fallback_applied,
                    "gpu_error_message": gpu_error_message,
                },
            },
        )

        command_traces: list[ColmapCommandTrace] = []
        stage_timings: dict[str, float] = {}

        feature_extractor_command = [
            resolved_binary,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--ImageReader.single_camera",
            "1" if self.single_camera else "0",
            "--ImageReader.camera_model",
            self.camera_model,
            f"--{feature_extraction_gpu_option}",
            "1" if effective_use_gpu else "0",
        ]
        feature_trace, feature_fell_back_to_cpu, feature_fallback_reason = self._run_command_with_optional_gpu_fallback(
            project_id=project_id,
            name="feature_extractor",
            command=feature_extractor_command,
            logs_dir=logs_dir,
            progress_callback=progress_callback,
            progress_value=0.10,
            stage_message="Ejecutando feature_extractor.",
            gpu_flag_name=f"--{feature_extraction_gpu_option}",
            allow_gpu_fallback=effective_use_gpu and allow_gpu_cpu_fallback,
        )
        if feature_fell_back_to_cpu:
            effective_use_gpu = False
            gpu_fallback_to_cpu = True
            gpu_error_message = feature_fallback_reason
            gpu_runtime_fallback_applied = True
            gpu_runtime_fallback_stage = "feature_extractor"
            gpu_runtime_fallback_reason = feature_fallback_reason
            warning = (
                "COLMAP fallo al iniciar en GPU durante feature_extractor. "
                "Se reintento en CPU para mantener estabilidad."
            )
            warnings.append(warning)
            logger.warning("Project %s | %s", project_id, warning)
        elif gpu_requested:
            gpu_used = True
        command_traces.append(feature_trace)
        stage_timings["feature_extractor"] = feature_trace.duration_seconds
        self._notify_progress(
            progress_callback,
            {
                "current_stage": "feature_extractor",
                "progress": 0.18,
                "status_message": "feature_extractor completado.",
                "stage_timings_seconds": dict(stage_timings),
            },
        )

        exhaustive_matcher_command = [
            resolved_binary,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            f"--{feature_matching_gpu_option}",
            "1" if effective_use_gpu else "0",
        ]
        matcher_trace, matcher_fell_back_to_cpu, matcher_fallback_reason = self._run_command_with_optional_gpu_fallback(
            project_id=project_id,
            name="exhaustive_matcher",
            command=exhaustive_matcher_command,
            logs_dir=logs_dir,
            progress_callback=progress_callback,
            progress_value=0.22,
            stage_message="Ejecutando exhaustive_matcher.",
            gpu_flag_name=f"--{feature_matching_gpu_option}",
            allow_gpu_fallback=effective_use_gpu and allow_gpu_cpu_fallback,
        )
        if matcher_fell_back_to_cpu:
            effective_use_gpu = False
            gpu_fallback_to_cpu = True
            gpu_error_message = matcher_fallback_reason
            gpu_runtime_fallback_applied = True
            gpu_runtime_fallback_stage = "exhaustive_matcher"
            gpu_runtime_fallback_reason = matcher_fallback_reason
            warning = (
                "COLMAP fallo al iniciar en GPU durante exhaustive_matcher. "
                "Se reintento en CPU para mantener estabilidad."
            )
            warnings.append(warning)
            logger.warning("Project %s | %s", project_id, warning)
        elif gpu_requested:
            gpu_used = True
        command_traces.append(matcher_trace)
        stage_timings["exhaustive_matcher"] = matcher_trace.duration_seconds
        self._notify_progress(
            progress_callback,
            {
                "current_stage": "exhaustive_matcher",
                "progress": 0.32,
                "status_message": "exhaustive_matcher completado.",
                "stage_timings_seconds": dict(stage_timings),
            },
        )

        mapper_trace = self._run_command(
            project_id,
            "mapper",
            [
                resolved_binary,
                "mapper",
                "--database_path",
                str(database_path),
                "--image_path",
                str(images_dir),
                "--output_path",
                str(sparse_dir),
            ],
            logs_dir,
            progress_callback,
            0.36,
            "Ejecutando mapper.",
        )
        command_traces.append(mapper_trace)
        stage_timings["mapper"] = mapper_trace.duration_seconds
        sparse_model_dir = self._validate_mapper_output(
            sparse_dir=sparse_dir,
            resolved_binary=resolved_binary,
            project_id=project_id,
        )
        self._notify_progress(
            progress_callback,
            {
                "current_stage": "mapper",
                "progress": 0.46,
                "status_message": "mapper completado y modelo sparse validado.",
                "workspace": {
                    "selected_sparse_model_dir": str(sparse_model_dir),
                },
                "stage_timings_seconds": dict(stage_timings),
            },
        )

        converter_txt_trace = self._run_command(
            project_id,
            "model_converter_txt",
            [
                resolved_binary,
                "model_converter",
                "--input_path",
                str(sparse_model_dir),
                "--output_path",
                str(txt_model_dir),
                "--output_type",
                "TXT",
            ],
            logs_dir,
            progress_callback,
            0.50,
            "Exportando modelo sparse a TXT.",
        )
        command_traces.append(converter_txt_trace)
        stage_timings["model_converter_txt"] = converter_txt_trace.duration_seconds

        try:
            converter_ply_trace = self._run_command(
                project_id,
                "model_converter_ply",
                [
                    resolved_binary,
                    "model_converter",
                    "--input_path",
                    str(sparse_model_dir),
                    "--output_path",
                    str(raw_ply_path),
                    "--output_type",
                    "PLY",
                ],
                logs_dir,
                progress_callback,
                0.54,
                "Exportando modelo sparse a PLY.",
            )
            command_traces.append(converter_ply_trace)
            stage_timings["model_converter_ply"] = converter_ply_trace.duration_seconds
        except ProcessingError as exc:
            warning = f"No se pudo exportar el artefacto PLY de COLMAP: {exc}"
            warnings.append(warning)
            logger.warning("Project %s | %s", project_id, warning)

        points = self._load_sparse_points(txt_model_dir / "points3D.txt")
        if not points:
            raise ProcessingError("COLMAP genero un modelo sparse, pero no produjo puntos 3D utilizables.")

        registered_image_count = self._validate_registered_image_count(txt_model_dir / "images.txt")
        camera_count = self._load_camera_count(txt_model_dir / "cameras.txt")
        sparse_points_ply_path = sparse_model_dir / "points3D.ply"

        sparse_ply_trace = self._run_command(
            project_id,
            "model_converter_sparse_ply",
            [
                resolved_binary,
                "model_converter",
                "--input_path",
                str(sparse_model_dir),
                "--output_path",
                str(sparse_points_ply_path),
                "--output_type",
                "PLY",
            ],
            logs_dir,
            progress_callback,
            0.58,
            "Exportando points3D.ply del modelo sparse.",
        )
        command_traces.append(sparse_ply_trace)
        stage_timings["model_converter_sparse_ply"] = sparse_ply_trace.duration_seconds
        self._validate_output_file(sparse_points_ply_path, "el archivo points3D.ply del modelo sparse")

        if not dense_reconstruction_supported:
            if dense_stages_enabled:
                warning = (
                    "COLMAP no tiene soporte CUDA. Se omitio la reconstruccion densa y se genero una malla aproximada "
                    "usando solo la reconstruccion sparse."
                )
                sparse_fallback_status = "COLMAP sin CUDA; generando mesh aproximado desde la nube sparse."
            else:
                warning = (
                    "Las etapas densas de COLMAP estan deshabilitadas por configuracion. "
                    "Se genero una malla aproximada desde la reconstruccion sparse."
                )
                sparse_fallback_status = (
                    "Etapas densas deshabilitadas por configuracion; generando mesh aproximado desde la nube sparse."
                )
            warnings.append(warning)
            logger.warning("Project %s | %s", project_id, warning)
            self._notify_progress(
                progress_callback,
                {
                    "current_stage": "sparse_mesh_fallback",
                    "progress": 0.68,
                    "status_message": sparse_fallback_status,
                    "artifacts": {
                        "sparse_points_ply": str(sparse_points_ply_path),
                    },
                    "stage_timings_seconds": dict(stage_timings),
                },
            )

            fallback_started_at = time.perf_counter()
            (
                final_mesh,
                mesh_method,
                mesh_summary,
                sparse_point_cloud_count,
                sparse_fallback_diagnostics,
            ) = self._build_sparse_fallback_mesh(
                trimesh_module,
                sparse_points_ply_path,
                sparse_model_dir,
                resolved_binary,
                project_id,
                logs_dir,
                progress_callback,
                command_traces,
                stage_timings,
            )
            stage_timings["sparse_mesh_fallback"] = round(time.perf_counter() - fallback_started_at, 3)
            self._notify_progress(
                progress_callback,
                {
                    "current_stage": "sparse_mesh_fallback",
                    "progress": 0.82,
                    "status_message": f"Mesh aproximado generado usando {mesh_method}.",
                    "metrics": {
                        "sparse_point_cloud_count": sparse_point_cloud_count,
                        "mesh_vertex_count": mesh_summary.vertex_count,
                        "mesh_face_count": mesh_summary.face_count,
                    },
                    "sparse_fallback": dict(sparse_fallback_diagnostics),
                    "stage_timings_seconds": dict(stage_timings),
                },
            )

            obj_started_at = time.perf_counter()
            self._export_mesh_asset(project_id, final_mesh, obj_model_path, "obj")
            stage_timings["obj_export"] = round(time.perf_counter() - obj_started_at, 3)

            glb_started_at = time.perf_counter()
            self._export_mesh_asset(project_id, final_mesh, glb_model_path, "glb")
            glb_summary = self._validate_exported_glb(trimesh_module, glb_model_path)
            stage_timings["glb_export"] = round(time.perf_counter() - glb_started_at, 3)

            model_path = glb_model_path if output_format == OutputFormat.GLB else obj_model_path
            self._validate_output_file(model_path, f"el artefacto final solicitado ({model_path.name})")

            elapsed_seconds = round(time.perf_counter() - started_at, 3)
            metrics = {
                "total_processing_seconds": elapsed_seconds,
                "image_count_processed": len(image_paths),
                "reconstructed_camera_count": registered_image_count,
                "intrinsic_camera_count": camera_count,
                "point_3d_count": len(points),
                "sparse_point_cloud_count": sparse_point_cloud_count,
                "mesh_vertex_count": glb_summary.vertex_count,
                "mesh_face_count": glb_summary.face_count,
            }
            artifacts = {
                "model_path": str(model_path),
                "obj_model_path": str(obj_model_path) if obj_model_path.exists() else None,
                "glb_model_path": str(glb_model_path) if glb_model_path.exists() else None,
                "sparse_txt_dir": str(txt_model_dir),
                "raw_sparse_ply": str(sparse_points_ply_path) if sparse_points_ply_path.exists() else None,
                "sparse_delaunay_mesh_ply": sparse_fallback_diagnostics.get("delaunay_mesh_path"),
                "fused_ply_path": None,
                "poisson_mesh_ply": None,
                "logs_dir": str(logs_dir),
            }
            colmap_report_path = self._write_colmap_report(
                project_id=project_id,
                output_dir=output_dir,
                resolved_binary=resolved_binary,
                command_traces=command_traces,
                stage_timings=stage_timings,
                sparse_created=True,
                cameras_reconstructed=camera_count,
                images_registered=registered_image_count,
                points3d_count=len(points),
                model_outputs=artifacts,
                warnings=warnings,
                fallback_used=False,
                gpu_requested=gpu_requested,
                gpu_used=gpu_used,
                gpu_fallback_to_cpu=gpu_fallback_to_cpu,
                gpu_error_message=gpu_error_message,
            )
            artifacts["colmap_report"] = str(colmap_report_path)

            self._notify_progress(
                progress_callback,
                {
                    "engine": self.name,
                    "current_stage": "export",
                    "progress": 0.94,
                    "status_message": "Artefactos finales exportados desde la malla aproximada sparse.",
                    "metrics": metrics,
                    "artifacts": artifacts,
                    "stage_timings_seconds": dict(stage_timings),
                },
            )

            metadata = {
                "engine": self.name,
                "engine_requested": self.name,
                "processing_seconds": elapsed_seconds,
                "output_path": str(model_path),
                "requested_output_format": output_format.value,
                "actual_output_format": model_path.suffix.lower().lstrip("."),
                "reconstruction_type": "sparse_photogrammetry_mesh_fallback",
                "dense_stages_enabled": dense_stages_enabled,
                "dense_reconstruction_supported": dense_reconstruction_supported,
                "dense_reconstruction_required": self.require_dense_reconstruction,
                "gpu_mode": self.gpu_mode,
                "gpu_probe_reason": self.gpu_probe_reason,
                "gpu_requested": gpu_requested,
                "gpu_used": gpu_used,
                "gpu_effective": effective_use_gpu,
                "gpu_fallback_to_cpu": gpu_fallback_to_cpu,
                "gpu_error_message": gpu_error_message,
                "gpu_runtime_fallback_applied": gpu_runtime_fallback_applied,
                "gpu_runtime_fallback_stage": gpu_runtime_fallback_stage,
                "gpu_runtime_fallback_reason": gpu_runtime_fallback_reason,
                "image_count_processed": len(image_paths),
                "registered_image_count": registered_image_count,
                "camera_count": camera_count,
                "point_count": len(points),
                "mesh_vertex_count": glb_summary.vertex_count,
                "mesh_face_count": glb_summary.face_count,
                "current_stage": "completed_with_fallback",
                "progress": 1.0,
                "status_message": "Reconstruccion completada con fallback sparse.",
                "metrics": metrics,
                "stage_timings_seconds": stage_timings,
                "workspace": {
                    "root": str(workspace_dir),
                    "database_path": str(database_path),
                    "sparse_dir": str(sparse_dir),
                    "dense_dir": str(dense_dir),
                    "selected_sparse_model_dir": str(sparse_model_dir),
                },
                "artifacts": artifacts,
                "logs": {
                    "directory": str(logs_dir),
                },
                "commands": [asdict(trace) for trace in command_traces],
                "fallback": {
                    "used": False,
                    "from_engine": None,
                    "reason": None,
                },
                "sparse_fallback": {
                    "used": True,
                    "reason": warning,
                    "mesh_method": mesh_method,
                    "source_ply_path": str(sparse_points_ply_path),
                    "scipy_available": sparse_fallback_diagnostics["scipy_available"],
                    "attempted_mesh_method": sparse_fallback_diagnostics["attempted_method"],
                    "final_mesh_method": sparse_fallback_diagnostics["final_method"],
                    "convex_hull_exception": sparse_fallback_diagnostics["convex_hull_exception"],
                    "delaunay_exception": sparse_fallback_diagnostics["delaunay_exception"],
                    "delaunay_mesh_path": sparse_fallback_diagnostics["delaunay_mesh_path"],
                    "color_transfer_method": sparse_fallback_diagnostics["color_transfer_method"],
                    "shape_diagnostics": sparse_fallback_diagnostics["shape_diagnostics"],
                },
                "warnings": warnings,
                "colmap_binary": resolved_binary,
                "metadata_path": str(metadata_path),
                "colmap_report_path": str(colmap_report_path),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

            logger.info(
                "Project %s | Sparse fallback reconstruction completed in %.3fs. registered_images=%s cameras=%s sparse_points=%s mesh_vertices=%s mesh_faces=%s scipy_available=%s attempted_method=%s final_method=%s model=%s",
                project_id,
                elapsed_seconds,
                registered_image_count,
                camera_count,
                len(points),
                glb_summary.vertex_count,
                glb_summary.face_count,
                sparse_fallback_diagnostics["scipy_available"],
                sparse_fallback_diagnostics["attempted_method"],
                sparse_fallback_diagnostics["final_method"],
                model_path,
            )
            self._notify_progress(
                progress_callback,
                {
                    "engine": self.name,
                    "current_stage": "completed_with_fallback",
                    "progress": 1.0,
                    "status_message": "Reconstruccion completada con fallback sparse.",
                    "metrics": metrics,
                    "artifacts": artifacts,
                    "stage_timings_seconds": dict(stage_timings),
                },
            )

            return ReconstructionResult(
                engine_name=self.name,
                requested_output_format=output_format,
                model_path=model_path,
                metadata=metadata,
            )

        undistorter_trace = self._run_command(
            project_id,
            "image_undistorter",
            [
                resolved_binary,
                "image_undistorter",
                "--image_path",
                str(images_dir),
                "--input_path",
                str(sparse_model_dir),
                "--output_path",
                str(dense_dir),
                "--output_type",
                "COLMAP",
            ],
            logs_dir,
            progress_callback,
            0.58,
            "Ejecutando image_undistorter.",
        )
        command_traces.append(undistorter_trace)
        stage_timings["image_undistorter"] = undistorter_trace.duration_seconds
        dense_workspace = self._validate_image_undistorter_output(dense_dir)
        self._notify_progress(
            progress_callback,
            {
                "current_stage": "image_undistorter",
                "progress": 0.66,
                "status_message": "image_undistorter completado y workspace denso validado.",
                "workspace": dense_workspace,
                "stage_timings_seconds": dict(stage_timings),
            },
        )

        patch_match_trace = self._run_command(
            project_id,
            "patch_match_stereo",
            [
                resolved_binary,
                "patch_match_stereo",
                "--workspace_path",
                str(dense_dir),
                "--workspace_format",
                "COLMAP",
                "--PatchMatchStereo.geom_consistency",
                "true",
            ],
            logs_dir,
            progress_callback,
            0.70,
            "Ejecutando patch_match_stereo.",
        )
        command_traces.append(patch_match_trace)
        stage_timings["patch_match_stereo"] = patch_match_trace.duration_seconds
        depth_maps_dir = self._validate_patch_match_output(dense_dir)
        self._notify_progress(
            progress_callback,
            {
                "current_stage": "patch_match_stereo",
                "progress": 0.78,
                "status_message": "patch_match_stereo completado y depth maps validados.",
                "workspace": {
                    "depth_maps_dir": str(depth_maps_dir),
                },
                "stage_timings_seconds": dict(stage_timings),
            },
        )

        fusion_trace = self._run_command(
            project_id,
            "stereo_fusion",
            [
                resolved_binary,
                "stereo_fusion",
                "--workspace_path",
                str(dense_dir),
                "--workspace_format",
                "COLMAP",
                "--input_type",
                "geometric",
                "--output_path",
                str(fused_ply_path),
            ],
            logs_dir,
            progress_callback,
            0.82,
            "Ejecutando stereo_fusion.",
        )
        command_traces.append(fusion_trace)
        stage_timings["stereo_fusion"] = fusion_trace.duration_seconds
        self._validate_output_file(fused_ply_path, "el archivo fused.ply")
        self._notify_progress(
            progress_callback,
            {
                "current_stage": "stereo_fusion",
                "progress": 0.86,
                "status_message": "stereo_fusion completado y fused.ply validado.",
                "artifacts": {
                    "fused_ply_path": str(fused_ply_path),
                },
                "stage_timings_seconds": dict(stage_timings),
            },
        )

        poisson_trace = self._run_command(
            project_id,
            "poisson_mesher",
            [
                resolved_binary,
                "poisson_mesher",
                "--input_path",
                str(fused_ply_path),
                "--output_path",
                str(poisson_mesh_path),
            ],
            logs_dir,
            progress_callback,
            0.88,
            "Ejecutando poisson_mesher.",
        )
        command_traces.append(poisson_trace)
        stage_timings["poisson_mesher"] = poisson_trace.duration_seconds
        self._validate_output_file(poisson_mesh_path, "la malla meshed-poisson.ply")
        self._notify_progress(
            progress_callback,
            {
                "current_stage": "poisson_mesher",
                "progress": 0.92,
                "status_message": "poisson_mesher completado y mesh final validado.",
                "artifacts": {
                    "poisson_mesh_ply": str(poisson_mesh_path),
                },
                "stage_timings_seconds": dict(stage_timings),
            },
        )

        reconstruction_type = "dense_photogrammetry_mesh"
        status_message = "Reconstruccion completada."
        sparse_fallback_metadata: dict[str, object] | None = None
        surface_reconstruction_metadata: dict[str, object] | None = None
        geometric_prior_metadata: dict[str, object] | None = None
        texture_projection_metadata: dict[str, object] | None = None
        surface_attempted = False
        surface_success = False
        surface_failure_reason: str | None = None
        mesh_vertex_count = 0
        mesh_face_count = 0
        point_cloud_count = 0
        dense_faces_count_real = 0
        dense_vertices_count_real = 0
        surface_faces_count_real = 0
        surface_vertices_count_real = 0
        visual_faces_count = 0
        visual_vertices_count = 0
        visual_geometry_type = "dense_mesh"
        mesh_face_count_is_visual_only = False

        dense_mesh_usable = False
        try:
            final_mesh = self._load_trimesh_asset(trimesh_module, poisson_mesh_path, "ply", "mesh")
            mesh_summary = self._validate_mesh_asset(final_mesh, poisson_mesh_path.name)

            obj_started_at = time.perf_counter()
            self._export_mesh_asset(project_id, final_mesh, obj_model_path, "obj")
            stage_timings["obj_export"] = round(time.perf_counter() - obj_started_at, 3)

            glb_started_at = time.perf_counter()
            self._export_mesh_asset(project_id, final_mesh, glb_model_path, "glb")
            glb_summary = self._validate_exported_glb(trimesh_module, glb_model_path)
            stage_timings["glb_export"] = round(time.perf_counter() - glb_started_at, 3)
            mesh_vertex_count = glb_summary.vertex_count
            mesh_face_count = glb_summary.face_count
            dense_vertices_count_real = mesh_vertex_count
            dense_faces_count_real = mesh_face_count
            visual_vertices_count = mesh_vertex_count
            visual_faces_count = mesh_face_count
            visual_geometry_type = "dense_mesh"
            dense_mesh_usable = mesh_face_count >= 500
            if not dense_mesh_usable:
                raise ProcessingError(
                    f"La malla densa existe pero no es utilizable para visualizacion solida (faces={mesh_face_count} < 500)."
                )
        except ProcessingError as exc:
            warning = (
                "La malla densa Poisson no fue utilizable; se exportara nube de puntos sparse real "
                "como salida parcial defendible."
            )
            warnings.append(warning)
            logger.warning("Project %s | %s Detalle: %s", project_id, warning, exc)
            sparse_xyz = [(point.x, point.y, point.z) for point in points]
            sparse_rgb = [(point.r, point.g, point.b) for point in points]
            surface_result = self._surface_reconstruction.reconstruct_from_sparse(
                points_xyz=sparse_xyz,
                point_colors_rgb=sparse_rgb,
                trimesh_module=trimesh_module,
            )
            surface_attempted = True
            if surface_result is not None:
                reconstruction_type = "sparse_surface_reconstruction"
                status_message = "Reconstruccion completada con superficie aproximada desde sparse."
                sparse_fallback_metadata = {
                    "used": True,
                    "reason": str(exc),
                    "mesh_method": "surface_from_sparse",
                    "final_mesh_method": surface_result.method_used,
                    "source_points": "colmap_sparse_txt/points3D.txt",
                    "visualization_type": "reconstructed_surface",
                    "color_transfer_method": surface_result.color_strategy,
                }
                surface_reconstruction_metadata = surface_result.to_dict()
                final_mesh = surface_result.mesh
                obj_started_at = time.perf_counter()
                self._export_mesh_asset(project_id, final_mesh, obj_model_path, "obj")
                stage_timings["obj_export"] = round(time.perf_counter() - obj_started_at, 3)

                glb_started_at = time.perf_counter()
                self._export_mesh_asset(project_id, final_mesh, glb_model_path, "glb")
                glb_summary = self._validate_exported_glb(trimesh_module, glb_model_path)
                stage_timings["glb_export"] = round(time.perf_counter() - glb_started_at, 3)
                mesh_vertex_count = glb_summary.vertex_count
                mesh_face_count = glb_summary.face_count
                surface_vertices_count_real = int(surface_result.vertices_count)
                surface_faces_count_real = int(surface_result.faces_count)
                visual_vertices_count = mesh_vertex_count
                visual_faces_count = mesh_face_count
                visual_geometry_type = "approximated_surface"
                surface_success = surface_faces_count_real >= 500 and surface_vertices_count_real >= 100
                if not surface_success:
                    surface_failure_reason = (
                        f"surface_too_weak vertices={surface_vertices_count_real} faces={surface_faces_count_real}"
                    )
                if mesh_face_count < 500 or mesh_vertex_count < 100:
                    surface_reconstruction_metadata = surface_result.to_dict()
                    surface_reconstruction_metadata["failure_reason"] = (
                        f"surface_too_weak vertices={mesh_vertex_count} faces={mesh_face_count}"
                    )
                    reconstruction_type = "sparse_photogrammetry_point_cloud_fallback"
                    status_message = "Superficie aproximada insuficiente; salida final en nube sparse visible."
                    obj_started_at = time.perf_counter()
                    self._write_obj_point_cloud(obj_model_path, points, project_id)
                    stage_timings["obj_export"] = round(time.perf_counter() - obj_started_at, 3)
                    glb_started_at = time.perf_counter()
                    visualization_type = self._write_glb_point_cloud(glb_model_path, points, project_id)
                    stage_timings["glb_export"] = round(time.perf_counter() - glb_started_at, 3)
                    point_cloud_count = len(points)
                    visual_geometry_type = visualization_type
                    visual_vertices_count = point_cloud_count
                    visual_faces_count = 0
                    mesh_face_count_is_visual_only = True
                    sparse_fallback_metadata = {
                        "used": False,
                        "reason": str(exc),
                        "mesh_method": "point_cloud_from_sparse",
                        "final_mesh_method": "point_cloud_from_sparse",
                        "source_points": "colmap_sparse_txt/points3D.txt",
                        "visualization_type": visualization_type,
                        "surface_failure_reason": surface_reconstruction_metadata["failure_reason"],
                    }
                    mesh_vertex_count = 0
                    mesh_face_count = 0
                    prior_result = self._geometric_priors.detect_and_build(
                        points_xyz=sparse_xyz,
                        trimesh_module=trimesh_module,
                    )
                    geometric_prior_metadata = prior_result.to_dict()
                    if prior_result.prior_used and prior_result.mesh is not None:
                        try:
                            texture_result = self._texture_projection.apply(
                                mesh=prior_result.mesh,
                                point_colors_rgb=sparse_rgb,
                                image_dir=images_dir,
                                detected_shape_prior=prior_result.detected_shape_prior,
                                output_dir=output_dir,
                            )
                            texture_projection_metadata = texture_result.to_dict()
                        except Exception:
                            texture_projection_metadata = {
                                "texture_source": "average_image_color",
                                "texture_method": "exception_fallback_average_color",
                                "selected_images": [],
                                "texture_confidence": 0.2,
                                "textured_faces_count": 0,
                                "untextured_faces_count": self._safe_len(getattr(prior_result.mesh, "faces", ()) if prior_result.mesh is not None else ()),
                                "texture_limitations": ["texture_projection_failed"],
                                "fallback_texture_used": True,
                            }
                        reconstruction_type = "sparse_geometric_prior_reconstruction"
                        status_message = "Superficie insuficiente; se genero prior geometrico aproximado desde sparse."
                        obj_started_at = time.perf_counter()
                        self._export_mesh_asset(project_id, prior_result.mesh, obj_model_path, "obj")
                        stage_timings["obj_export"] = round(time.perf_counter() - obj_started_at, 3)
                        glb_started_at = time.perf_counter()
                        self._export_mesh_asset(project_id, prior_result.mesh, glb_model_path, "glb")
                        glb_summary = self._validate_exported_glb(trimesh_module, glb_model_path)
                        stage_timings["glb_export"] = round(time.perf_counter() - glb_started_at, 3)
                        visual_geometry_type = "approximated_surface"
                        visual_vertices_count = int(glb_summary.vertex_count)
                        visual_faces_count = int(glb_summary.face_count)
                        mesh_face_count = 0
                        mesh_vertex_count = 0
                        mesh_face_count_is_visual_only = True
                        sparse_fallback_metadata["visualization_type"] = "approximated_surface"
                        sparse_fallback_metadata["mesh_method"] = "geometric_prior"
                        sparse_fallback_metadata["final_mesh_method"] = "geometric_prior"
            else:
                reconstruction_type = "sparse_photogrammetry_point_cloud_fallback"
                status_message = "Reconstruccion completada con salida real sparse (point cloud)."
                surface_success = False
                surface_failure_reason = "surface_reconstruction_failed"
                sparse_fallback_metadata = {
                    "used": True,
                    "reason": str(exc),
                    "mesh_method": "point_cloud_from_sparse",
                    "final_mesh_method": "point_cloud_from_sparse",
                    "source_points": "colmap_sparse_txt/points3D.txt",
                }

                obj_started_at = time.perf_counter()
                self._write_obj_point_cloud(obj_model_path, points, project_id)
                stage_timings["obj_export"] = round(time.perf_counter() - obj_started_at, 3)

                glb_started_at = time.perf_counter()
                visualization_type = self._write_glb_point_cloud(glb_model_path, points, project_id)
                stage_timings["glb_export"] = round(time.perf_counter() - glb_started_at, 3)

                point_cloud_count = len(points)
                visual_geometry_type = visualization_type
                visual_vertices_count = point_cloud_count
                visual_faces_count = 0
                mesh_face_count_is_visual_only = True
                sparse_fallback_metadata["visualization_type"] = visualization_type

        model_path = glb_model_path if output_format == OutputFormat.GLB else obj_model_path
        self._validate_output_file(model_path, f"el artefacto final solicitado ({model_path.name})")

        elapsed_seconds = round(time.perf_counter() - started_at, 3)
        metrics = {
            "total_processing_seconds": elapsed_seconds,
            "image_count_processed": len(image_paths),
            "reconstructed_camera_count": registered_image_count,
            "intrinsic_camera_count": camera_count,
            "point_3d_count": len(points),
            "mesh_vertex_count": mesh_vertex_count,
            "mesh_face_count": mesh_face_count,
            "sparse_point_cloud_count": point_cloud_count,
            "dense_faces_count_real": dense_faces_count_real,
            "dense_vertices_count_real": dense_vertices_count_real,
            "surface_faces_count_real": surface_faces_count_real,
            "surface_vertices_count_real": surface_vertices_count_real,
            "visual_faces_count": visual_faces_count,
            "visual_vertices_count": visual_vertices_count,
            "visual_geometry_type": visual_geometry_type,
            "mesh_face_count_is_visual_only": mesh_face_count_is_visual_only,
            "surface_attempted": surface_attempted,
            "surface_success": surface_success,
            "surface_failure_reason": surface_failure_reason,
        }
        artifacts = {
            "model_path": str(model_path),
            "obj_model_path": str(obj_model_path) if obj_model_path.exists() else None,
            "glb_model_path": str(glb_model_path) if glb_model_path.exists() else None,
            "sparse_txt_dir": str(txt_model_dir),
            "raw_sparse_ply": str(raw_ply_path) if raw_ply_path.exists() else None,
            "fused_ply_path": str(fused_ply_path) if fused_ply_path.exists() else None,
            "poisson_mesh_ply": str(poisson_mesh_path) if poisson_mesh_path.exists() else None,
            "logs_dir": str(logs_dir),
        }
        colmap_report_path = self._write_colmap_report(
            project_id=project_id,
            output_dir=output_dir,
            resolved_binary=resolved_binary,
            command_traces=command_traces,
            stage_timings=stage_timings,
            sparse_created=True,
            cameras_reconstructed=camera_count,
            images_registered=registered_image_count,
            points3d_count=len(points),
            model_outputs=artifacts,
            warnings=warnings,
            fallback_used=False,
            gpu_requested=gpu_requested,
            gpu_used=gpu_used,
            gpu_fallback_to_cpu=gpu_fallback_to_cpu,
            gpu_error_message=gpu_error_message,
        )
        artifacts["colmap_report"] = str(colmap_report_path)

        self._notify_progress(
            progress_callback,
            {
                "engine": self.name,
                "current_stage": "export",
                "progress": 0.96,
                "status_message": "Artefactos finales exportados desde la malla densa.",
                "metrics": metrics,
                "artifacts": artifacts,
                "stage_timings_seconds": dict(stage_timings),
            },
        )

        metadata = {
            "engine": self.name,
            "engine_requested": self.name,
            "processing_seconds": elapsed_seconds,
            "output_path": str(model_path),
            "requested_output_format": output_format.value,
            "actual_output_format": model_path.suffix.lower().lstrip("."),
            "reconstruction_type": reconstruction_type,
                "dense_stages_enabled": dense_stages_enabled,
                "dense_reconstruction_supported": dense_reconstruction_supported,
                "dense_reconstruction_required": self.require_dense_reconstruction,
                "gpu_mode": self.gpu_mode,
                "gpu_probe_reason": self.gpu_probe_reason,
                "gpu_requested": gpu_requested,
                "gpu_used": gpu_used,
                "gpu_effective": effective_use_gpu,
                "gpu_fallback_to_cpu": gpu_fallback_to_cpu,
                "gpu_error_message": gpu_error_message,
                "gpu_runtime_fallback_applied": gpu_runtime_fallback_applied,
                "gpu_runtime_fallback_stage": gpu_runtime_fallback_stage,
                "gpu_runtime_fallback_reason": gpu_runtime_fallback_reason,
                "image_count_processed": len(image_paths),
                "registered_image_count": registered_image_count,
                "camera_count": camera_count,
            "point_count": len(points),
            "mesh_vertex_count": mesh_vertex_count,
            "mesh_face_count": mesh_face_count,
            "current_stage": "completed",
            "progress": 1.0,
            "status_message": status_message,
            "metrics": metrics,
            "stage_timings_seconds": stage_timings,
            "workspace": {
                "root": str(workspace_dir),
                "database_path": str(database_path),
                "sparse_dir": str(sparse_dir),
                "dense_dir": str(dense_dir),
                "selected_sparse_model_dir": str(sparse_model_dir),
                **dense_workspace,
                "depth_maps_dir": str(depth_maps_dir),
            },
            "artifacts": artifacts,
            "logs": {
                "directory": str(logs_dir),
            },
            "commands": [asdict(trace) for trace in command_traces],
            "fallback": {
                "used": False,
                "from_engine": None,
                "reason": None,
            },
            "sparse_fallback": sparse_fallback_metadata,
            "surface_reconstruction": surface_reconstruction_metadata,
            "geometric_prior": geometric_prior_metadata,
            "texture_projection": texture_projection_metadata,
            "surface_attempted": surface_attempted,
            "surface_success": surface_success,
            "surface_failure_reason": surface_failure_reason or (surface_reconstruction_metadata or {}).get("failure_reason"),
            "dense_mesh_usable": dense_mesh_usable,
            "real_geometry_metrics": {
                "dense_faces_count_real": dense_faces_count_real,
                "dense_vertices_count_real": dense_vertices_count_real,
                "surface_faces_count_real": surface_faces_count_real,
                "surface_vertices_count_real": surface_vertices_count_real,
                "real_mesh_available": bool(dense_faces_count_real >= 500 or surface_faces_count_real >= 500),
                "real_mesh_source": (
                    "colmap_dense"
                    if dense_faces_count_real >= 500
                    else "surface_from_sparse"
                    if surface_faces_count_real >= 500
                    else "none"
                ),
            },
            "visualization_metrics": {
                "visual_faces_count": visual_faces_count,
                "visual_vertices_count": visual_vertices_count,
                "visual_geometry_type": visual_geometry_type,
                "visual_faces_are_reconstruction": not mesh_face_count_is_visual_only,
            },
            "warnings": warnings,
            "colmap_binary": resolved_binary,
            "metadata_path": str(metadata_path),
            "colmap_report_path": str(colmap_report_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(
            "Project %s | COLMAP reconstruction completed in %.3fs. registered_images=%s cameras=%s sparse_points=%s mesh_vertices=%s mesh_faces=%s point_cloud_points=%s model=%s reconstruction_type=%s",
            project_id,
            elapsed_seconds,
            registered_image_count,
            camera_count,
            len(points),
            mesh_vertex_count,
            mesh_face_count,
            point_cloud_count,
            model_path,
            reconstruction_type,
        )
        self._notify_progress(
            progress_callback,
            {
                "engine": self.name,
                "current_stage": "completed",
                "progress": 1.0,
                "status_message": "Reconstruccion completada.",
                "metrics": metrics,
                "artifacts": artifacts,
                "stage_timings_seconds": dict(stage_timings),
            },
        )

        return ReconstructionResult(
            engine_name=self.name,
            requested_output_format=output_format,
            model_path=model_path,
            metadata=metadata,
        )

    def _write_colmap_report(
        self,
        *,
        project_id: str,
        output_dir: Path,
        resolved_binary: str | None,
        command_traces: list[ColmapCommandTrace],
        stage_timings: dict[str, float],
        sparse_created: bool,
        cameras_reconstructed: int,
        images_registered: int,
        points3d_count: int,
        model_outputs: dict[str, object],
        warnings: list[str],
        fallback_used: bool,
        gpu_requested: bool,
        gpu_used: bool,
        gpu_fallback_to_cpu: bool,
        gpu_error_message: str | None = None,
        failure_reason: str | None = None,
    ) -> Path:
        pipeline_dir = output_dir / "pipeline"
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        gpu_probe = self.detect_nvidia_gpu(timeout_seconds=3)
        report_payload = {
            "project_id": project_id,
            "colmap_binary": resolved_binary,
            "colmap_version": self.get_colmap_version(resolved_binary),
            "gpu_detected": bool(gpu_probe.get("available")),
            "gpu_name": ", ".join(gpu_probe.get("names") or []) or None,
            "rtx_4050_detected": bool(gpu_probe.get("rtx_4050_detected")),
            "gpu_requested": bool(gpu_requested),
            "gpu_used": bool(gpu_used),
            "gpu_fallback_to_cpu": bool(gpu_fallback_to_cpu),
            "gpu_error_message": gpu_error_message,
            "profile": self.profile,
            "profile_options": self.profile_options(self.profile, gpu_available=bool(gpu_probe.get("available"))),
            "commands_executed": [asdict(trace) for trace in command_traces],
            "command_durations": dict(stage_timings),
            "stage_timings_by_phase": self._summarize_stage_timings(stage_timings),
            "sparse_created": bool(sparse_created),
            "cameras_reconstructed": int(cameras_reconstructed),
            "images_registered": int(images_registered),
            "points3D_count": int(points3d_count),
            "model_outputs": model_outputs,
            "warnings": list(warnings),
            "fallback_used": bool(fallback_used),
            "failure_reason": failure_reason,
        }
        report_path = pipeline_dir / "colmap_report.json"
        report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return report_path

    @classmethod
    def write_failure_report(
        cls,
        *,
        project_id: str,
        output_dir: Path,
        colmap_binary: str | None,
        profile: str,
        failure_reason: str,
        error_context: dict[str, object] | None = None,
        fallback_used: bool = False,
    ) -> Path:
        pipeline_dir = output_dir / "pipeline"
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        gpu_probe = cls.detect_nvidia_gpu(timeout_seconds=3)
        commands_executed: list[object] = []
        command_durations: dict[str, object] = {}
        if isinstance(error_context, dict):
            raw_commands = error_context.get("commands")
            if isinstance(raw_commands, list):
                commands_executed = raw_commands
            raw_timings = error_context.get("stage_timings_seconds")
            if isinstance(raw_timings, dict):
                command_durations = raw_timings

        report_payload = {
            "project_id": project_id,
            "colmap_binary": colmap_binary,
            "colmap_version": None,
            "gpu_detected": bool(gpu_probe.get("available")),
            "gpu_name": ", ".join(gpu_probe.get("names") or []) or None,
            "rtx_4050_detected": bool(gpu_probe.get("rtx_4050_detected")),
            "gpu_requested": bool((error_context or {}).get("gpu_requested", False)),
            "gpu_used": False,
            "gpu_fallback_to_cpu": bool((error_context or {}).get("gpu_fallback_to_cpu", False)),
            "gpu_error_message": (error_context or {}).get("gpu_error_message"),
            "profile": cls._normalize_profile(profile),
            "profile_options": cls.profile_options(profile, gpu_available=bool(gpu_probe.get("available"))),
            "commands_executed": commands_executed,
            "command_durations": command_durations,
            "stage_timings_by_phase": cls._summarize_stage_timings(command_durations),
            "sparse_created": False,
            "cameras_reconstructed": 0,
            "images_registered": int((error_context or {}).get("registered_image_count") or 0),
            "points3D_count": 0,
            "model_outputs": {},
            "warnings": [
                "COLMAP no produjo una reconstruccion sparse real utilizable.",
                "Si el fallback academico esta habilitado, el resultado final no debe reportarse como SfM real.",
            ],
            "fallback_used": bool(fallback_used),
            "failure_reason": failure_reason,
            "error_context": error_context or {},
        }
        report_path = pipeline_dir / "colmap_report.json"
        report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return report_path

    def _build_binary_candidates(self) -> list[str]:
        configured = self.colmap_binary
        env_configured = os.environ.get("LOCAL3D_COLMAP_BINARY") or os.environ.get("LOCAL3D_COLMAP_PATH")
        configured_path = Path(configured)
        candidates: list[str] = []
        seen: set[str] = set()

        def add(candidate: str | None) -> None:
            if not candidate:
                return
            normalized = str(candidate).strip().strip('"').strip("'").strip()
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            candidates.append(normalized)

        if configured_path.is_dir():
            add(str(configured_path / "COLMAP.bat"))
            add(str(configured_path / "colmap.bat"))
            add(str(configured_path / "colmap.exe"))
            add(str(configured_path / "bin" / "colmap.exe"))
            add(str(configured_path / "bin" / "COLMAP.bat"))

        add(configured)
        if configured_path.suffix == "":
            add(f"{configured}.exe")
            add(f"{configured}.bat")
            if configured_path.is_absolute() or str(configured_path.parent) not in {"", "."}:
                add(str(configured_path.with_suffix(".exe")))
                add(str(configured_path.with_suffix(".bat")))
        elif configured_path.suffix.lower() == ".exe":
            add(str(configured_path.with_suffix(".bat")))
            if not configured_path.is_absolute():
                add(configured_path.stem)
        elif configured_path.suffix.lower() == ".bat":
            add(str(configured_path.with_suffix(".exe")))
            if not configured_path.is_absolute():
                add(configured_path.stem)

        if env_configured:
            env_path = Path(env_configured)
            if env_path.is_dir():
                add(str(env_path / "COLMAP.bat"))
                add(str(env_path / "colmap.bat"))
                add(str(env_path / "colmap.exe"))
                add(str(env_path / "bin" / "colmap.exe"))
                add(str(env_path / "bin" / "COLMAP.bat"))
            add(env_configured)

        path_candidate = shutil.which("colmap")
        if path_candidate:
            add(path_candidate)

        add("colmap")
        add("colmap.exe")
        add("colmap.bat")
        return candidates

    def _probe_binary(self, candidate: str) -> bool:
        logger.info("Probing COLMAP candidate with '-h': %s", candidate)
        try:
            completed = subprocess.run(
                [candidate, "-h"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._PROBE_TIMEOUT_SECONDS,
                check=False,
            )
        except FileNotFoundError:
            return False
        except subprocess.TimeoutExpired:
            logger.warning("Timeout while probing COLMAP candidate=%s", candidate)
            return False
        except OSError as exc:
            logger.warning("OS error while probing COLMAP candidate=%s error=%s", candidate, exc)
            return False

        output = f"{completed.stdout}\n{completed.stderr}".strip()
        if completed.returncode == 0:
            logger.info("COLMAP probe succeeded for candidate=%s", candidate)
            return True

        logger.warning(
            "COLMAP probe failed for candidate=%s returncode=%s output=%s",
            candidate,
            completed.returncode,
            self._tail(output),
        )
        return False

    def _probe_binary_help(self, binary: str) -> str:
        try:
            completed = subprocess.run(
                [binary, "-h"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._PROBE_TIMEOUT_SECONDS,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                "Unable to inspect COLMAP root help for binary=%s error=%s",
                binary,
                exc,
            )
            return ""
        return f"{completed.stdout}\n{completed.stderr}"

    def _probe_subcommand_help(self, binary: str, subcommand: str) -> str:
        try:
            completed = subprocess.run(
                [binary, subcommand, "-h"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._PROBE_TIMEOUT_SECONDS,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                "Unable to inspect COLMAP help for subcommand=%s binary=%s error=%s",
                subcommand,
                binary,
                exc,
            )
            return ""
        return f"{completed.stdout}\n{completed.stderr}"

    def _supports_dense_reconstruction(self, binary: str) -> bool:
        if self._dense_reconstruction_supported is not None:
            return self._dense_reconstruction_supported

        help_text = f"{self._probe_binary_help(binary)}\n{self._probe_subcommand_help(binary, 'patch_match_stereo')}"
        normalized_help = help_text.lower()
        if "without cuda" in normalized_help:
            self._dense_reconstruction_supported = False
            logger.warning(
                "COLMAP binary '%s' reports no CUDA support. Dense reconstruction stages will be skipped.",
                binary,
            )
        else:
            self._dense_reconstruction_supported = True
            logger.info(
                "COLMAP binary '%s' appears to support dense reconstruction. CUDA warning not detected in help output.",
                binary,
            )
        return self._dense_reconstruction_supported

    def _get_feature_extraction_gpu_option(self, binary: str) -> str:
        if self._feature_extraction_gpu_option is not None:
            return self._feature_extraction_gpu_option

        help_text = self._probe_subcommand_help(binary, "feature_extractor")
        if "--FeatureExtraction.use_gpu" in help_text:
            self._feature_extraction_gpu_option = "FeatureExtraction.use_gpu"
        elif "--SiftExtraction.use_gpu" in help_text:
            self._feature_extraction_gpu_option = "SiftExtraction.use_gpu"
        else:
            self._feature_extraction_gpu_option = "FeatureExtraction.use_gpu"

        logger.info(
            "Resolved COLMAP GPU flag for feature_extractor: %s",
            self._feature_extraction_gpu_option,
        )
        return self._feature_extraction_gpu_option

    def _get_feature_matching_gpu_option(self, binary: str) -> str:
        if self._feature_matching_gpu_option is not None:
            return self._feature_matching_gpu_option

        help_text = self._probe_subcommand_help(binary, "exhaustive_matcher")
        if "--FeatureMatching.use_gpu" in help_text:
            self._feature_matching_gpu_option = "FeatureMatching.use_gpu"
        elif "--SiftMatching.use_gpu" in help_text:
            self._feature_matching_gpu_option = "SiftMatching.use_gpu"
        else:
            self._feature_matching_gpu_option = "FeatureMatching.use_gpu"

        logger.info(
            "Resolved COLMAP GPU flag for exhaustive_matcher: %s",
            self._feature_matching_gpu_option,
        )
        return self._feature_matching_gpu_option

    @staticmethod
    def _collect_images(images_dir: Path) -> list[Path]:
        if not images_dir.exists():
            raise ProcessingError(f"La carpeta de imagenes no existe: {images_dir}")
        image_paths = sorted(path for path in images_dir.iterdir() if path.is_file())
        if not image_paths:
            raise ProcessingError("No hay imagenes disponibles para reconstruir.")
        return image_paths

    def _run_command(
        self,
        project_id: str,
        name: str,
        command: list[str],
        logs_dir: Path,
        progress_callback: ReconstructionProgressCallback | None,
        progress_value: float,
        stage_message: str,
    ) -> ColmapCommandTrace:
        logger.info(
            "Project %s | Starting COLMAP stage '%s': %s",
            project_id,
            name,
            subprocess.list2cmdline(command),
        )
        self._notify_progress(
            progress_callback,
            {
                "engine": self.name,
                "current_stage": name,
                "progress": progress_value,
                "status_message": stage_message,
            },
        )
        started_at = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            binary_with_spaces = bool(os.name == "nt" and " " in str(command[0]))
            hint = (
                "En Windows, verifica que LOCAL3D_COLMAP_BINARY apunte a COLMAP.bat/colmap.exe sin comillas anidadas."
                if binary_with_spaces
                else None
            )
            logger.exception("COLMAP binary disappeared during execution. command=%s", command[0], exc_info=exc)
            raise ProcessingError(
                "No se pudo ejecutar COLMAP. Verifica la ruta del ejecutable configurado en LOCAL3D_COLMAP_PATH "
                "o LOCAL3D_COLMAP_BINARY.",
                reason_code="colmap_binary_not_found",
                current_stage=name,
                metadata={
                    "current_stage": name,
                    "reason_code": "colmap_binary_not_found",
                    "failed_command": name,
                    "status_message": f"No se encontro ejecutable COLMAP para la etapa '{name}'.",
                    "command": subprocess.list2cmdline(command),
                    "windows_path_with_spaces": binary_with_spaces,
                    "hint": hint,
                },
                retryable=False,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            stdout_path, stderr_path = self._write_command_logs(logs_dir, name, exc.stdout, exc.stderr)
            logger.exception("COLMAP step '%s' timed out after %s seconds", name, self.timeout_seconds, exc_info=exc)
            raise ProcessingError(
                f"COLMAP agoto el tiempo de espera en el paso '{name}' despues de {self.timeout_seconds} segundos. "
                f"Logs: stdout={stdout_path}, stderr={stderr_path}",
                reason_code="colmap_command_timeout",
                current_stage=name,
                metadata={
                    "current_stage": name,
                    "reason_code": "colmap_command_timeout",
                    "failed_command": name,
                    "timeout_seconds": self.timeout_seconds,
                    "status_message": f"Tiempo de espera agotado en etapa '{name}'.",
                    "command": subprocess.list2cmdline(command),
                    "logs": {
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                    },
                },
                retryable=True,
            ) from exc

        duration_seconds = round(time.perf_counter() - started_at, 3)
        stdout_tail = self._tail(completed.stdout)
        stderr_tail = self._tail(completed.stderr)
        stdout_path, stderr_path = self._write_command_logs(logs_dir, name, completed.stdout, completed.stderr)
        trace = ColmapCommandTrace(
            name=name,
            command=subprocess.list2cmdline(command),
            duration_seconds=duration_seconds,
            return_code=completed.returncode,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

        if completed.returncode != 0:
            logger.error(
                "Project %s | COLMAP stage '%s' failed with code %s. stdout=%s stderr=%s",
                project_id,
                name,
                completed.returncode,
                stdout_tail,
                stderr_tail,
            )
            if name == "mapper":
                mapper_error = self._detect_mapper_insufficient_registered_images_error(
                    project_id=project_id,
                    stdout_text=completed.stdout,
                    stderr_text=completed.stderr,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                if mapper_error is not None:
                    raise mapper_error
            detail = stderr_tail or stdout_tail or "Sin detalle adicional."
            raise ProcessingError(
                f"COLMAP fallo en el paso '{name}' con codigo {completed.returncode}. Detalle: {detail}. "
                f"Logs: stdout={stdout_path}, stderr={stderr_path}",
                reason_code="colmap_command_failed",
                current_stage=name,
                metadata={
                    "current_stage": name,
                    "reason_code": "colmap_command_failed",
                    "failed_command": name,
                    "return_code": completed.returncode,
                    "status_message": f"La etapa '{name}' retorno codigo {completed.returncode}.",
                    "command": subprocess.list2cmdline(command),
                    "logs": {
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                    },
                },
                retryable=True,
            )

        logger.info(
            "Project %s | COLMAP stage '%s' completed in %.3fs",
            project_id,
            name,
            duration_seconds,
        )
        return trace

    def _run_command_with_optional_gpu_fallback(
        self,
        *,
        project_id: str,
        name: str,
        command: list[str],
        logs_dir: Path,
        progress_callback: ReconstructionProgressCallback | None,
        progress_value: float,
        stage_message: str,
        gpu_flag_name: str | None,
        allow_gpu_fallback: bool,
    ) -> tuple[ColmapCommandTrace, bool, str | None]:
        try:
            trace = self._run_command(
                project_id=project_id,
                name=name,
                command=command,
                logs_dir=logs_dir,
                progress_callback=progress_callback,
                progress_value=progress_value,
                stage_message=stage_message,
            )
            return trace, False, None
        except ProcessingError as exc:
            if not allow_gpu_fallback or not gpu_flag_name or not self._is_gpu_runtime_failure(exc):
                raise

            retry_command = self._set_command_flag(command, gpu_flag_name, "0")
            reason = str(exc)
            logger.warning(
                "Project %s | GPU runtime issue detected at stage=%s. Retrying in CPU mode. reason=%s",
                project_id,
                name,
                self._tail(reason),
            )
            retry_trace = self._run_command(
                project_id=project_id,
                name=name,
                command=retry_command,
                logs_dir=logs_dir,
                progress_callback=progress_callback,
                progress_value=progress_value,
                stage_message=f"{stage_message} Reintentando en CPU por error de GPU.",
            )
            return retry_trace, True, self._tail(reason)

    @staticmethod
    def _set_command_flag(command: list[str], flag_name: str, value: str) -> list[str]:
        patched = list(command)
        try:
            flag_index = patched.index(flag_name)
        except ValueError:
            return patched

        value_index = flag_index + 1
        if value_index < len(patched):
            patched[value_index] = value
        return patched

    @classmethod
    def _is_gpu_runtime_failure(cls, exc: ProcessingError) -> bool:
        text = f"{exc}\n{json.dumps(getattr(exc, 'metadata', {}) or {}, ensure_ascii=False)}".lower()
        gpu_error_tokens = (
            "cuda",
            "gpu",
            "opengl",
            "no device",
            "no cuda-capable device",
            "invalid device function",
            "device-side assert",
            "cudart",
            "nvidia",
            "egl",
            "context",
        )
        return any(token in text for token in gpu_error_tokens)

    def _profile_allows_gpu_cpu_fallback(self) -> bool:
        return self.profile in {"conservative", "balanced"}

    @staticmethod
    def _summarize_stage_timings(stage_timings: dict[str, object]) -> dict[str, float]:
        def _safe_float(value: object) -> float:
            try:
                return max(0.0, float(value))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return 0.0

        model_converter_total = sum(
            _safe_float(value)
            for key, value in stage_timings.items()
            if str(key).startswith("model_converter")
        )
        dense_total = sum(
            _safe_float(stage_timings.get(stage_name))
            for stage_name in ("image_undistorter", "patch_match_stereo", "stereo_fusion", "poisson_mesher")
        )

        return {
            "feature_extractor": round(_safe_float(stage_timings.get("feature_extractor")), 3),
            "matcher": round(_safe_float(stage_timings.get("exhaustive_matcher")), 3),
            "mapper": round(_safe_float(stage_timings.get("mapper")), 3),
            "model_converter": round(model_converter_total, 3),
            "dense": round(dense_total, 3),
        }

    def _detect_mapper_insufficient_registered_images_error(
        self,
        project_id: str,
        stdout_text: str | None,
        stderr_text: str | None,
        stdout_path: Path,
        stderr_path: Path,
    ) -> ProcessingError | None:
        combined_output = "\n".join(
            part for part in (self._normalize_output(stderr_text), self._normalize_output(stdout_text)) if part
        )
        if not self._MAPPER_INSUFFICIENT_REGISTERED_IMAGES_PATTERN.search(combined_output):
            return None

        registered_image_count = self._extract_registered_image_count_from_mapper_output(combined_output)
        logger.warning(
            "Project %s | Mapper failed because COLMAP could not keep at least two registered images. "
            "registered_image_count=%s stdout_log=%s stderr_log=%s",
            project_id,
            registered_image_count,
            stdout_path,
            stderr_path,
        )
        return self._build_insufficient_registered_images_error(
            registered_image_count=registered_image_count,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    def _validate_registered_image_count(self, images_path: Path) -> int:
        if not images_path.exists() or not images_path.is_file():
            raise ProcessingError(
                "COLMAP exporto el modelo sparse sin el archivo images.txt necesario para validar las imagenes registradas."
            )

        registered_image_count = self._load_registered_image_count(images_path)
        if registered_image_count >= 2:
            return registered_image_count

        logger.warning(
            "Sparse model validation failed: registered_image_count=%s images_path=%s",
            registered_image_count,
            images_path,
        )
        raise self._build_insufficient_registered_images_error(
            registered_image_count=registered_image_count,
        )

    @classmethod
    def _build_insufficient_registered_images_error(
        cls,
        registered_image_count: int | None,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
    ) -> ProcessingError:
        metadata: dict[str, object] = {
            "current_stage": cls._INSUFFICIENT_REGISTERED_IMAGES_STAGE,
            "reason_code": cls._INSUFFICIENT_REGISTERED_IMAGES_REASON_CODE,
            "status_message": cls._INSUFFICIENT_REGISTERED_IMAGES_MESSAGE,
            "failed_command": "mapper",
        }
        if registered_image_count is not None:
            metadata["registered_image_count"] = registered_image_count
            metadata["metrics"] = {"registered_image_count": registered_image_count}
        if stdout_path is not None or stderr_path is not None:
            metadata["logs"] = {
                "stdout_path": str(stdout_path) if stdout_path is not None else None,
                "stderr_path": str(stderr_path) if stderr_path is not None else None,
            }

        return ProcessingError(
            cls._INSUFFICIENT_REGISTERED_IMAGES_MESSAGE,
            reason_code=cls._INSUFFICIENT_REGISTERED_IMAGES_REASON_CODE,
            current_stage=cls._INSUFFICIENT_REGISTERED_IMAGES_STAGE,
            metadata=metadata,
            allow_fallback=False,
            retryable=False,
        )

    @classmethod
    def _extract_registered_image_count_from_mapper_output(cls, output_text: str) -> int | None:
        num_images_match = cls._MAPPER_NUM_IMAGES_PATTERN.search(output_text or "")
        if num_images_match:
            try:
                return int(num_images_match.group(1))
            except (TypeError, ValueError):
                pass

        registered_frames = cls._MAPPER_REGISTERED_FRAME_PATTERN.findall(output_text or "")
        if registered_frames:
            try:
                return int(registered_frames[-1])
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _validate_output_file(output_path: Path, description: str) -> Path:
        if not output_path.exists() or not output_path.is_file():
            raise ProcessingError(f"COLMAP no genero {description} esperado: {output_path}")
        if output_path.stat().st_size <= 0:
            raise ProcessingError(f"COLMAP genero {description}, pero el archivo esta vacio: {output_path}")
        return output_path

    def _validate_image_undistorter_output(self, dense_dir: Path) -> dict[str, str]:
        required_dirs = {
            "dense_images_dir": dense_dir / "images",
            "dense_sparse_dir": dense_dir / "sparse",
            "dense_stereo_dir": dense_dir / "stereo",
        }
        missing = [name for name, path in required_dirs.items() if not path.exists() or not path.is_dir()]
        if missing:
            raise ProcessingError(
                "COLMAP termino el paso 'image_undistorter', pero faltan directorios obligatorios del workspace denso. "
                f"Faltantes: {', '.join(missing)}."
            )
        return {name: str(path) for name, path in required_dirs.items()}

    def _validate_patch_match_output(self, dense_dir: Path) -> Path:
        depth_maps_dir = dense_dir / "stereo" / "depth_maps"
        if not depth_maps_dir.exists() or not depth_maps_dir.is_dir():
            raise ProcessingError(
                "COLMAP termino el paso 'patch_match_stereo', pero no genero la carpeta 'stereo/depth_maps'."
            )

        depth_map_files = [path for path in depth_maps_dir.rglob("*") if path.is_file()]
        if not depth_map_files:
            raise ProcessingError(
                "COLMAP termino el paso 'patch_match_stereo', pero no produjo mapas de profundidad utilizables."
            )
        return depth_maps_dir

    @staticmethod
    def _import_trimesh() -> object:
        try:
            return importlib.import_module("trimesh")
        except ImportError as exc:
            raise ProcessingError(
                "La exportacion de la malla final requiere la dependencia 'trimesh'. Ejecuta 'pip install -r requirements.txt'."
            ) from exc

    @staticmethod
    def _load_trimesh_asset(
        trimesh_module: object,
        asset_path: Path,
        file_type: str,
        force: str | None = None,
    ) -> object:
        try:
            if force is None:
                return trimesh_module.load(str(asset_path), file_type=file_type)
            return trimesh_module.load(str(asset_path), file_type=file_type, force=force)
        except Exception as exc:
            raise ProcessingError(
                f"No se pudo cargar con trimesh el artefacto '{asset_path.name}' ({file_type}): {exc}"
            ) from exc

    @staticmethod
    def _detect_scipy_available() -> bool:
        try:
            importlib.import_module("scipy")
            return True
        except ImportError:
            return False

    @classmethod
    def _validate_mesh_asset(cls, mesh_asset: object, asset_name: str) -> MeshArtifactSummary:
        summary = cls._extract_mesh_counts(mesh_asset)
        if summary.vertex_count <= 0:
            raise ProcessingError(f"La malla '{asset_name}' no contiene vertices utilizables.")
        if summary.face_count <= 0:
            raise ProcessingError(
                f"La malla '{asset_name}' no contiene caras. El resultado seria una nube de puntos, no un mesh solido."
            )
        return summary

    def _build_sparse_fallback_mesh(
        self,
        trimesh_module: object,
        sparse_points_ply_path: Path,
        sparse_model_dir: Path,
        resolved_binary: str,
        project_id: str,
        logs_dir: Path,
        progress_callback: ReconstructionProgressCallback | None,
        command_traces: list[ColmapCommandTrace],
        stage_timings: dict[str, float],
    ) -> tuple[object, str, MeshArtifactSummary, int, dict[str, object]]:
        cloud = self._load_trimesh_asset(trimesh_module, sparse_points_ply_path, "ply")
        point_count = self._safe_len(getattr(cloud, "vertices", ()))
        if point_count <= 0:
            raise ProcessingError(
                f"El archivo sparse '{sparse_points_ply_path.name}' no contiene vertices para generar un mesh aproximado."
            )

        scipy_available = self._detect_scipy_available()
        diagnostics: dict[str, object] = {
            "scipy_available": scipy_available,
            "attempted_method": "delaunay_mesher_sparse",
            "final_method": None,
            "convex_hull_exception": None,
            "delaunay_exception": None,
            "delaunay_mesh_path": None,
            "color_transfer_method": None,
            "shape_diagnostics": None,
        }

        logger.warning(
            "Project %s | Using sparse reconstruction only to generate an approximate mesh.",
            project_id,
        )
        logger.info(
            "Project %s | Loading sparse point cloud for approximate mesh generation: %s (points=%s)",
            project_id,
            sparse_points_ply_path,
            point_count,
        )
        logger.info(
            "Project %s | Sparse fallback diagnostics: scipy_available=%s attempted_method=%s",
            project_id,
            scipy_available,
            diagnostics["attempted_method"],
        )
        logger.info(
            "Project %s | Attempting sparse fallback mesh with method=%s",
            project_id,
            diagnostics["attempted_method"],
        )

        delaunay_mesh_path = sparse_model_dir / "meshed-delaunay-sparse.ply"
        try:
            delaunay_trace = self._run_command(
                project_id,
                "delaunay_mesher_sparse",
                [
                    resolved_binary,
                    "delaunay_mesher",
                    "--input_path",
                    str(sparse_model_dir),
                    "--input_type",
                    "sparse",
                    "--output_path",
                    str(delaunay_mesh_path),
                    "--DelaunayMeshing.max_proj_dist",
                    "2",
                    "--DelaunayMeshing.quality_regularization",
                    "0",
                    "--DelaunayMeshing.max_side_length_factor",
                    "4",
                ],
                logs_dir,
                progress_callback,
                0.74,
                "Ejecutando delaunay_mesher (fallback sparse).",
            )
            command_traces.append(delaunay_trace)
            stage_timings["delaunay_mesher_sparse"] = delaunay_trace.duration_seconds
            self._validate_output_file(delaunay_mesh_path, "la malla sparse 'meshed-delaunay-sparse.ply'")
            mesh_asset = self._load_trimesh_asset(trimesh_module, delaunay_mesh_path, "ply", "mesh")
            summary = self._validate_mesh_asset(mesh_asset, "mesh aproximado (delaunay_mesher_sparse)")
            diagnostics["final_method"] = "delaunay_mesher_sparse"
            diagnostics["delaunay_mesh_path"] = str(delaunay_mesh_path)
            diagnostics["color_transfer_method"] = self._transfer_sparse_colors_to_mesh(mesh_asset, cloud)
            diagnostics["shape_diagnostics"] = self._compute_mesh_shape_diagnostics(mesh_asset)
            logger.info(
                "Project %s | Approximate sparse mesh generated successfully with delaunay_mesher_sparse. vertices=%s faces=%s",
                project_id,
                summary.vertex_count,
                summary.face_count,
            )
            return mesh_asset, "delaunay_mesher_sparse", summary, point_count, diagnostics
        except Exception as exc:
            diagnostics["delaunay_exception"] = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Project %s | delaunay_mesher_sparse failed for sparse fallback. scipy_available=%s exception=%s. Falling back to convex_hull.",
                project_id,
                scipy_available,
                diagnostics["delaunay_exception"],
            )

        mesh_method = "convex_hull"
        try:
            mesh_asset = cloud.convex_hull
            diagnostics["final_method"] = mesh_method
        except Exception as exc:
            diagnostics["convex_hull_exception"] = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Project %s | convex_hull failed for sparse fallback. scipy_available=%s exception=%s. Falling back to bounding_box.",
                project_id,
                scipy_available,
                diagnostics["convex_hull_exception"],
            )
            mesh_method = "bounding_box"
            diagnostics["final_method"] = mesh_method
            try:
                mesh_asset = cloud.bounding_box
            except Exception as fallback_exc:
                raise ProcessingError(
                    "No se pudo generar una malla aproximada desde la nube sparse ni con delaunay_mesher_sparse, convex_hull ni bounding_box. "
                    f"Error delaunay: {diagnostics['delaunay_exception']}. Error convex_hull: {exc}. Error bounding_box: {fallback_exc}."
                ) from fallback_exc

        summary = self._validate_mesh_asset(mesh_asset, f"mesh aproximado ({mesh_method})")
        diagnostics["color_transfer_method"] = self._transfer_sparse_colors_to_mesh(mesh_asset, cloud)
        diagnostics["shape_diagnostics"] = self._compute_mesh_shape_diagnostics(mesh_asset)
        logger.info(
            "Project %s | Approximate sparse mesh generated successfully. scipy_available=%s attempted_method=%s final_method=%s vertices=%s faces=%s",
            project_id,
            scipy_available,
            diagnostics["attempted_method"],
            diagnostics["final_method"],
            summary.vertex_count,
            summary.face_count,
        )
        return mesh_asset, mesh_method, summary, point_count, diagnostics

    @staticmethod
    def _transfer_sparse_colors_to_mesh(mesh_asset: object, sparse_cloud: object) -> str | None:
        try:
            import numpy as np
        except ImportError:
            return None

        mesh_vertices = np.asarray(getattr(mesh_asset, "vertices", ()))
        cloud_vertices = np.asarray(getattr(sparse_cloud, "vertices", ()))
        if mesh_vertices.size == 0 or cloud_vertices.size == 0:
            return None

        cloud_colors = getattr(sparse_cloud, "colors", None)
        if cloud_colors is None:
            cloud_visual = getattr(sparse_cloud, "visual", None)
            cloud_colors = getattr(cloud_visual, "vertex_colors", None)
        if cloud_colors is None:
            return None

        colors_array = np.asarray(cloud_colors)
        if colors_array.ndim != 2 or colors_array.shape[0] <= 0:
            return None
        if colors_array.shape[1] >= 4:
            colors_array = colors_array[:, :4]
        elif colors_array.shape[1] == 3:
            alpha = np.full((colors_array.shape[0], 1), 255, dtype=colors_array.dtype)
            colors_array = np.concatenate((colors_array, alpha), axis=1)
        else:
            return None

        if colors_array.dtype != np.uint8:
            colors_array = np.clip(colors_array, 0, 255).astype(np.uint8)

        mesh_visual = getattr(mesh_asset, "visual", None)
        existing_colors = getattr(mesh_visual, "vertex_colors", None)
        if existing_colors is not None:
            existing_array = np.asarray(existing_colors)
            if existing_array.ndim == 2 and existing_array.shape[0] == mesh_vertices.shape[0]:
                unique_colors = np.unique(existing_array[:, :3], axis=0)
                if unique_colors.shape[0] > 1:
                    return "mesh_existing"

        mapped_colors = None
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(cloud_vertices)
            _, nearest_indices = tree.query(mesh_vertices, k=1)
            mapped_colors = colors_array[np.asarray(nearest_indices, dtype=np.int64)]
            transfer_method = "nearest_neighbor"
        except Exception:
            repeats = max(1, int((mesh_vertices.shape[0] + colors_array.shape[0] - 1) / colors_array.shape[0]))
            mapped_colors = np.tile(colors_array, (repeats, 1))[: mesh_vertices.shape[0]]
            transfer_method = "cyclic"

        try:
            mesh_asset.visual.vertex_colors = mapped_colors
            return transfer_method
        except Exception:
            return None

    @classmethod
    def _compute_mesh_shape_diagnostics(cls, mesh_asset: object) -> dict[str, object]:
        diagnostics: dict[str, object] = {
            "bbox_extents": None,
            "extent_ratio_max_min": None,
            "bbox_volume": None,
            "mesh_volume": None,
            "mesh_volume_to_bbox_volume_ratio": None,
            "mesh_surface_area": None,
            "is_watertight": None,
        }

        extents: list[float] | None = None
        bounds = getattr(mesh_asset, "bounds", None)
        if bounds is not None:
            try:
                import numpy as np

                bounds_array = np.asarray(bounds, dtype=float)
                if bounds_array.shape == (2, 3):
                    extents_array = np.abs(bounds_array[1] - bounds_array[0])
                    extents = [float(item) for item in extents_array.tolist()]
            except Exception:
                extents = None

        if extents is None:
            try:
                bbox = getattr(mesh_asset, "bounding_box", None)
                bbox_extents = getattr(bbox, "extents", None)
                if bbox_extents is not None:
                    extents = [float(item) for item in bbox_extents]
            except Exception:
                extents = None

        finite_extents = [value for value in (extents or []) if cls._to_finite_float(value) is not None]
        if len(finite_extents) == 3:
            diagnostics["bbox_extents"] = [round(value, 6) for value in finite_extents]
            positive = [value for value in finite_extents if value > 1e-9]
            if positive:
                extent_ratio = max(positive) / max(min(positive), 1e-9)
                diagnostics["extent_ratio_max_min"] = round(extent_ratio, 6)
                bbox_volume = positive[0] * positive[1] * positive[2] if len(positive) == 3 else None
                if bbox_volume is not None:
                    diagnostics["bbox_volume"] = round(bbox_volume, 6)

        mesh_volume = cls._to_finite_float(getattr(mesh_asset, "volume", None))
        mesh_area = cls._to_finite_float(getattr(mesh_asset, "area", None))
        is_watertight = getattr(mesh_asset, "is_watertight", None)
        diagnostics["mesh_surface_area"] = round(mesh_area, 6) if mesh_area is not None else None
        diagnostics["is_watertight"] = bool(is_watertight) if isinstance(is_watertight, bool) else None
        if mesh_volume is not None:
            diagnostics["mesh_volume"] = round(abs(mesh_volume), 6)

        bbox_volume_value = cls._to_finite_float(diagnostics.get("bbox_volume"))
        if mesh_volume is not None and bbox_volume_value is not None and bbox_volume_value > 1e-9:
            diagnostics["mesh_volume_to_bbox_volume_ratio"] = round(abs(mesh_volume) / bbox_volume_value, 6)

        return diagnostics

    @staticmethod
    def _to_finite_float(value: object) -> float | None:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

        if numeric != numeric:  # NaN
            return None
        if numeric in {float("inf"), float("-inf")}:
            return None
        return numeric

    @staticmethod
    def _export_mesh_asset(project_id: str, mesh_asset: object, output_path: Path, file_type: str) -> None:
        logger.info(
            "Project %s | Exporting final mesh to %s: %s",
            project_id,
            file_type.upper(),
            output_path,
        )
        if str(file_type).strip().lower() == "glb":
            ColmapReconstructionEngine._prepare_mesh_visual_for_glb(project_id, mesh_asset)
        try:
            payload = mesh_asset.export(file_type=file_type)
        except TypeError:
            payload = mesh_asset.export(file_type)
        except Exception as exc:
            raise ProcessingError(
                f"No se pudo exportar la malla final a {file_type.upper()} usando trimesh: {exc}"
            ) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, bytes):
            output_path.write_bytes(payload)
        else:
            output_path.write_text(str(payload), encoding="utf-8")

        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise ProcessingError(
                f"La exportacion a {file_type.upper()} finalizo sin generar un archivo valido: {output_path}"
            )

    @staticmethod
    def _prepare_mesh_visual_for_glb(project_id: str, mesh_asset: object) -> None:
        try:
            visual = getattr(mesh_asset, "visual", None)
            visual_kind = type(visual).__name__ if visual is not None else "None"
            material = getattr(visual, "material", None) if visual is not None else None
            material_type = type(material).__name__ if material is not None else "None"
            vertex_colors = getattr(visual, "vertex_colors", None) if visual is not None else None
            shape = getattr(vertex_colors, "shape", None)
            dtype = getattr(vertex_colors, "dtype", None)
            logger.info(
                "Project %s | GLB visual audit before export. visual_kind=%s material_type=%s vertex_colors_shape=%s vertex_colors_dtype=%s",
                project_id,
                visual_kind,
                material_type,
                shape,
                dtype,
            )
            normalized = ColmapReconstructionEngine._normalize_vertex_colors_for_glb(
                vertex_colors=vertex_colors,
                vertex_count=ColmapReconstructionEngine._safe_len(getattr(mesh_asset, "vertices", ())),
            )
            if normalized is not None:
                visual.vertex_colors = normalized
                logger.info(
                    "Project %s | GLB vertex colors normalized successfully. shape=%s dtype=%s",
                    project_id,
                    getattr(normalized, "shape", None),
                    getattr(normalized, "dtype", None),
                )
        except Exception as exc:
            logger.warning(
                "Project %s | Unable to normalize visual colors for GLB export: %s",
                project_id,
                exc,
            )

    @staticmethod
    def _normalize_vertex_colors_for_glb(*, vertex_colors: object, vertex_count: int):
        if vertex_colors is None or vertex_count <= 0:
            return None
        try:
            import numpy as np
        except Exception:
            return None
        try:
            colors = np.asarray(vertex_colors)
            if colors.size == 0:
                return None
            if colors.ndim == 1:
                if colors.shape[0] in {3, 4}:
                    colors = np.tile(colors.reshape(1, -1), (vertex_count, 1))
                else:
                    return None
            if colors.ndim != 2:
                return None
            if colors.shape[0] != vertex_count:
                if colors.shape[0] == 1:
                    colors = np.tile(colors, (vertex_count, 1))
                else:
                    repeat = int(np.ceil(vertex_count / max(1, colors.shape[0])))
                    colors = np.tile(colors, (repeat, 1))[:vertex_count]
            if colors.shape[1] == 3:
                alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
                colors = np.concatenate((colors, alpha), axis=1)
            elif colors.shape[1] > 4:
                colors = colors[:, :4]
            elif colors.shape[1] < 3:
                return None

            if np.issubdtype(colors.dtype, np.floating):
                maxv = float(np.max(colors)) if colors.size else 0.0
                if maxv <= 1.0:
                    colors = colors * 255.0
                colors = np.clip(colors, 0, 255).astype(np.uint8)
            else:
                colors = np.clip(colors, 0, 255).astype(np.uint8)
            return colors
        except Exception:
            return None

    @classmethod
    def _validate_exported_glb(cls, trimesh_module: object, glb_model_path: Path) -> MeshArtifactSummary:
        cls._validate_output_file(glb_model_path, "el archivo GLB final")
        exported_glb = cls._load_trimesh_asset(trimesh_module, glb_model_path, "glb", "scene")
        summary = cls._validate_mesh_asset(exported_glb, glb_model_path.name)
        logger.info(
            "Validated final GLB mesh successfully. path=%s vertices=%s faces=%s",
            glb_model_path,
            summary.vertex_count,
            summary.face_count,
        )
        return summary

    @classmethod
    def _extract_mesh_counts(cls, mesh_asset: object) -> MeshArtifactSummary:
        geometry = getattr(mesh_asset, "geometry", None)
        if isinstance(geometry, dict):
            vertex_count = sum(cls._safe_len(getattr(item, "vertices", ())) for item in geometry.values())
            face_count = sum(cls._safe_len(getattr(item, "faces", ())) for item in geometry.values())
            return MeshArtifactSummary(vertex_count=vertex_count, face_count=face_count)

        return MeshArtifactSummary(
            vertex_count=cls._safe_len(getattr(mesh_asset, "vertices", ())),
            face_count=cls._safe_len(getattr(mesh_asset, "faces", ())),
        )

    @staticmethod
    def _safe_len(values: object) -> int:
        try:
            return len(values)
        except TypeError:
            return 0

    def _validate_mapper_output(
        self,
        *,
        sparse_dir: Path,
        resolved_binary: str,
        project_id: str,
    ) -> Path:
        available_dirs = sorted(
            [path for path in sparse_dir.iterdir() if path.is_dir()],
            key=self._sparse_model_sort_key,
        ) if sparse_dir.exists() else []
        if not available_dirs:
            raise ProcessingError(
                "COLMAP termino el paso 'mapper', pero no genero ningun submodelo en la carpeta sparse."
            )

        candidates: list[Path] = []
        missing_by_dir: dict[str, list[str]] = {}
        for model_dir in available_dirs:
            missing_files = self._missing_sparse_model_files(model_dir)
            if missing_files:
                missing_by_dir[model_dir.name] = missing_files
                continue
            candidates.append(model_dir)

        if not candidates:
            details = ", ".join(
                f"{name}: faltan {', '.join(values)}"
                for name, values in missing_by_dir.items()
            ) or "sin detalle"
            raise ProcessingError(
                "COLMAP genero subdirectorios sparse, pero faltan archivos de reconstruccion requeridos. "
                f"Detalle: {details}."
            )

        if len(candidates) == 1:
            return candidates[0]

        scored_candidates: list[tuple[int, int, Path]] = []
        for candidate in candidates:
            registered_images, points = self._analyze_sparse_model(candidate, resolved_binary)
            scored_candidates.append((registered_images, points, candidate))

        best_registered, best_points, selected = max(
            scored_candidates,
            key=lambda item: (item[0], item[1], self._sparse_model_sort_key(item[2])),
        )
        logger.info(
            "Project %s | Selected sparse model directory '%s' among %s candidates. "
            "registered_images=%s points=%s",
            project_id,
            selected.name,
            len(candidates),
            best_registered,
            best_points,
        )
        return selected

    @classmethod
    def _sparse_model_sort_key(cls, model_dir: Path) -> tuple[int, int | str]:
        try:
            return (0, int(model_dir.name))
        except (TypeError, ValueError):
            return (1, model_dir.name)

    def _analyze_sparse_model(self, model_dir: Path, resolved_binary: str) -> tuple[int, int]:
        default_points = int((model_dir / "points3D.bin").stat().st_size) if (model_dir / "points3D.bin").exists() else 0
        default_registered = int((model_dir / "images.bin").stat().st_size) if (model_dir / "images.bin").exists() else 0
        try:
            completed = subprocess.run(
                [resolved_binary, "model_analyzer", "--path", str(model_dir)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=min(self.timeout_seconds, 120),
                check=False,
            )
        except Exception:
            return default_registered, default_points

        combined = f"{completed.stdout or ''}\n{completed.stderr or ''}"
        registered = default_registered
        points = default_points
        registered_match = self._MODEL_ANALYZER_REGISTERED_IMAGES_PATTERN.search(combined)
        if registered_match is not None:
            try:
                registered = int(registered_match.group(1))
            except (TypeError, ValueError):
                pass
        points_match = self._MODEL_ANALYZER_POINTS_PATTERN.search(combined)
        if points_match is not None:
            try:
                points = int(points_match.group(1))
            except (TypeError, ValueError):
                pass
        return registered, points

    @staticmethod
    def _missing_sparse_model_files(model_dir: Path) -> list[str]:
        required_prefixes = ("cameras", "images", "points3D")
        missing: list[str] = []
        for prefix in required_prefixes:
            if not any((model_dir / f"{prefix}{suffix}").exists() for suffix in (".bin", ".txt")):
                missing.append(prefix)
        return missing

    @staticmethod
    def _load_sparse_points(points_path: Path) -> list[SparsePoint]:
        if not points_path.exists():
            raise ProcessingError(f"No se encontro el archivo de puntos sparse exportado por COLMAP: {points_path}")

        points: list[SparsePoint] = []
        for raw_line in points_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            try:
                points.append(
                    SparsePoint(
                        x=float(parts[1]),
                        y=float(parts[2]),
                        z=float(parts[3]),
                        r=int(parts[4]),
                        g=int(parts[5]),
                        b=int(parts[6]),
                        error=float(parts[7]),
                    )
                )
            except ValueError:
                continue

        return points

    @staticmethod
    def _load_registered_image_count(images_path: Path) -> int:
        if not images_path.exists():
            return 0
        non_comment_lines = [
            line.strip()
            for line in images_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        return len(non_comment_lines) // 2

    @staticmethod
    def _load_camera_count(cameras_path: Path) -> int:
        if not cameras_path.exists():
            return 0
        return len(
            [
                line.strip()
                for line in cameras_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            ]
        )

    @staticmethod
    def _write_obj_point_cloud(model_path: Path, points: list[SparsePoint], project_id: str) -> None:
        lines = [
            "# OBJ generado desde COLMAP sparse reconstruction",
            f"# project_id={project_id}",
            f"# point_count={len(points)}",
            "o SparsePointCloud",
        ]
        for point in points:
            lines.append(f"v {point.x} {point.y} {point.z}")
        for index in range(len(points)):
            lines.append(f"p {index + 1}")
        model_path.write_text("\n".join(lines), encoding="utf-8")

    def _write_glb_point_cloud(self, model_path: Path, points: list[SparsePoint], project_id: str) -> str:
        if not points:
            raise ProcessingError("No hay puntos disponibles para exportar el GLB sparse.")
        try:
            trimesh_module = self._import_trimesh()
            sphere_points = self._sample_sparse_points(points, max_points=1500)
            centered_points, min_bounds, max_bounds = self._center_and_scale_points(sphere_points)
            radius = self._estimate_sparse_sphere_radius(min_bounds=min_bounds, max_bounds=max_bounds)
            sphere_meshes: list[object] = []
            for point in centered_points:
                try:
                    sphere = trimesh_module.creation.icosphere(subdivisions=1, radius=radius)
                    sphere.apply_translation((point.x, point.y, point.z))
                    sphere.visual.vertex_colors = [point.r, point.g, point.b, 255]
                    sphere_meshes.append(sphere)
                except Exception:
                    continue
            if sphere_meshes:
                combined = trimesh_module.util.concatenate(sphere_meshes)
                payload = combined.export(file_type="glb")
                if isinstance(payload, bytes):
                    model_path.write_bytes(payload)
                else:
                    model_path.write_text(str(payload), encoding="utf-8")
                return "point_spheres"
        except Exception:
            pass

        centered_all_points, _, _ = self._center_and_scale_points(points)
        positions = struct.pack(
            "<" + "f" * (len(points) * 3),
            *[component for point in centered_all_points for component in (point.x, point.y, point.z)],
        )
        colors = struct.pack(
            "<" + "B" * (len(points) * 4),
            *[
                component
                for point in points
                for component in (point.r, point.g, point.b, 255)
            ],
        )

        positions_offset = 0
        colors_offset = len(positions)
        bin_payload = positions + colors
        if len(bin_payload) % 4:
            bin_payload += b"\x00" * (4 - (len(bin_payload) % 4))

        json_payload = {
            "asset": {
                "version": "2.0",
                "generator": "ColmapReconstructionEngine",
            },
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [
                {
                    "name": "SparsePointCloud",
                    "primitives": [
                        {
                            "attributes": {
                                "POSITION": 0,
                                "COLOR_0": 1,
                            },
                            "mode": 0,
                        }
                    ],
                }
            ],
            "buffers": [{"byteLength": len(bin_payload)}],
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteOffset": positions_offset,
                    "byteLength": len(positions),
                    "target": 34962,
                },
                {
                    "buffer": 0,
                    "byteOffset": colors_offset,
                    "byteLength": len(colors),
                    "target": 34962,
                },
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": len(points),
                    "type": "VEC3",
                    "min": self._min_bounds(points),
                    "max": self._max_bounds(points),
                },
                {
                    "bufferView": 1,
                    "componentType": 5121,
                    "normalized": True,
                    "count": len(points),
                    "type": "VEC4",
                },
            ],
            "extras": {
                "projectId": project_id,
                "pointCount": len(points),
                "representation": "sparse-point-cloud",
            },
        }

        json_bytes = json.dumps(json_payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(json_bytes) % 4:
            json_bytes += b" " * (4 - (len(json_bytes) % 4))

        total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_payload)
        header = struct.pack("<4sII", b"glTF", 2, total_length)
        json_chunk = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
        bin_chunk = struct.pack("<I4s", len(bin_payload), b"BIN\x00") + bin_payload
        model_path.write_bytes(header + json_chunk + bin_chunk)
        return "point_cloud"

    @staticmethod
    def _sample_sparse_points(points: list[SparsePoint], *, max_points: int) -> list[SparsePoint]:
        if len(points) <= max_points:
            return list(points)
        stride = max(1, len(points) // max_points)
        sampled = points[::stride]
        if len(sampled) > max_points:
            sampled = sampled[:max_points]
        return sampled

    @staticmethod
    def _center_and_scale_points(points: list[SparsePoint]) -> tuple[list[SparsePoint], list[float], list[float]]:
        min_x = min(point.x for point in points)
        min_y = min(point.y for point in points)
        min_z = min(point.z for point in points)
        max_x = max(point.x for point in points)
        max_y = max(point.y for point in points)
        max_z = max(point.z for point in points)
        center_x = (min_x + max_x) * 0.5
        center_y = (min_y + max_y) * 0.5
        center_z = (min_z + max_z) * 0.5
        extent_x = max_x - min_x
        extent_y = max_y - min_y
        extent_z = max_z - min_z
        max_extent = max(extent_x, extent_y, extent_z, 1e-6)
        scale = 2.0 / max_extent
        normalized = [
            SparsePoint(
                x=(point.x - center_x) * scale,
                y=(point.y - center_y) * scale,
                z=(point.z - center_z) * scale,
                r=point.r,
                g=point.g,
                b=point.b,
                error=point.error,
            )
            for point in points
        ]
        return normalized, [min_x, min_y, min_z], [max_x, max_y, max_z]

    @staticmethod
    def _estimate_sparse_sphere_radius(*, min_bounds: list[float], max_bounds: list[float]) -> float:
        dx = max_bounds[0] - min_bounds[0]
        dy = max_bounds[1] - min_bounds[1]
        dz = max_bounds[2] - min_bounds[2]
        diag = max((dx * dx + dy * dy + dz * dz) ** 0.5, 1e-6)
        return max(0.01, min(0.05, diag * 0.01))

    @staticmethod
    def _write_command_logs(logs_dir: Path, name: str, stdout_text: str | bytes | None, stderr_text: str | bytes | None) -> tuple[Path, Path]:
        logs_dir.mkdir(parents=True, exist_ok=True)
        safe_name = name.replace(" ", "_")
        stdout_path = logs_dir / f"{safe_name}.stdout.log"
        stderr_path = logs_dir / f"{safe_name}.stderr.log"
        stdout_value = ColmapReconstructionEngine._normalize_output(stdout_text)
        stderr_value = ColmapReconstructionEngine._normalize_output(stderr_text)
        stdout_path.write_text(stdout_value, encoding="utf-8")
        stderr_path.write_text(stderr_value, encoding="utf-8")
        return stdout_path, stderr_path

    @staticmethod
    def _normalize_output(text: str | bytes | None) -> str:
        if text is None:
            return ""
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="replace")
        return str(text)

    @staticmethod
    def _tail(text: str | None) -> str:
        if not text:
            return ""
        normalized = text.strip()
        if len(normalized) <= ColmapReconstructionEngine._MAX_LOG_TAIL:
            return normalized
        return normalized[-ColmapReconstructionEngine._MAX_LOG_TAIL :]

    @staticmethod
    def _notify_progress(
        progress_callback: ReconstructionProgressCallback | None,
        payload: dict[str, object],
    ) -> None:
        if progress_callback is None:
            return
        progress_callback({key: value for key, value in payload.items() if value is not None})

    @staticmethod
    def _min_bounds(points: list[SparsePoint]) -> list[float]:
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        zs = [point.z for point in points]
        return [min(xs), min(ys), min(zs)]

    @staticmethod
    def _max_bounds(points: list[SparsePoint]) -> list[float]:
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        zs = [point.z for point in points]
        return [max(xs), max(ys), max(zs)]
