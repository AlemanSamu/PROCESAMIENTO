from __future__ import annotations

import importlib
import json
import logging
import re
import struct
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

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

    def __init__(
        self,
        colmap_binary: str = "colmap",
        timeout_seconds: int = 1800,
        use_gpu: bool = False,
        camera_model: str = "SIMPLE_RADIAL",
        single_camera: bool = True,
    ) -> None:
        self.colmap_binary = (colmap_binary or "colmap").strip() or "colmap"
        self.timeout_seconds = max(timeout_seconds, 30)
        self.use_gpu = use_gpu
        self.camera_model = camera_model
        self.single_camera = single_camera
        self._detected_binary: str | None = None
        self._detection_attempted = False
        self._feature_extraction_gpu_option: str | None = None
        self._feature_matching_gpu_option: str | None = None
        self._dense_reconstruction_supported: bool | None = None

    @property
    def detected_binary(self) -> str | None:
        return self._detected_binary

    def is_available(self) -> bool:
        return self.detect_binary() is not None

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
                "del ejecutable, o agrega 'colmap', 'colmap.exe' o 'colmap.bat' al PATH del sistema."
            )

        image_paths = self._collect_images(images_dir)
        if len(image_paths) < 2:
            raise ProcessingError("COLMAP requiere al menos 2 imagenes para intentar la reconstruccion sparse.")
        if output_format not in {OutputFormat.GLB, OutputFormat.OBJ}:
            raise ProcessingError(f"Formato de salida no soportado para COLMAP: {output_format.value}.")

        trimesh_module = self._import_trimesh()
        feature_extraction_gpu_option = self._get_feature_extraction_gpu_option(resolved_binary)
        feature_matching_gpu_option = self._get_feature_matching_gpu_option(resolved_binary)

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
            },
        )

        command_traces: list[ColmapCommandTrace] = []
        warnings: list[str] = []
        stage_timings: dict[str, float] = {}

        feature_trace = self._run_command(
            project_id,
            "feature_extractor",
            [
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
                "1" if self.use_gpu else "0",
            ],
            logs_dir,
            progress_callback,
            0.10,
            "Ejecutando feature_extractor.",
        )
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

        matcher_trace = self._run_command(
            project_id,
            "exhaustive_matcher",
            [
                resolved_binary,
                "exhaustive_matcher",
                "--database_path",
                str(database_path),
                f"--{feature_matching_gpu_option}",
                "1" if self.use_gpu else "0",
            ],
            logs_dir,
            progress_callback,
            0.22,
            "Ejecutando exhaustive_matcher.",
        )
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
        sparse_model_dir = self._validate_mapper_output(sparse_dir)
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

        if not self._supports_dense_reconstruction(resolved_binary):
            warning = (
                "COLMAP no tiene soporte CUDA. Se omitio la reconstruccion densa y se genero una malla aproximada "
                "usando solo la reconstruccion sparse."
            )
            warnings.append(warning)
            logger.warning("Project %s | %s", project_id, warning)
            self._notify_progress(
                progress_callback,
                {
                    "current_stage": "sparse_mesh_fallback",
                    "progress": 0.68,
                    "status_message": "COLMAP sin CUDA; generando mesh aproximado desde la nube sparse.",
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
                project_id,
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
                "fused_ply_path": None,
                "poisson_mesh_ply": None,
                "logs_dir": str(logs_dir),
            }

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
                },
                "warnings": warnings,
                "colmap_binary": resolved_binary,
                "metadata_path": str(metadata_path),
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

        final_mesh = self._load_trimesh_asset(trimesh_module, poisson_mesh_path, "ply", "mesh")
        mesh_summary = self._validate_mesh_asset(final_mesh, poisson_mesh_path.name)

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
            "mesh_vertex_count": glb_summary.vertex_count,
            "mesh_face_count": glb_summary.face_count,
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
            "reconstruction_type": "dense_photogrammetry_mesh",
            "image_count_processed": len(image_paths),
            "registered_image_count": registered_image_count,
            "camera_count": camera_count,
            "point_count": len(points),
            "mesh_vertex_count": glb_summary.vertex_count,
            "mesh_face_count": glb_summary.face_count,
            "current_stage": "completed",
            "progress": 1.0,
            "status_message": "Reconstruccion completada.",
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
            "warnings": warnings,
            "colmap_binary": resolved_binary,
            "metadata_path": str(metadata_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(
            "Project %s | COLMAP reconstruction completed in %.3fs. registered_images=%s cameras=%s sparse_points=%s mesh_vertices=%s mesh_faces=%s model=%s",
            project_id,
            elapsed_seconds,
            registered_image_count,
            camera_count,
            len(points),
            mesh_summary.vertex_count,
            mesh_summary.face_count,
            model_path,
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

    def _build_binary_candidates(self) -> list[str]:
        configured = self.colmap_binary
        configured_path = Path(configured)
        candidates: list[str] = []
        seen: set[str] = set()

        def add(candidate: str | None) -> None:
            if not candidate:
                return
            normalized = str(candidate).strip()
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
            logger.exception("COLMAP binary disappeared during execution. command=%s", command[0], exc_info=exc)
            raise ProcessingError(
                "No se pudo ejecutar COLMAP. Verifica la ruta del ejecutable configurado en LOCAL3D_COLMAP_PATH "
                "o LOCAL3D_COLMAP_BINARY."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            stdout_path, stderr_path = self._write_command_logs(logs_dir, name, exc.stdout, exc.stderr)
            logger.exception("COLMAP step '%s' timed out after %s seconds", name, self.timeout_seconds, exc_info=exc)
            raise ProcessingError(
                f"COLMAP agoto el tiempo de espera en el paso '{name}' despues de {self.timeout_seconds} segundos. "
                f"Logs: stdout={stdout_path}, stderr={stderr_path}"
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
                f"Logs: stdout={stdout_path}, stderr={stderr_path}"
            )

        logger.info(
            "Project %s | COLMAP stage '%s' completed in %.3fs",
            project_id,
            name,
            duration_seconds,
        )
        return trace

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
        project_id: str,
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
            "attempted_method": "convex_hull",
            "final_method": None,
            "convex_hull_exception": None,
        }

        logger.warning(
            "Project %s | No CUDA detected. Using sparse reconstruction only to generate an approximate mesh.",
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
                    "No se pudo generar una malla aproximada desde la nube sparse ni con convex_hull ni con bounding_box. "
                    f"Error original: {exc}. Error fallback: {fallback_exc}."
                ) from fallback_exc

        summary = self._validate_mesh_asset(mesh_asset, f"mesh aproximado ({mesh_method})")
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
    def _export_mesh_asset(project_id: str, mesh_asset: object, output_path: Path, file_type: str) -> None:
        logger.info(
            "Project %s | Exporting final mesh to %s: %s",
            project_id,
            file_type.upper(),
            output_path,
        )
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

    def _validate_mapper_output(self, sparse_dir: Path) -> Path:
        sparse_model_dir = sparse_dir / "0"
        if not sparse_model_dir.exists() or not sparse_model_dir.is_dir():
            available_dirs = sorted(path.name for path in sparse_dir.iterdir() if path.is_dir()) if sparse_dir.exists() else []
            raise ProcessingError(
                "COLMAP termino el paso 'mapper', pero no genero la carpeta requerida 'sparse/0'. "
                f"Directorios encontrados: {available_dirs or 'ninguno'}."
            )

        missing_files = self._missing_sparse_model_files(sparse_model_dir)
        if missing_files:
            raise ProcessingError(
                "COLMAP genero 'sparse/0', pero faltan archivos de reconstruccion. "
                f"Faltantes: {', '.join(missing_files)}."
            )
        return sparse_model_dir

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

    def _write_glb_point_cloud(self, model_path: Path, points: list[SparsePoint], project_id: str) -> None:
        if not points:
            raise ProcessingError("No hay puntos disponibles para exportar el GLB sparse.")

        positions = struct.pack(
            "<" + "f" * (len(points) * 3),
            *[component for point in points for component in (point.x, point.y, point.z)],
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
