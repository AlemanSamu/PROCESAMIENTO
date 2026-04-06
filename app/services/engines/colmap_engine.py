from __future__ import annotations

import json
import logging
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


class ColmapReconstructionEngine(ReconstructionEngine):
    name = "colmap"
    is_implemented = True
    _MAX_LOG_TAIL = 4000
    _PROBE_TIMEOUT_SECONDS = 10

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
        obj_model_path = output_dir / f"{project_id}_model.obj"
        glb_model_path = output_dir / f"{project_id}_model.glb"
        metadata_path = output_dir / f"{project_id}_colmap_metadata.json"

        logger.info(
            "Project %s | Starting COLMAP reconstruction with %s images. output_dir=%s",
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
            0.12,
            "Ejecutando feature_extractor.",
        )
        command_traces.append(feature_trace)
        stage_timings["feature_extractor"] = feature_trace.duration_seconds
        self._notify_progress(
            progress_callback,
            {
                "current_stage": "feature_extractor",
                "progress": 0.28,
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
            0.36,
            "Ejecutando exhaustive_matcher.",
        )
        command_traces.append(matcher_trace)
        stage_timings["exhaustive_matcher"] = matcher_trace.duration_seconds
        self._notify_progress(
            progress_callback,
            {
                "current_stage": "exhaustive_matcher",
                "progress": 0.52,
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
            0.6,
            "Ejecutando mapper.",
        )
        command_traces.append(mapper_trace)
        stage_timings["mapper"] = mapper_trace.duration_seconds
        sparse_model_dir = self._validate_mapper_output(sparse_dir)
        self._notify_progress(
            progress_callback,
            {
                "current_stage": "mapper",
                "progress": 0.74,
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
            0.82,
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
                0.86,
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

        registered_image_count = self._load_registered_image_count(txt_model_dir / "images.txt")
        camera_count = self._load_camera_count(txt_model_dir / "cameras.txt")

        obj_started_at = time.perf_counter()
        self._write_obj_point_cloud(obj_model_path, points, project_id)
        stage_timings["obj_export"] = round(time.perf_counter() - obj_started_at, 3)

        model_path = obj_model_path
        if output_format == OutputFormat.GLB:
            glb_started_at = time.perf_counter()
            self._write_glb_point_cloud(glb_model_path, points, project_id)
            stage_timings["glb_export"] = round(time.perf_counter() - glb_started_at, 3)
            model_path = glb_model_path
        elif output_format != OutputFormat.OBJ:
            raise ProcessingError(f"Formato de salida no soportado para COLMAP: {output_format.value}.")

        elapsed_seconds = round(time.perf_counter() - started_at, 3)
        metrics = {
            "total_processing_seconds": elapsed_seconds,
            "image_count_processed": len(image_paths),
            "reconstructed_camera_count": registered_image_count,
            "intrinsic_camera_count": camera_count,
            "point_3d_count": len(points),
        }
        artifacts = {
            "model_path": str(model_path),
            "obj_model_path": str(obj_model_path),
            "glb_model_path": str(glb_model_path) if glb_model_path.exists() else None,
            "sparse_txt_dir": str(txt_model_dir),
            "raw_sparse_ply": str(raw_ply_path) if raw_ply_path.exists() else None,
            "logs_dir": str(logs_dir),
        }

        self._notify_progress(
            progress_callback,
            {
                "engine": self.name,
                "current_stage": "export",
                "progress": 0.95,
                "status_message": "Artefactos finales exportados.",
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
            "reconstruction_type": "sparse_photogrammetry",
            "image_count_processed": len(image_paths),
            "registered_image_count": registered_image_count,
            "camera_count": camera_count,
            "point_count": len(points),
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
            "Project %s | COLMAP reconstruction completed in %.3fs. registered_images=%s cameras=%s points=%s model=%s",
            project_id,
            elapsed_seconds,
            registered_image_count,
            camera_count,
            len(points),
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