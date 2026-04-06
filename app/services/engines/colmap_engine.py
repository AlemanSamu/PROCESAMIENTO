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
from app.services.engines.base_engine import ReconstructionEngine, ReconstructionResult

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

        output_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir = output_dir / "workspace"
        sparse_dir = workspace_dir / "sparse"
        dense_dir = workspace_dir / "dense"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        dense_dir.mkdir(parents=True, exist_ok=True)

        database_path = workspace_dir / "database.db"
        txt_model_dir = output_dir / "colmap_sparse_txt"
        txt_model_dir.mkdir(parents=True, exist_ok=True)
        raw_ply_path = output_dir / f"{project_id}_sparse.ply"

        command_traces: list[ColmapCommandTrace] = []
        warnings: list[str] = []

        command_traces.append(
            self._run_command(
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
                    "--SiftExtraction.use_gpu",
                    "1" if self.use_gpu else "0",
                ],
            )
        )
        command_traces.append(
            self._run_command(
                "feature_matching",
                [
                    resolved_binary,
                    "exhaustive_matcher",
                    "--database_path",
                    str(database_path),
                    "--SiftMatching.use_gpu",
                    "1" if self.use_gpu else "0",
                ],
            )
        )
        command_traces.append(
            self._run_command(
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
            )
        )

        sparse_model_dir = self._select_sparse_model_dir(sparse_dir)
        command_traces.append(
            self._run_command(
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
            )
        )

        try:
            command_traces.append(
                self._run_command(
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
                )
            )
        except ProcessingError as exc:
            warning = f"No se pudo exportar el artefacto PLY de COLMAP: {exc}"
            warnings.append(warning)
            logger.warning(warning)

        points = self._load_sparse_points(txt_model_dir / "points3D.txt")
        if not points:
            raise ProcessingError("COLMAP genero un modelo sparse, pero no produjo puntos 3D utilizables.")

        registered_image_count = self._load_registered_image_count(txt_model_dir / "images.txt")
        if output_format == OutputFormat.OBJ:
            model_path = output_dir / f"{project_id}_model.obj"
            self._write_obj_point_cloud(model_path, points, project_id)
        elif output_format == OutputFormat.GLB:
            model_path = output_dir / f"{project_id}_model.glb"
            self._write_glb_point_cloud(model_path, points, project_id)
        else:
            raise ProcessingError(f"Formato de salida no soportado para COLMAP: {output_format.value}.")

        elapsed_seconds = round(time.perf_counter() - started_at, 3)
        metadata_path = output_dir / f"{project_id}_colmap_metadata.json"
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
            "point_count": len(points),
            "workspace": {
                "root": str(workspace_dir),
                "database_path": str(database_path),
                "sparse_dir": str(sparse_dir),
                "dense_dir": str(dense_dir),
                "selected_sparse_model_dir": str(sparse_model_dir),
            },
            "artifacts": {
                "model_path": str(model_path),
                "sparse_txt_dir": str(txt_model_dir),
                "raw_sparse_ply": str(raw_ply_path) if raw_ply_path.exists() else None,
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

    @staticmethod
    def _collect_images(images_dir: Path) -> list[Path]:
        if not images_dir.exists():
            raise ProcessingError(f"La carpeta de imagenes no existe: {images_dir}")
        image_paths = sorted(path for path in images_dir.iterdir() if path.is_file())
        if not image_paths:
            raise ProcessingError("No hay imagenes disponibles para reconstruir.")
        return image_paths

    def _run_command(self, name: str, command: list[str]) -> ColmapCommandTrace:
        logger.info("Running COLMAP step '%s': %s", name, subprocess.list2cmdline(command))
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
            logger.exception("COLMAP step '%s' timed out after %s seconds", name, self.timeout_seconds, exc_info=exc)
            raise ProcessingError(
                f"COLMAP agoto el tiempo de espera en el paso '{name}' despues de {self.timeout_seconds} segundos."
            ) from exc

        duration_seconds = round(time.perf_counter() - started_at, 3)
        stdout_tail = self._tail(completed.stdout)
        stderr_tail = self._tail(completed.stderr)
        trace = ColmapCommandTrace(
            name=name,
            command=subprocess.list2cmdline(command),
            duration_seconds=duration_seconds,
            return_code=completed.returncode,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )

        if completed.returncode != 0:
            logger.error(
                "COLMAP step '%s' failed with code %s. stdout=%s stderr=%s",
                name,
                completed.returncode,
                stdout_tail,
                stderr_tail,
            )
            detail = stderr_tail or stdout_tail or "Sin detalle adicional."
            raise ProcessingError(
                f"COLMAP fallo en el paso '{name}' con codigo {completed.returncode}. Detalle: {detail}"
            )

        logger.info("COLMAP step '%s' completed in %.3fs", name, duration_seconds)
        return trace

    @staticmethod
    def _tail(text: str | None) -> str:
        if not text:
            return ""
        normalized = text.strip()
        if len(normalized) <= ColmapReconstructionEngine._MAX_LOG_TAIL:
            return normalized
        return normalized[-ColmapReconstructionEngine._MAX_LOG_TAIL :]

    def _select_sparse_model_dir(self, sparse_dir: Path) -> Path:
        candidates = [path for path in sparse_dir.iterdir() if path.is_dir()]
        valid_models = [path for path in candidates if self._has_sparse_model_files(path)]
        if not valid_models:
            raise ProcessingError(
                "COLMAP no genero un modelo sparse valido en el mapper. Revisa overlap entre imagenes y calibracion."
            )
        return max(valid_models, key=self._sparse_model_score)

    @staticmethod
    def _has_sparse_model_files(model_dir: Path) -> bool:
        required_prefixes = ("cameras", "images", "points3D")
        for prefix in required_prefixes:
            if not any((model_dir / f"{prefix}{suffix}").exists() for suffix in (".bin", ".txt")):
                return False
        return True

    @staticmethod
    def _sparse_model_score(model_dir: Path) -> int:
        points_bin = model_dir / "points3D.bin"
        points_txt = model_dir / "points3D.txt"
        if points_bin.exists():
            return points_bin.stat().st_size
        if points_txt.exists():
            return points_txt.stat().st_size
        return 0

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