from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models.schemas import OutputFormat

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PresentationPostprocessDecision:
    apply: bool
    profile: dict[str, Any]
    profile_path: Path | None


@dataclass(frozen=True)
class PresentationPostprocessResult:
    applied: bool
    model_path: Path
    details: dict[str, Any]


class PresentationPostprocessService:
    PROFILE_FILENAME = "presentation_profile.json"
    SUPPORTED_MODE = "sparse_convex_hull_cleanup"
    ORIENTED_BOX_MODE = "sparse_oriented_box_cleanup"
    DEFAULT_COLOR_RGBA = (198, 203, 210, 255)

    def should_apply(self, project_dir: Path, project_id: str) -> PresentationPostprocessDecision:
        profile_path = project_dir / self.PROFILE_FILENAME
        if not profile_path.exists():
            return PresentationPostprocessDecision(
                apply=False,
                profile={},
                profile_path=None,
            )

        try:
            profile_payload = json.loads(profile_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            logger.warning(
                "Ignoring invalid presentation profile. project_id=%s path=%s error=%s",
                project_id,
                profile_path,
                exc,
            )
            return PresentationPostprocessDecision(
                apply=False,
                profile={},
                profile_path=profile_path,
            )

        profile = profile_payload if isinstance(profile_payload, dict) else {}
        if not bool(profile.get("enabled", False)):
            return PresentationPostprocessDecision(
                apply=False,
                profile=profile,
                profile_path=profile_path,
            )

        expected_project_id = str(profile.get("project_id") or project_id).strip()
        if expected_project_id and expected_project_id != project_id:
            return PresentationPostprocessDecision(
                apply=False,
                profile=profile,
                profile_path=profile_path,
            )

        mode = str(profile.get("mode") or self.SUPPORTED_MODE).strip().lower()
        supported_modes = {self.SUPPORTED_MODE, self.ORIENTED_BOX_MODE}
        if mode not in supported_modes:
            logger.warning(
                "Ignoring presentation profile with unsupported mode. project_id=%s mode=%s supported_modes=%s",
                project_id,
                mode,
                sorted(supported_modes),
            )
            return PresentationPostprocessDecision(
                apply=False,
                profile=profile,
                profile_path=profile_path,
            )

        return PresentationPostprocessDecision(
            apply=True,
            profile=profile,
            profile_path=profile_path,
        )

    def apply(
        self,
        *,
        project_id: str,
        output_dir: Path,
        output_format: OutputFormat,
        current_model_path: Path,
        profile: dict[str, Any],
        profile_path: Path | None,
    ) -> PresentationPostprocessResult:
        trimesh_module = importlib.import_module("trimesh")
        numpy_module = importlib.import_module("numpy")
        points_path = output_dir / "workspace" / "sparse" / "0" / "points3D.ply"

        if not points_path.exists():
            raise RuntimeError(
                "No se encontro la nube sparse 'points3D.ply' requerida para aplicar el perfil de presentacion."
            )

        cloud = trimesh_module.load(str(points_path), file_type="ply")
        points = numpy_module.asarray(getattr(cloud, "vertices", ()), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 8:
            raise RuntimeError(
                "La nube sparse no contiene suficientes vertices 3D para construir una malla presentable."
            )

        filtered_points = self._filter_outliers(points, numpy_module)
        requested_mode = str(profile.get("mode") or self.SUPPORTED_MODE).strip().lower()
        if requested_mode == self.ORIENTED_BOX_MODE:
            mesh = self._build_oriented_box(trimesh_module, numpy_module, filtered_points, profile)
            method_used = self.ORIENTED_BOX_MODE
        else:
            base_mesh = trimesh_module.Trimesh(vertices=filtered_points, process=True)
            mesh = base_mesh.convex_hull
            method_used = self.SUPPORTED_MODE
            if len(getattr(mesh, "vertices", ())) < 4 or len(getattr(mesh, "faces", ())) < 4:
                mesh = self._build_oriented_box(trimesh_module, numpy_module, filtered_points, profile)
                method_used = self.ORIENTED_BOX_MODE

        mesh = self._apply_bevel_if_requested(mesh, trimesh_module, profile)
        rgba = self._resolve_rgba_color(profile)
        self._apply_vertex_shading(mesh, numpy_module, rgba, profile)
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()

        glb_model_path = output_dir / f"{project_id}_model.glb"
        obj_model_path = output_dir / f"{project_id}_model.obj"
        self._write_export(glb_model_path, mesh.export(file_type="glb"))
        self._write_export(obj_model_path, mesh.export(file_type="obj"))

        final_model_path = glb_model_path if output_format == OutputFormat.GLB else obj_model_path
        details = {
            "applied": True,
            "mode": str(profile.get("mode") or self.SUPPORTED_MODE),
            "method": method_used,
            "profile_path": str(profile_path) if profile_path is not None else None,
            "source_sparse_points_path": str(points_path),
            "source_sparse_points": int(points.shape[0]),
            "source_filtered_points": int(filtered_points.shape[0]),
            "result_vertex_count": int(len(mesh.vertices)),
            "result_face_count": int(len(mesh.faces)),
            "output_model_path": str(final_model_path),
            "replaced_models": [str(glb_model_path), str(obj_model_path)],
            "color_rgba": list(rgba),
            "notes": (
                "Malla de presentacion reconstruida desde la nube sparse para evitar artefactos espurios "
                "del fallback delaunay."
            ),
        }
        return PresentationPostprocessResult(
            applied=True,
            model_path=final_model_path,
            details=details,
        )

    @staticmethod
    def _write_export(path: Path, payload: object) -> None:
        if isinstance(payload, bytes):
            path.write_bytes(payload)
            return
        path.write_text(str(payload), encoding="utf-8")

    @classmethod
    def _filter_outliers(cls, points: Any, numpy_module: Any) -> Any:
        if int(points.shape[0]) < 20:
            return points

        center = numpy_module.median(points, axis=0)
        radial = numpy_module.linalg.norm(points - center, axis=1)
        q1 = float(numpy_module.percentile(radial, 25))
        q3 = float(numpy_module.percentile(radial, 75))
        iqr = max(q3 - q1, 1e-9)
        upper_limit = q3 + (1.5 * iqr)
        keep = radial <= upper_limit
        filtered = points[keep]
        if int(filtered.shape[0]) < 8:
            filtered = points

        median = numpy_module.median(filtered, axis=0)
        mad = numpy_module.median(numpy_module.abs(filtered - median), axis=0) + 1e-9
        robust_score = numpy_module.abs((filtered - median) / (1.4826 * mad))
        keep_axis = numpy_module.all(robust_score <= 3.5, axis=1)
        filtered_axis = filtered[keep_axis]
        if int(filtered_axis.shape[0]) < 8:
            return filtered
        return filtered_axis

    @classmethod
    def _trim_points_for_bounds(cls, points: Any, numpy_module: Any, profile: dict[str, Any]) -> Any:
        if int(points.shape[0]) < 50:
            return points

        try:
            trim_quantile = float(profile.get("bounds_trim_quantile", 0.04))
        except (TypeError, ValueError):
            trim_quantile = 0.04
        trim_quantile = max(0.0, min(0.2, trim_quantile))
        if trim_quantile <= 0.0:
            return points

        lower = numpy_module.quantile(points, trim_quantile, axis=0)
        upper = numpy_module.quantile(points, 1.0 - trim_quantile, axis=0)
        keep = numpy_module.all((points >= lower) & (points <= upper), axis=1)
        trimmed = points[keep]
        if int(trimmed.shape[0]) < 8:
            return points
        return trimmed

    @classmethod
    def _build_oriented_box(
        cls,
        trimesh_module: Any,
        numpy_module: Any,
        points: Any,
        profile: dict[str, Any],
    ) -> Any:
        bounds_points = cls._trim_points_for_bounds(points, numpy_module, profile)
        obb_transform, extents = trimesh_module.bounds.oriented_bounds(bounds_points)
        try:
            min_extent = float(profile.get("min_extent", 0.02))
        except (TypeError, ValueError):
            min_extent = 0.02
        min_extent = max(0.001, min_extent)
        safe_extents = numpy_module.maximum(extents, min_extent)

        try:
            min_ratio = float(profile.get("min_extent_ratio_of_median", 0.45))
        except (TypeError, ValueError):
            min_ratio = 0.45
        min_ratio = max(0.0, min(1.0, min_ratio))
        median_extent = float(numpy_module.median(safe_extents))
        safe_extents = numpy_module.maximum(safe_extents, max(min_extent, median_extent * min_ratio))

        try:
            max_aspect = float(profile.get("max_aspect_ratio", 2.2))
        except (TypeError, ValueError):
            max_aspect = 2.2
        max_aspect = max(1.0, max_aspect)
        min_current = float(max(float(numpy_module.min(safe_extents)), 1e-6))
        max_allowed = min_current * max_aspect
        safe_extents = numpy_module.minimum(safe_extents, max_allowed)

        return trimesh_module.creation.box(
            extents=safe_extents,
            transform=numpy_module.linalg.inv(obb_transform),
        )

    @staticmethod
    def _apply_bevel_if_requested(mesh: Any, trimesh_module: Any, profile: dict[str, Any]) -> Any:
        try:
            subdivide_iterations = int(profile.get("bevel_subdivide_iterations", 1))
        except (TypeError, ValueError):
            subdivide_iterations = 1
        subdivide_iterations = max(0, min(2, subdivide_iterations))
        if subdivide_iterations <= 0:
            return mesh

        polished = mesh.copy()
        for _ in range(subdivide_iterations):
            polished = polished.subdivide()

        try:
            smooth_iterations = int(profile.get("bevel_smooth_iterations", 6))
        except (TypeError, ValueError):
            smooth_iterations = 6
        smooth_iterations = max(0, min(20, smooth_iterations))
        if smooth_iterations > 0:
            try:
                smooth_lambda = float(profile.get("bevel_smooth_lambda", 0.32))
            except (TypeError, ValueError):
                smooth_lambda = 0.32
            smooth_lambda = max(0.0, min(1.0, smooth_lambda))
            try:
                trimesh_module.smoothing.filter_laplacian(
                    polished,
                    lamb=smooth_lambda,
                    iterations=smooth_iterations,
                    implicit_time_integration=False,
                )
            except Exception:
                # Si no hay soporte de smoothing, mantenemos el mesh subdividido.
                pass

        polished.remove_unreferenced_vertices()
        polished.fix_normals()
        return polished

    @staticmethod
    def _apply_vertex_shading(mesh: Any, numpy_module: Any, rgba: tuple[int, int, int, int], profile: dict[str, Any]) -> None:
        vertex_count = len(getattr(mesh, "vertices", ()))
        if vertex_count <= 0:
            return

        base = numpy_module.array(rgba, dtype=float)
        rgb = base[:3]
        alpha = int(base[3])

        try:
            shade_strength = float(profile.get("shade_strength", 0.20))
        except (TypeError, ValueError):
            shade_strength = 0.20
        shade_strength = max(0.0, min(0.5, shade_strength))

        normals = numpy_module.asarray(getattr(mesh, "vertex_normals", ()), dtype=float)
        if normals.ndim != 2 or normals.shape[0] != vertex_count or normals.shape[1] != 3:
            colors = numpy_module.tile(numpy_module.array(rgba, dtype=numpy_module.uint8), (vertex_count, 1))
            mesh.visual.vertex_colors = colors
            return

        light_dir = numpy_module.array([0.25, 0.45, 0.86], dtype=float)
        light_norm = float(numpy_module.linalg.norm(light_dir))
        if light_norm > 0:
            light_dir = light_dir / light_norm

        dots = normals @ light_dir
        # 0.82..1.18 aprox: contraste sutil para que se lea volumen en visores simples.
        factors = 1.0 + (shade_strength * dots)
        factors = numpy_module.clip(factors, 0.72, 1.25)
        shaded_rgb = numpy_module.clip(rgb[None, :] * factors[:, None], 0, 255).astype(numpy_module.uint8)
        alpha_col = numpy_module.full((vertex_count, 1), alpha, dtype=numpy_module.uint8)
        mesh.visual.vertex_colors = numpy_module.concatenate([shaded_rgb, alpha_col], axis=1)

    @classmethod
    def _resolve_rgba_color(cls, profile: dict[str, Any]) -> tuple[int, int, int, int]:
        raw = profile.get("color_rgba")
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            return cls.DEFAULT_COLOR_RGBA
        values: list[int] = []
        for item in raw:
            try:
                values.append(max(0, min(255, int(item))))
            except (TypeError, ValueError):
                return cls.DEFAULT_COLOR_RGBA
        return tuple(values)  # type: ignore[return-value]
