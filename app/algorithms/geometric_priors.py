from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeometricPriorResult:
    detected_shape_prior: str
    prior_confidence: float
    prior_used: bool
    prior_limitations: list[str]
    mesh: Any | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_shape_prior": self.detected_shape_prior,
            "prior_confidence": self.prior_confidence,
            "prior_used": self.prior_used,
            "prior_limitations": list(self.prior_limitations),
        }


class GeometricPriorDetector:
    def detect_and_build(
        self,
        *,
        points_xyz: list[tuple[float, float, float]],
        trimesh_module: Any,
    ) -> GeometricPriorResult:
        if len(points_xyz) < 8:
            return GeometricPriorResult(
                detected_shape_prior="irregular",
                prior_confidence=0.0,
                prior_used=False,
                prior_limitations=["Pocos puntos para inferir un prior geometrico robusto."],
                mesh=None,
            )

        import numpy as np

        pts = np.asarray(points_xyz, dtype=float)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        extents = np.maximum(maxs - mins, 1e-9)
        ex, ey, ez = float(extents[0]), float(extents[1]), float(extents[2])
        ratio_xy = max(ex, ey) / max(min(ex, ey), 1e-9)
        ratio_xz = max(ex, ez) / max(min(ex, ez), 1e-9)
        ratio_yz = max(ey, ez) / max(min(ey, ez), 1e-9)

        center = (mins + maxs) * 0.5
        radial = np.linalg.norm(pts[:, :2] - center[:2], axis=1)
        radial_std = float(np.std(radial) / max(float(np.mean(radial)), 1e-6))
        dispersion = float(np.std(pts, axis=0).mean())

        limitations: list[str] = []

        if min(ex, ey, ez) < 0.12 * max(ex, ey, ez):
            shape = "flat_object"
            confidence = max(0.45, min(0.92, 1.0 - min(ex, ey, ez) / max(ex, ey, ez)))
            mesh = self._build_flat_extrusion(extents=(ex, ey, ez), trimesh_module=trimesh_module)
            limitations.append("Extrusion delgada aproximada; no conserva detalle fino.")
        elif radial_std < 0.42 and ratio_xy < 1.6 and ratio_xz > 1.2 and ratio_yz > 1.2:
            shape = "cylinder_like"
            confidence = max(0.4, min(0.9, 0.85 - radial_std * 0.4))
            mesh = self._build_cylinder(extents=(ex, ey, ez), trimesh_module=trimesh_module)
            limitations.append("Cilindro aproximado inferido desde dispersion sparse.")
        elif ratio_xy < 2.4 and ratio_xz < 2.4 and ratio_yz < 2.4:
            shape = "box_like"
            confidence = max(0.4, min(0.88, 0.82 - dispersion * 0.1))
            mesh = self._build_box(extents=(ex, ey, ez), trimesh_module=trimesh_module)
            limitations.append("Prisma aproximado; bordes y cavidades no garantizados.")
        else:
            shape = "irregular"
            confidence = 0.35
            mesh = None
            limitations.append("No hay evidencia suficiente para un prior canonico confiable.")

        return GeometricPriorResult(
            detected_shape_prior=shape,
            prior_confidence=round(float(confidence), 3),
            prior_used=mesh is not None and shape != "irregular",
            prior_limitations=limitations,
            mesh=mesh,
        )

    @staticmethod
    def _build_box(*, extents: tuple[float, float, float], trimesh_module: Any) -> Any:
        mesh = trimesh_module.creation.box(extents=extents)
        try:
            mesh = mesh.subdivide()
        except Exception:
            pass
        return mesh

    @staticmethod
    def _build_cylinder(*, extents: tuple[float, float, float], trimesh_module: Any) -> Any:
        ex, ey, ez = extents
        radius = max(1e-3, min(ex, ey) * 0.5)
        height = max(1e-3, ez)
        return trimesh_module.creation.cylinder(radius=radius, height=height, sections=48)

    @staticmethod
    def _build_flat_extrusion(*, extents: tuple[float, float, float], trimesh_module: Any) -> Any:
        ex, ey, ez = extents
        th = max(1e-3, min(ex, ey, ez))
        if ez <= min(ex, ey):
            return trimesh_module.creation.box(extents=(ex, ey, th))
        if ex <= min(ey, ez):
            return trimesh_module.creation.box(extents=(th, ey, ez))
        return trimesh_module.creation.box(extents=(ex, th, ez))
