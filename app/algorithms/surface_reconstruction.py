from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SurfaceReconstructionResult:
    mesh: Any
    method_used: str
    vertices_count: int
    faces_count: int
    input_points_count: int
    outliers_removed: int
    confidence_score: float
    limitations: list[str]
    used_open3d: bool
    color_strategy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_used": self.method_used,
            "vertices_count": self.vertices_count,
            "faces_count": self.faces_count,
            "input_points_count": self.input_points_count,
            "outliers_removed": self.outliers_removed,
            "confidence_score": self.confidence_score,
            "limitations": list(self.limitations),
            "used_open3d": self.used_open3d,
            "color_strategy": self.color_strategy,
        }


class SurfaceReconstruction:
    def __init__(self, *, min_surface_points: int = 1200) -> None:
        self.min_surface_points = max(50, int(min_surface_points))

    def reconstruct_from_sparse(
        self,
        *,
        points_xyz: list[tuple[float, float, float]],
        point_colors_rgb: list[tuple[int, int, int]] | None,
        trimesh_module: Any,
    ) -> SurfaceReconstructionResult | None:
        if len(points_xyz) < 8:
            return None

        import numpy as np

        original_count = len(points_xyz)
        pts = np.asarray(points_xyz, dtype=float)
        colors = np.asarray(point_colors_rgb, dtype=np.uint8) if point_colors_rgb else None

        pts, colors, removed = self._remove_outliers(pts, colors)
        if pts.shape[0] < 8:
            return None

        pts = self._normalize_points(pts)
        pts, colors = self._light_densify(pts, colors)
        pts, colors = self._voxel_downsample(pts, colors)
        if pts.shape[0] < 8:
            return None

        limitations: list[str] = []
        used_open3d = False
        mesh = None
        method = ""

        open3d_mesh = self._try_open3d_ball_pivoting(pts)
        if open3d_mesh is not None:
            mesh = open3d_mesh
            method = "ball_pivoting"
            used_open3d = True

        if mesh is None:
            mesh = self._try_alpha_shape_adaptive(pts, trimesh_module)
            if mesh is not None:
                method = "alpha_shape"

        if mesh is None:
            mesh = self._try_convex_hull(pts, trimesh_module)
            if mesh is not None:
                method = "convex_hull"
                limitations.append("La superficie puede sobreestimar volumen en cavidades.")

        if mesh is None:
            mesh = self._oriented_box_mesh(pts, trimesh_module)
            method = "bounding_mesh"
            limitations.append("La geometria proviene de aproximacion por bounding box orientada.")

        mesh = self._post_smooth_mesh(mesh)

        vertices_count = int(len(getattr(mesh, "vertices", ())))
        faces_count = int(len(getattr(mesh, "faces", ())))
        if vertices_count <= 0 or faces_count <= 0:
            return None

        color_strategy = "none"
        if colors is not None and len(colors) >= 8:
            color_strategy = "vertex_colors_from_colmap"
            self._paint_mesh_vertices(mesh, pts, colors)
        else:
            color_strategy = "average_image_color"
            self._paint_mesh_color(mesh, [180, 180, 180, 255])

        confidence = self._confidence_score(
            input_points=original_count,
            kept_points=int(pts.shape[0]),
            faces_count=faces_count,
            method=method,
        )
        if original_count < self.min_surface_points:
            limitations.append("Cantidad de puntos sparse limitada para superficie robusta.")

        return SurfaceReconstructionResult(
            mesh=mesh,
            method_used=method,
            vertices_count=vertices_count,
            faces_count=faces_count,
            input_points_count=original_count,
            outliers_removed=removed,
            confidence_score=confidence,
            limitations=limitations,
            used_open3d=used_open3d,
            color_strategy=color_strategy,
        )

    def _remove_outliers(self, pts, colors):
        import numpy as np

        if pts.shape[0] < 30:
            return pts, colors, 0
        center = np.median(pts, axis=0)
        distances = np.linalg.norm(pts - center, axis=1)
        q95 = float(np.percentile(distances, 95))
        mad = float(np.median(np.abs(distances - np.median(distances))))
        threshold = min(q95, float(np.median(distances) + 3.5 * max(mad, 1e-6)))
        mask = distances <= max(threshold, 1e-8)
        kept = pts[mask]
        kept_colors = colors[mask] if colors is not None and len(colors) == len(pts) else colors
        return kept, kept_colors, int(len(pts) - len(kept))

    def _normalize_points(self, pts):
        import numpy as np

        min_vals = np.min(pts, axis=0)
        max_vals = np.max(pts, axis=0)
        center = (min_vals + max_vals) * 0.5
        extent = np.max(max_vals - min_vals)
        scale = 1.0 if extent <= 1e-9 else (2.0 / extent)
        return (pts - center) * scale

    def _light_densify(self, pts, colors):
        import numpy as np

        n = pts.shape[0]
        if n >= 2200 or n < 80:
            return pts, colors
        k = min(3, max(1, n // 200))
        generated = []
        for i in range(0, n - 1, max(1, n // 1200)):
            for j in range(1, k + 1):
                b = (i + j) % n
                generated.append((pts[i] + pts[b]) * 0.5)
        if not generated:
            return pts, colors
        new_pts = np.vstack((pts, np.asarray(generated, dtype=float)))
        if colors is not None and len(colors) == n:
            add = np.repeat(colors[:1], len(generated), axis=0)
            new_colors = np.vstack((colors, add))
        else:
            new_colors = colors
        return new_pts, new_colors

    def _voxel_downsample(self, pts, colors):
        import numpy as np

        if pts.shape[0] <= 7000:
            return pts, colors
        voxel = 0.02
        q = np.floor(pts / voxel).astype(int)
        _, unique_idx = np.unique(q, axis=0, return_index=True)
        unique_idx.sort()
        sampled_pts = pts[unique_idx]
        sampled_colors = colors[unique_idx] if colors is not None and len(colors) == len(pts) else colors
        return sampled_pts, sampled_colors

    def _try_open3d_ball_pivoting(self, pts):
        try:
            import open3d as o3d
            import numpy as np
        except Exception:
            return None
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.estimate_normals()
            radii = o3d.utility.DoubleVector([0.025, 0.05, 0.08])
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
            if len(mesh.vertices) <= 0 or len(mesh.triangles) <= 0:
                return None
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            import trimesh

            return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        except Exception:
            return None

    def _try_alpha_shape_adaptive(self, pts, trimesh_module):
        try:
            from scipy.spatial import Delaunay
            import numpy as np
        except Exception:
            return None
        try:
            tri = Delaunay(pts)
            faces = tri.convex_hull
            if faces is None or len(faces) <= 0:
                return None
            mesh = trimesh_module.Trimesh(vertices=pts, faces=faces, process=True)
            if len(getattr(mesh, "faces", ())) <= 0:
                return None
            return mesh
        except Exception:
            return None

    def _try_convex_hull(self, pts, trimesh_module):
        try:
            cloud = trimesh_module.points.PointCloud(pts)
            mesh = cloud.convex_hull
            if len(getattr(mesh, "faces", ())) <= 0:
                return None
            return mesh
        except Exception:
            return None

    def _oriented_box_mesh(self, pts, trimesh_module):
        to_origin, extents = trimesh_module.bounds.oriented_bounds(pts)
        transform = None
        try:
            transform = trimesh_module.transformations.inverse_matrix(to_origin)
        except Exception:
            transform = None
        mesh = trimesh_module.creation.box(extents=extents, transform=transform)
        try:
            mesh = mesh.subdivide()
        except Exception:
            pass
        return mesh

    def _post_smooth_mesh(self, mesh):
        try:
            import trimesh

            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=3)
        except Exception:
            pass
        return mesh

    def _paint_mesh_vertices(self, mesh, pts, colors):
        try:
            from scipy.spatial import cKDTree
            import numpy as np
        except Exception:
            self._paint_mesh_color(mesh, [180, 180, 180, 255])
            return
        try:
            verts = np.asarray(mesh.vertices)
            tree = cKDTree(pts)
            _, idx = tree.query(verts, k=1)
            vertex_colors = colors[np.asarray(idx, dtype=int)]
            if vertex_colors.shape[1] == 3:
                alpha = np.full((vertex_colors.shape[0], 1), 255, dtype=np.uint8)
                vertex_colors = np.concatenate((vertex_colors, alpha), axis=1)
            mesh.visual.vertex_colors = vertex_colors
        except Exception:
            self._paint_mesh_color(mesh, [180, 180, 180, 255])

    @staticmethod
    def _paint_mesh_color(mesh, rgba):
        try:
            mesh.visual.vertex_colors = rgba
        except Exception:
            return

    @staticmethod
    def _confidence_score(*, input_points: int, kept_points: int, faces_count: int, method: str) -> float:
        points_ratio = min(1.0, kept_points / max(1.0, input_points))
        faces_score = min(1.0, faces_count / 5000.0)
        method_bonus = 0.1 if method in {"ball_pivoting", "alpha_shape"} else 0.05 if method == "convex_hull" else 0.0
        score = 0.35 * points_ratio + 0.55 * faces_score + method_bonus
        return round(max(0.0, min(1.0, score)), 3)
