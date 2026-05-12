from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


@dataclass(frozen=True)
class TextureProjectionResult:
    texture_source: str
    texture_method: str
    selected_images: list[str]
    texture_confidence: float
    textured_faces_count: int
    untextured_faces_count: int
    texture_limitations: list[str]
    fallback_texture_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "texture_source": self.texture_source,
            "texture_method": self.texture_method,
            "selected_images": list(self.selected_images),
            "texture_confidence": self.texture_confidence,
            "textured_faces_count": self.textured_faces_count,
            "untextured_faces_count": self.untextured_faces_count,
            "texture_limitations": list(self.texture_limitations),
            "fallback_texture_used": self.fallback_texture_used,
        }


class TextureProjection:
    def apply(
        self,
        *,
        mesh: Any,
        point_colors_rgb: list[tuple[int, int, int]] | None,
        image_dir: Path | None,
        detected_shape_prior: str | None,
        output_dir: Path | None = None,
    ) -> TextureProjectionResult:
        candidate_dir = self._pick_candidate_image_dir(image_dir=image_dir, output_dir=output_dir)
        ranked_images = self._rank_images(candidate_dir, output_dir=output_dir)

        if detected_shape_prior == "box_like" and ranked_images:
            try:
                textured_faces = self._paint_box_like_faces(mesh, ranked_images)
                face_total = self._safe_face_count(mesh)
                return TextureProjectionResult(
                    texture_source="best_image_projection",
                    texture_method="box_like_multiview_face_colors",
                    selected_images=[str(p) for p in ranked_images[:6]],
                    texture_confidence=0.66,
                    textured_faces_count=textured_faces,
                    untextured_faces_count=max(0, face_total - textured_faces),
                    texture_limitations=["Proyeccion por color de cara; no UV fotometrico completo."],
                    fallback_texture_used=False,
                )
            except Exception:
                pass

        if point_colors_rgb:
            try:
                self._paint_vertex_from_point_colors(mesh, point_colors_rgb)
                face_total = self._safe_face_count(mesh)
                return TextureProjectionResult(
                    texture_source="vertex_colors_from_colmap",
                    texture_method="nearest_sparse_color_transfer",
                    selected_images=[str(p) for p in ranked_images[:3]],
                    texture_confidence=0.58,
                    textured_faces_count=face_total,
                    untextured_faces_count=0,
                    texture_limitations=["Color por vertice aproximado por vecindad, no UV fotometrico real."],
                    fallback_texture_used=False,
                )
            except Exception:
                pass

        avg_rgb = self._compute_average_image_color(candidate_dir)
        face_total = self._safe_face_count(mesh)
        if avg_rgb is not None:
            self._paint_solid_color(mesh, (*avg_rgb, 255))
            return TextureProjectionResult(
                texture_source="average_image_color",
                texture_method="global_average_color",
                selected_images=[str(p) for p in ranked_images[:2]],
                texture_confidence=0.3,
                textured_faces_count=0,
                untextured_faces_count=face_total,
                texture_limitations=["Se uso color promedio por falta de textura multi-vista confiable."],
                fallback_texture_used=True,
            )

        self._paint_solid_color(mesh, (180, 180, 180, 255))
        return TextureProjectionResult(
            texture_source="average_image_color",
            texture_method="neutral_fallback_color",
            selected_images=[],
            texture_confidence=0.15,
            textured_faces_count=0,
            untextured_faces_count=face_total,
            texture_limitations=["No fue posible estimar color de imagen; se uso color neutro."],
            fallback_texture_used=True,
        )

    def _paint_box_like_faces(self, mesh: Any, ranked_images: list[Path]) -> int:
        import numpy as np

        faces = np.asarray(getattr(mesh, "faces", []), dtype=int)
        verts = np.asarray(getattr(mesh, "vertices", []), dtype=float)
        if faces.size == 0 or verts.size == 0:
            raise ValueError("mesh without faces")

        fn = self._face_normals(verts, faces)
        groups = {
            "front": np.array([0.0, 0.0, 1.0]),
            "back": np.array([0.0, 0.0, -1.0]),
            "left": np.array([-1.0, 0.0, 0.0]),
            "right": np.array([1.0, 0.0, 0.0]),
            "top": np.array([0.0, 1.0, 0.0]),
            "bottom": np.array([0.0, -1.0, 0.0]),
        }
        face_groups: dict[str, list[int]] = {k: [] for k in groups}
        for i, n in enumerate(fn):
            best_key = max(groups, key=lambda k: float(np.dot(n, groups[k])))
            face_groups[best_key].append(i)

        palette = [self._dominant_color(path) for path in ranked_images[:6]]
        while len(palette) < 6:
            palette.append((180, 180, 180))
        color_by_group = {
            "front": palette[0],
            "back": palette[1],
            "left": palette[2],
            "right": palette[3],
            "top": palette[4],
            "bottom": palette[5],
        }

        vcolors = np.full((len(verts), 4), [180, 180, 180, 255], dtype=np.uint8)
        touched_faces = 0
        for key, idxs in face_groups.items():
            if not idxs:
                continue
            touched_faces += len(idxs)
            color = np.asarray([*color_by_group[key], 255], dtype=np.uint8)
            face_vertices = faces[np.asarray(idxs, dtype=int)].reshape(-1)
            vcolors[face_vertices] = color
        mesh.visual.vertex_colors = vcolors
        return touched_faces

    @staticmethod
    def _safe_face_count(mesh: Any) -> int:
        faces = getattr(mesh, "faces", None)
        if faces is None:
            return 0
        try:
            return int(len(faces))
        except Exception:
            return 0

    @staticmethod
    def _face_normals(verts, faces):
        import numpy as np

        tri = verts[faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        return n / norms

    @staticmethod
    def _paint_vertex_from_point_colors(mesh: Any, point_colors_rgb: list[tuple[int, int, int]]) -> None:
        import numpy as np

        colors = np.asarray(point_colors_rgb, dtype=np.uint8)
        if colors.size == 0:
            raise ValueError("No colors")
        verts = np.asarray(mesh.vertices)
        if len(verts) <= 0:
            raise ValueError("No vertices")
        idx = np.linspace(0, len(colors) - 1, len(verts)).astype(int)
        picked = colors[idx]
        alpha = np.full((picked.shape[0], 1), 255, dtype=np.uint8)
        mesh.visual.vertex_colors = np.concatenate((picked, alpha), axis=1)

    @staticmethod
    def _paint_solid_color(mesh: Any, rgba: tuple[int, int, int, int]) -> None:
        mesh.visual.vertex_colors = rgba

    def _rank_images(self, image_dir: Path | None, *, output_dir: Path | None) -> list[Path]:
        if image_dir is None or not image_dir.exists() or not image_dir.is_dir():
            return []
        seg_ratios = self._segmentation_foreground_ratios(output_dir)
        scores: list[tuple[float, Path]] = []
        for path in sorted(image_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
                continue
            try:
                with Image.open(path) as img:
                    rgb = img.convert("RGB")
                    gray = img.convert("L")
                    sharp = self._laplacian_variance(gray)
                    feat = self._feature_count(gray)
                    fg = seg_ratios.get(path.name, 0.25)
                    score = sharp * 0.45 + feat * 0.35 + fg * 0.2
                    scores.append((score, path))
            except Exception:
                continue
        scores.sort(key=lambda t: t[0], reverse=True)
        return [p for _, p in scores]

    @staticmethod
    def _laplacian_variance(gray: Image.Image) -> float:
        try:
            import cv2
            import numpy as np

            arr = np.asarray(gray)
            var = float(cv2.Laplacian(arr, cv2.CV_64F).var())
            return min(1.0, var / 500.0)
        except Exception:
            stat = ImageStat.Stat(gray)
            return min(1.0, float((stat.stddev[0] if stat.stddev else 0.0) / 64.0))

    @staticmethod
    def _feature_count(gray: Image.Image) -> float:
        try:
            import cv2
            import numpy as np

            arr = np.asarray(gray)
            orb = cv2.ORB_create(nfeatures=300)
            kp = orb.detect(arr, None)
            return min(1.0, float(len(kp or [])) / 220.0)
        except Exception:
            return 0.2

    def _segmentation_foreground_ratios(self, output_dir: Path | None) -> dict[str, float]:
        if output_dir is None:
            return {}
        report = output_dir / "segmentation" / "segmentation_report.json"
        if not report.exists():
            return {}
        try:
            import json

            payload = json.loads(report.read_text(encoding="utf-8"))
            images = payload.get("images") if isinstance(payload, dict) else None
            if not isinstance(images, list):
                return {}
            out: dict[str, float] = {}
            for item in images:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("filename") or "").strip()
                if not name:
                    continue
                try:
                    out[name] = float(item.get("foreground_ratio") or 0.0)
                except (TypeError, ValueError):
                    pass
            return out
        except Exception:
            return {}

    @staticmethod
    def _pick_candidate_image_dir(*, image_dir: Path | None, output_dir: Path | None) -> Path | None:
        candidates: list[Path] = []
        if output_dir is not None:
            candidates.extend(
                [
                    output_dir / "segmentation" / "segmented_images",
                    output_dir / "preprocessed_images",
                    output_dir / "validation" / "selected_images",
                    output_dir / "validation" / "accepted_images",
                ]
            )
        if image_dir is not None:
            candidates.append(image_dir)
        for path in candidates:
            if path.exists() and path.is_dir() and any(p.is_file() for p in path.iterdir()):
                return path
        return image_dir

    @staticmethod
    def _dominant_color(path: Path) -> tuple[int, int, int]:
        with Image.open(path) as img:
            small = img.convert("RGB").resize((80, 80))
            mean = ImageStat.Stat(small).mean
            return int(round(mean[0])), int(round(mean[1])), int(round(mean[2]))

    @staticmethod
    def _compute_average_image_color(image_dir: Path | None) -> tuple[int, int, int] | None:
        if image_dir is None or not image_dir.exists() or not image_dir.is_dir():
            return None
        channels = [[], [], []]
        for path in sorted(image_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
                continue
            try:
                with Image.open(path) as img:
                    mean = ImageStat.Stat(img.convert("RGB")).mean
                    for i in range(3):
                        channels[i].append(float(mean[i]))
            except Exception:
                continue
        if not channels[0]:
            return None
        return tuple(int(round(sum(ch) / len(ch))) for ch in channels)
