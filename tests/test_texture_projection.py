from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.algorithms.texture_projection import TextureProjection


class _Mesh:
    def __init__(self) -> None:
        self.vertices = [
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ]
        self.faces = [
            (0, 1, 2), (0, 2, 3),
            (4, 5, 6), (4, 6, 7),
            (0, 4, 7), (0, 7, 3),
            (1, 5, 6), (1, 6, 2),
            (3, 2, 6), (3, 6, 7),
            (0, 1, 5), (0, 5, 4),
        ]
        self.visual = type("Visual", (), {})()
        self.visual.vertex_colors = None


class TextureProjectionTests(unittest.TestCase):
    def test_geometric_prior_box_like_uses_multiview_texture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, color in enumerate([(180, 30, 30), (30, 180, 30), (30, 30, 180)], start=1):
                Image.new("RGB", (320, 240), color).save(root / f"img_{i}.jpg")
            mesh = _Mesh()
            projection = TextureProjection()
            result = projection.apply(
                mesh=mesh,
                point_colors_rgb=None,
                image_dir=root,
                detected_shape_prior="box_like",
                output_dir=None,
            )
            payload = result.to_dict()
            self.assertEqual(payload["texture_source"], "best_image_projection")
            self.assertEqual(payload["texture_method"], "box_like_multiview_face_colors")
            self.assertGreaterEqual(payload["textured_faces_count"], 6)

    def test_texture_fallback_uses_average_color_when_no_features(self) -> None:
        mesh = _Mesh()
        projection = TextureProjection()
        result = projection.apply(
            mesh=mesh,
            point_colors_rgb=None,
            image_dir=None,
            detected_shape_prior="irregular",
            output_dir=None,
        )
        payload = result.to_dict()
        self.assertEqual(payload["texture_source"], "average_image_color")
        self.assertTrue(payload["fallback_texture_used"])


if __name__ == "__main__":
    unittest.main()
