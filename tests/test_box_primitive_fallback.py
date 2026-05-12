from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.algorithms.box_primitive_fallback import (
    BoxPrimitiveFallback,
    BoxPrimitiveFallbackSettings,
)
from app.models.schemas import OutputFormat


class BoxPrimitiveFallbackTests(unittest.TestCase):
    @staticmethod
    def _write_box_like_image(path: Path, offset: int) -> None:
        canvas = Image.new("RGB", (800, 600), (230, 230, 230))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            (190 + offset, 120, 610 + offset, 500),
            outline=(40, 40, 40),
            fill=(170, 170, 170),
            width=6,
        )
        draw.line((230 + offset, 170, 560 + offset, 170), fill=(80, 80, 80), width=4)
        draw.line((230 + offset, 210, 560 + offset, 210), fill=(80, 80, 80), width=4)
        canvas.save(path)

    def test_builds_obj_box_fallback_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected_images_dir = root / "selected_images"
            selected_images_dir.mkdir(parents=True, exist_ok=True)
            for index in range(4):
                self._write_box_like_image(selected_images_dir / f"sample_{index + 1}.jpg", offset=index * 8)

            output_dir = root / "output"
            fallback = BoxPrimitiveFallback(
                settings=BoxPrimitiveFallbackSettings(
                    enabled=True,
                    min_selected_images=3,
                )
            )
            result = fallback.build_from_images(
                project_id="demo",
                selected_images_dir=selected_images_dir,
                output_dir=output_dir,
                output_format=OutputFormat.OBJ,
                source_reason="unit_test_failure",
            )

            self.assertTrue(result.model_path.exists())
            self.assertEqual(result.model_path.suffix.lower(), ".obj")
            self.assertTrue(result.report_path.exists())
            self.assertEqual(result.metadata["method_used"], "primitive_box")
            self.assertEqual(result.metadata["reconstruction_type"], "approximate_box_primitive_fallback")
            self.assertEqual(result.metadata["metrics"]["mesh_face_count"], 12)
            self.assertEqual(result.metadata["current_stage"], "completed_with_fallback")

            payload = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["fallback_name"], "primitive_box_fallback")
            self.assertEqual(payload["estimated_box_dimensions"]["height"], 1.0)
            self.assertEqual(payload["selected_images_count"], 4)

    def test_uses_sparse_reference_when_points3d_txt_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected_images_dir = root / "selected_images"
            selected_images_dir.mkdir(parents=True, exist_ok=True)
            for index in range(4):
                self._write_box_like_image(selected_images_dir / f"sample_{index + 1}.jpg", offset=index * 6)

            output_dir = root / "output"
            sparse_txt_dir = output_dir / "colmap_sparse_txt"
            sparse_txt_dir.mkdir(parents=True, exist_ok=True)
            (sparse_txt_dir / "points3D.txt").write_text(
                "# points\n"
                "1 0.0 0.0 0.0 255 0 0 0.1\n"
                "2 1.2 0.0 0.1 255 0 0 0.1\n"
                "3 1.1 0.9 0.2 255 0 0 0.1\n"
                "4 0.0 1.0 0.3 255 0 0 0.1\n"
                "5 0.1 0.1 1.0 255 0 0 0.1\n"
                "6 1.1 0.2 1.1 255 0 0 0.1\n"
                "7 1.0 0.9 1.0 255 0 0 0.1\n"
                "8 0.2 0.8 1.1 255 0 0 0.1\n"
                "9 0.6 0.4 0.7 255 0 0 0.1\n"
                "10 0.5 0.6 0.8 255 0 0 0.1\n"
                "11 0.4 0.5 0.9 255 0 0 0.1\n"
                "12 0.7 0.3 0.6 255 0 0 0.1\n"
                "13 0.3 0.4 0.5 255 0 0 0.1\n"
                "14 0.8 0.7 0.4 255 0 0 0.1\n"
                "15 0.9 0.2 0.3 255 0 0 0.1\n"
                "16 0.2 0.9 0.2 255 0 0 0.1\n"
                "17 0.9 0.9 0.9 255 0 0 0.1\n"
                "18 0.1 0.9 0.9 255 0 0 0.1\n"
                "19 0.9 0.1 0.9 255 0 0 0.1\n"
                "20 0.9 0.9 0.1 255 0 0 0.1\n",
                encoding="utf-8",
            )

            fallback = BoxPrimitiveFallback(
                settings=BoxPrimitiveFallbackSettings(
                    enabled=True,
                    min_selected_images=3,
                )
            )
            result = fallback.build_from_images(
                project_id="demo",
                selected_images_dir=selected_images_dir,
                output_dir=output_dir,
                output_format=OutputFormat.OBJ,
                source_reason="unit_test_sparse",
            )

            self.assertTrue(result.metadata["approximate_geometry_fallback"]["sparse_reference"]["used"])

    def test_builds_textured_glb_from_input_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected_images_dir = root / "selected_images"
            selected_images_dir.mkdir(parents=True, exist_ok=True)
            for index in range(5):
                self._write_box_like_image(selected_images_dir / f"sample_{index + 1}.jpg", offset=index * 5)

            output_dir = root / "output"
            fallback = BoxPrimitiveFallback(
                settings=BoxPrimitiveFallbackSettings(
                    enabled=True,
                    min_selected_images=3,
                )
            )
            result = fallback.build_from_images(
                project_id="demo",
                selected_images_dir=selected_images_dir,
                output_dir=output_dir,
                output_format=OutputFormat.GLB,
                source_reason="unit_test_textured",
            )

            self.assertEqual(result.model_path.suffix.lower(), ".glb")
            self.assertTrue(result.model_path.exists())
            self.assertGreater(result.model_path.stat().st_size, 0)
            captured = result.metadata["approximate_geometry_fallback"]["captured_texture"]
            self.assertTrue(captured["applied"])
            self.assertTrue(Path(captured["atlas_path"]).exists())
            self.assertTrue(captured["enabled"])

    def test_builds_clean_glb_when_texture_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected_images_dir = root / "selected_images"
            selected_images_dir.mkdir(parents=True, exist_ok=True)
            for index in range(5):
                self._write_box_like_image(selected_images_dir / f"sample_{index + 1}.jpg", offset=index * 5)

            output_dir = root / "output"
            fallback = BoxPrimitiveFallback(
                settings=BoxPrimitiveFallbackSettings(
                    enabled=True,
                    min_selected_images=3,
                    texture_enabled=False,
                )
            )
            result = fallback.build_from_images(
                project_id="demo",
                selected_images_dir=selected_images_dir,
                output_dir=output_dir,
                output_format=OutputFormat.GLB,
                source_reason="unit_test_clean_glb",
            )

            self.assertEqual(result.model_path.suffix.lower(), ".glb")
            self.assertTrue(result.model_path.exists())
            self.assertGreater(result.model_path.stat().st_size, 0)
            captured = result.metadata["approximate_geometry_fallback"]["captured_texture"]
            self.assertFalse(captured["enabled"])
            self.assertFalse(captured["applied"])
            self.assertIsNone(captured["atlas_path"])

    def test_refine_texture_crop_from_saturation_recovers_center_box(self) -> None:
        image = Image.new("RGB", (720, 1280), (238, 238, 238))
        draw = ImageDraw.Draw(image)
        draw.rectangle((250, 520, 520, 760), fill=(245, 245, 245), outline=(30, 30, 30), width=5)
        draw.rectangle((250, 520, 520, 600), fill=(24, 78, 188))
        draw.text((270, 545), "Azitromicina", fill=(245, 245, 245))
        draw.rectangle((0, 0, 90, 420), fill=(32, 35, 39))
        draw.rectangle((610, 0, 720, 360), fill=(44, 38, 55))

        fallback = BoxPrimitiveFallback(
            settings=BoxPrimitiveFallbackSettings(
                enabled=True,
                min_selected_images=3,
            )
        )

        refined = fallback._refine_texture_crop_from_saturation(
            image=image,
            anchor_crop_box=(0, 420, 720, 1080),
            expected_ratio=1.6,
        )
        self.assertIsNotNone(refined)
        left, top, right, bottom = refined or (0, 0, 0, 0)

        width = right - left
        height = bottom - top
        self.assertGreater(width, 180)
        self.assertGreater(height, 160)
        self.assertLess((width * height) / float(720 * 1280), 0.65)
        self.assertLessEqual(
            fallback._count_bbox_border_touches((left, top, right, bottom), (720, 1280), margin=2),
            1,
        )

        center_x = (left + right) * 0.5
        center_y = (top + bottom) * 0.5
        self.assertGreater(center_x, 220)
        self.assertLess(center_x, 550)
        self.assertGreater(center_y, 500)
        self.assertLess(center_y, 860)


if __name__ == "__main__":
    unittest.main()
