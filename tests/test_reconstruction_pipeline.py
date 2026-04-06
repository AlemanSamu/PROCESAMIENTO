from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.algorithms.image_preprocessor import ImagePreprocessor
from app.algorithms.feature_matcher import FeatureMatcher
from app.algorithms.reconstruction_pipeline import ReconstructionPipeline
from app.models.schemas import OutputFormat
from app.services.engines.mock_engine import MockReconstructionEngine


class ReconstructionPipelineTests(unittest.TestCase):
    def _build_sample_images(self, base_dir: Path, count: int = 3) -> Path:
        images_dir = base_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image_path = images_dir / f"image_{index + 1}.png"
            canvas = Image.new("RGB", (160, 160), "white")
            draw = ImageDraw.Draw(canvas)
            shift = index * 5
            draw.rectangle((28 + shift, 34, 120 + shift, 122), outline="black", width=4)
            draw.ellipse((46, 46 + shift, 96, 96 + shift), outline="navy", width=3)
            draw.line((18, 20 + shift, 142, 138 - shift), fill="crimson", width=3)
            canvas.save(image_path)
        return images_dir

    def _build_gradient_image(self, base_dir: Path) -> Path:
        images_dir = base_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        image_path = images_dir / "gradient.png"
        canvas = Image.new("RGB", (128, 128), "black")
        draw = ImageDraw.Draw(canvas)
        for x in range(128):
            value = int((x / 127) * 255)
            draw.line((x, 0, x, 127), fill=(value, value, value))
        canvas.save(image_path)
        return images_dir

    def test_preprocessor_reports_real_metrics_and_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = self._build_gradient_image(root)
            work_dir = root / "work"
            work_dir.mkdir(parents=True, exist_ok=True)

            preprocessor = ImagePreprocessor()
            preprocessed_images, report = preprocessor.run(images_dir, work_dir)

            self.assertEqual(len(preprocessed_images), 1)
            self.assertEqual(report.mode, "real")
            self.assertEqual(report.metrics["mode"], "real")
            self.assertGreater(preprocessed_images[0].source.width, 0)
            self.assertGreater(preprocessed_images[0].source.height, 0)
            self.assertEqual(preprocessed_images[0].source.pixel_count, 128 * 128)
            self.assertGreater(preprocessed_images[0].brightness, 0.2)
            self.assertLess(preprocessed_images[0].contrast, 0.6)

    def test_feature_matcher_uses_real_pixel_data_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = self._build_sample_images(root, count=2)
            work_dir = root / "work"
            work_dir.mkdir(parents=True, exist_ok=True)

            preprocessor = ImagePreprocessor()
            preprocessed_images, preprocessing_report = preprocessor.run(images_dir, work_dir)
            self.assertEqual(preprocessing_report.mode, "real")

            matcher = FeatureMatcher()
            feature_sets, matches, report = matcher.run(preprocessed_images, work_dir)

            self.assertEqual(report.mode, "real")
            self.assertEqual(len(feature_sets), 2)
            self.assertEqual(len(matches), 1)
            self.assertGreater(len(matches[0].correspondences), 0)
            self.assertGreater(matches[0].matched_pairs, 0)

    def test_pipeline_exports_valid_glb_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = self._build_sample_images(root)
            output_dir = root / "output"

            pipeline = ReconstructionPipeline()
            result = pipeline.execute("demo", images_dir, output_dir, OutputFormat.GLB)

            self.assertTrue(result.model_path.exists())
            self.assertEqual(result.model_path.read_bytes()[:4], b"glTF")
            self.assertTrue(result.report_path.exists())
            self.assertEqual(result.image_count, 3)
            self.assertEqual(result.output_format, OutputFormat.GLB)
            self.assertGreaterEqual(len(result.stage_results), 6)
            self.assertEqual(result.stage_results[0].name, "image_validation_and_preprocessing")
            self.assertEqual(result.stage_results[-1].name, "model_export")
            self.assertEqual(result.stage_results[0].mode, "real")
            self.assertEqual(result.stage_results[1].mode, "real")
            self.assertEqual(result.stage_results[2].mode, "real")
            self.assertEqual(result.stage_results[3].mode, "real")
            self.assertEqual(result.stage_results[4].mode, "synthetic")
            self.assertEqual(result.stage_results[5].mode, "real")

    def test_pipeline_exports_obj_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = self._build_sample_images(root)
            output_dir = root / "output"

            pipeline = ReconstructionPipeline()
            result = pipeline.execute("demo_obj", images_dir, output_dir, OutputFormat.OBJ)

            self.assertTrue(result.model_path.exists())
            self.assertTrue(result.model_path.suffix.lower().endswith("obj"))
            content = result.model_path.read_text(encoding="utf-8")
            self.assertIn("OBJ generado por ReconstructionPipeline", content)
            self.assertIn("f 1 2 3", content)

    def test_mock_engine_delegates_to_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = self._build_sample_images(root)
            output_dir = root / "output"

            engine = MockReconstructionEngine(delay_seconds=0)
            model_path = engine.reconstruct(
                project_id="demo_engine",
                images_dir=images_dir,
                output_dir=output_dir,
                output_format=OutputFormat.GLB,
            )

            self.assertTrue(model_path.exists())
            self.assertEqual(model_path.read_bytes()[:4], b"glTF")
            self.assertTrue((output_dir / "pipeline" / "demo_engine_pipeline_report.json").exists())


if __name__ == "__main__":
    unittest.main()
