from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from app.algorithms.feature_matcher import FeatureMatcher
from app.algorithms.image_preprocessor import ImagePreprocessor
from app.algorithms.reconstruction_pipeline import ReconstructionPipeline
from app.core.errors import ProcessingError
from app.models.schemas import OutputFormat
from app.services.engines.base_engine import ReconstructionResult
from app.services.engines.colmap_engine import ColmapReconstructionEngine
from app.services.engines.factory import build_reconstruction_engines
from app.services.engines.mock_engine import MockReconstructionEngine
from app.services.processing_service import ProcessingService


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
            result = engine.reconstruct(
                project_id="demo_engine",
                images_dir=images_dir,
                output_dir=output_dir,
                output_format=OutputFormat.GLB,
            )

            self.assertTrue(result.model_path.exists())
            self.assertEqual(result.model_path.read_bytes()[:4], b"glTF")
            self.assertTrue((output_dir / "pipeline" / "demo_engine_pipeline_report.json").exists())
            self.assertEqual(result.metadata["engine"], "mock")
            self.assertEqual(result.metadata["reconstruction_type"], "synthetic_pipeline")
            self.assertEqual(result.metadata["current_stage"], "completed")
            self.assertEqual(result.metadata["progress"], 1.0)


class EngineFactoryTests(unittest.TestCase):
    class _Settings:
        processing_engine = "colmap"
        simulation_delay_seconds = 0
        colmap_path = None
        colmap_binary = "colmap"
        colmap_timeout_seconds = 120
        colmap_use_gpu = False
        colmap_camera_model = "SIMPLE_RADIAL"
        colmap_single_camera = True
        colmap_fallback_to_mock = False

    def test_factory_keeps_colmap_when_explicit_and_unavailable(self) -> None:
        settings = self._Settings()
        with patch.object(ColmapReconstructionEngine, "is_available", return_value=False):
            primary, fallback = build_reconstruction_engines(settings)

        self.assertIsInstance(primary, ColmapReconstructionEngine)
        self.assertIsNone(fallback)

    def test_factory_uses_mock_in_auto_mode_when_colmap_is_unavailable(self) -> None:
        settings = self._Settings()
        settings.processing_engine = "auto"
        with patch.object(ColmapReconstructionEngine, "is_available", return_value=False):
            primary, fallback = build_reconstruction_engines(settings)

        self.assertIsInstance(primary, MockReconstructionEngine)
        self.assertIsNone(fallback)

    def test_factory_uses_colmap_with_mock_fallback_in_auto_mode_when_available(self) -> None:
        settings = self._Settings()
        settings.processing_engine = "auto"
        settings.colmap_fallback_to_mock = True
        with patch.object(ColmapReconstructionEngine, "is_available", return_value=True):
            primary, fallback = build_reconstruction_engines(settings)

        self.assertIsInstance(primary, ColmapReconstructionEngine)
        self.assertIsInstance(fallback, MockReconstructionEngine)


class ProcessingFallbackTests(unittest.TestCase):
    class _AutoSettings:
        processing_engine = "auto"
        simulation_delay_seconds = 0
        colmap_path = None
        colmap_binary = "colmap"
        colmap_timeout_seconds = 120
        colmap_use_gpu = False
        colmap_camera_model = "SIMPLE_RADIAL"
        colmap_single_camera = True
        colmap_fallback_to_mock = True

    class _ExplicitColmapSettings:
        processing_engine = "colmap"
        simulation_delay_seconds = 0
        colmap_path = None
        colmap_binary = "colmap"
        colmap_timeout_seconds = 120
        colmap_use_gpu = False
        colmap_camera_model = "SIMPLE_RADIAL"
        colmap_single_camera = True
        colmap_fallback_to_mock = False

    class _FailingEngine:
        name = "colmap"

        def reconstruct(self, *_args, **_kwargs):
            raise RuntimeError("COLMAP mapper failed")

    class _SuccessfulEngine:
        name = "mock"

        def __init__(self, model_path: Path) -> None:
            self.model_path = model_path

        def reconstruct(self, _project_id, _images_dir, _output_dir, output_format, progress_callback=None):
            self.model_path.write_text("o mock", encoding="utf-8")
            return ReconstructionResult(
                engine_name="mock",
                requested_output_format=output_format,
                model_path=self.model_path,
                metadata={
                    "engine": "mock",
                    "processing_seconds": 0.25,
                },
            )

    def _build_services(self, images_dir: Path, output_dir: Path) -> tuple[MagicMock, MagicMock]:
        project_service = MagicMock()
        project_service.get_project.return_value.processing_metadata = {}
        storage_service = MagicMock()
        storage_service.get_images_dir.return_value = images_dir
        storage_service.get_output_dir.return_value = output_dir
        return project_service, storage_service

    def test_processing_service_falls_back_to_mock_only_in_auto_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            (images_dir / "sample.jpg").write_bytes(b"jpg")
            output_dir = root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / "fallback_model.obj"

            project_service, storage_service = self._build_services(images_dir, output_dir)

            with patch(
                "app.services.processing_service.build_reconstruction_engines",
                return_value=(self._FailingEngine(), self._SuccessfulEngine(model_path)),
            ):
                service = ProcessingService(project_service, storage_service, self._AutoSettings())

            service._run_reconstruction_job("demo-project", OutputFormat.OBJ)

            project_service.mark_completed.assert_called_once()
            project_service.mark_failed.assert_not_called()
            metadata = project_service.mark_completed.call_args.kwargs["processing_metadata"]
            self.assertEqual(metadata["engine"], "mock")
            self.assertTrue(metadata["fallback"]["used"])
            self.assertEqual(metadata["fallback"]["from_engine"], "colmap")
            storage_service.clear_output_files.assert_called_once_with("demo-project")

    def test_processing_service_marks_failed_when_colmap_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            (images_dir / "sample.jpg").write_bytes(b"jpg")
            output_dir = root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / "unused_fallback.obj"

            project_service, storage_service = self._build_services(images_dir, output_dir)

            with patch(
                "app.services.processing_service.build_reconstruction_engines",
                return_value=(self._FailingEngine(), self._SuccessfulEngine(model_path)),
            ):
                service = ProcessingService(project_service, storage_service, self._ExplicitColmapSettings())

            service._run_reconstruction_job("demo-project", OutputFormat.OBJ)

            project_service.mark_completed.assert_not_called()
            project_service.mark_failed.assert_called_once()
            metadata = project_service.mark_failed.call_args.kwargs["processing_metadata"]
            self.assertEqual(metadata["engine"], "colmap")
            self.assertEqual(metadata["current_stage"], "failed")
            self.assertEqual(metadata["failure_reason"], "COLMAP mapper failed")


class ColmapEngineTests(unittest.TestCase):
    def test_colmap_engine_detects_windows_candidates_via_help_command(self) -> None:
        engine = ColmapReconstructionEngine(colmap_binary="colmap")

        def fake_probe(command, capture_output, text, encoding, errors, timeout, check):
            if command[0] == "colmap":
                raise FileNotFoundError("not found")
            return subprocess.CompletedProcess(command, 0, stdout="COLMAP 3.9", stderr="")

        with patch("app.services.engines.colmap_engine.subprocess.run", side_effect=fake_probe):
            self.assertTrue(engine.is_available())

        self.assertEqual(engine.detected_binary, "colmap.exe")

    def test_colmap_engine_exports_sparse_glb_with_logs_metrics_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            (images_dir / "img_1.jpg").write_bytes(b"img1")
            (images_dir / "img_2.jpg").write_bytes(b"img2")
            output_dir = root / "output"

            engine = ColmapReconstructionEngine(
                colmap_binary="C:/COLMAP/colmap.exe",
                timeout_seconds=30,
            )
            progress_updates: list[dict[str, object]] = []

            def fake_run(command, capture_output, text, encoding, errors, timeout, check):
                command_name = command[1]
                if command[-1] == "-h":
                    if command_name == "feature_extractor":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureExtraction.use_gpu", stderr="")
                    if command_name == "exhaustive_matcher":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureMatching.use_gpu", stderr="")
                if command_name == "feature_extractor":
                    database_path = Path(command[command.index("--database_path") + 1])
                    database_path.parent.mkdir(parents=True, exist_ok=True)
                    database_path.write_bytes(b"db")
                elif command_name == "mapper":
                    sparse_dir = Path(command[command.index("--output_path") + 1]) / "0"
                    sparse_dir.mkdir(parents=True, exist_ok=True)
                    for name in ("cameras.bin", "images.bin", "points3D.bin"):
                        (sparse_dir / name).write_bytes(b"bin")
                elif command_name == "model_converter":
                    output_path = Path(command[command.index("--output_path") + 1])
                    output_type = command[command.index("--output_type") + 1]
                    if output_type == "TXT":
                        output_path.mkdir(parents=True, exist_ok=True)
                        (output_path / "cameras.txt").write_text(
                            "# cameras\n1 SIMPLE_RADIAL 800 600 500 400 300 0.01\n",
                            encoding="utf-8",
                        )
                        (output_path / "images.txt").write_text(
                            "# images\n1 1 0 0 0 0 0 0 1 img_1.jpg\n0 0 -1\n"
                            "2 1 0 0 0 0 0 0 1 img_2.jpg\n0 0 -1\n",
                            encoding="utf-8",
                        )
                        (output_path / "points3D.txt").write_text(
                            "# points\n"
                            "1 0.0 0.0 0.0 255 0 0 0.1 1 1\n"
                            "2 1.0 0.0 0.0 0 255 0 0.2 1 2\n"
                            "3 0.0 1.0 0.0 0 0 255 0.3 2 3\n",
                            encoding="utf-8",
                        )
                    elif output_type == "PLY":
                        output_path.write_text("ply\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with patch.object(engine, "detect_binary", return_value="C:/COLMAP/colmap.exe"):
                with patch("app.services.engines.colmap_engine.subprocess.run", side_effect=fake_run):
                    result = engine.reconstruct(
                        "demo",
                        images_dir,
                        output_dir,
                        OutputFormat.GLB,
                        progress_callback=lambda update: progress_updates.append(dict(update)),
                    )

            self.assertTrue(result.model_path.exists())
            self.assertEqual(result.model_path.read_bytes()[:4], b"glTF")
            self.assertEqual(result.metadata["engine"], "colmap")
            self.assertEqual(result.metadata["reconstruction_type"], "sparse_photogrammetry")
            self.assertEqual(result.metadata["registered_image_count"], 2)
            self.assertEqual(result.metadata["camera_count"], 1)
            self.assertEqual(result.metadata["point_count"], 3)
            self.assertEqual(result.metadata["metrics"]["reconstructed_camera_count"], 2)
            self.assertEqual(result.metadata["metrics"]["point_3d_count"], 3)
            self.assertEqual(result.metadata["current_stage"], "completed")
            self.assertEqual(result.metadata["progress"], 1.0)
            self.assertTrue((output_dir / "demo_colmap_metadata.json").exists())
            self.assertTrue((output_dir / "demo_sparse.ply").exists())
            self.assertTrue((output_dir / "demo_model.obj").exists())
            self.assertTrue((output_dir / "logs" / "colmap" / "feature_extractor.stdout.log").exists())
            self.assertEqual(
                [item["name"] for item in result.metadata["commands"]][:4],
                [
                    "feature_extractor",
                    "exhaustive_matcher",
                    "mapper",
                    "model_converter_txt",
                ],
            )
            self.assertIn("mapper", [update.get("current_stage") for update in progress_updates])
            self.assertEqual(progress_updates[-1].get("current_stage"), "completed")

    def test_colmap_engine_fails_when_mapper_does_not_create_sparse_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            (images_dir / "img_1.jpg").write_bytes(b"img1")
            (images_dir / "img_2.jpg").write_bytes(b"img2")
            output_dir = root / "output"

            engine = ColmapReconstructionEngine(
                colmap_binary="C:/COLMAP/colmap.exe",
                timeout_seconds=30,
            )

            def fake_run(command, capture_output, text, encoding, errors, timeout, check):
                command_name = command[1]
                if command[-1] == "-h":
                    if command_name == "feature_extractor":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureExtraction.use_gpu", stderr="")
                    if command_name == "exhaustive_matcher":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureMatching.use_gpu", stderr="")
                if command_name == "feature_extractor":
                    database_path = Path(command[command.index("--database_path") + 1])
                    database_path.parent.mkdir(parents=True, exist_ok=True)
                    database_path.write_bytes(b"db")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with patch.object(engine, "detect_binary", return_value="C:/COLMAP/colmap.exe"):
                with patch("app.services.engines.colmap_engine.subprocess.run", side_effect=fake_run):
                    with self.assertRaises(ProcessingError) as context:
                        engine.reconstruct("demo", images_dir, output_dir, OutputFormat.OBJ)

            self.assertIn("sparse/0", str(context.exception))


if __name__ == "__main__":
    unittest.main()