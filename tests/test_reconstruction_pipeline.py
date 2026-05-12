from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from app.algorithms.box_primitive_fallback import BoxPrimitiveFallbackResult
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
        colmap_enable_dense_stages = True
        colmap_camera_model = "SIMPLE_RADIAL"
        colmap_single_camera = True
        colmap_fallback_to_mock = False
        colmap_require_dense_reconstruction = False

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

    def test_factory_passes_dense_requirement_to_colmap_engine(self) -> None:
        settings = self._Settings()
        settings.colmap_require_dense_reconstruction = True
        with patch.object(ColmapReconstructionEngine, "is_available", return_value=True):
            primary, _fallback = build_reconstruction_engines(settings)

        self.assertIsInstance(primary, ColmapReconstructionEngine)
        self.assertTrue(primary.require_dense_reconstruction)

    def test_factory_passes_dense_stage_toggle_to_colmap_engine(self) -> None:
        settings = self._Settings()
        settings.colmap_enable_dense_stages = False
        with patch.object(ColmapReconstructionEngine, "is_available", return_value=True):
            primary, _fallback = build_reconstruction_engines(settings)

        self.assertIsInstance(primary, ColmapReconstructionEngine)
        self.assertFalse(primary.enable_dense_stages)


class ProcessingFallbackTests(unittest.TestCase):
    class _AutoSettings:
        processing_engine = "auto"
        simulation_delay_seconds = 0
        image_validation_enabled = False
        image_validation_min_images_required = 1
        colmap_path = None
        colmap_binary = "colmap"
        colmap_timeout_seconds = 120
        colmap_use_gpu = False
        colmap_enable_dense_stages = True
        colmap_camera_model = "SIMPLE_RADIAL"
        colmap_single_camera = True
        colmap_fallback_to_mock = True
        colmap_require_dense_reconstruction = False

    class _ExplicitColmapSettings:
        processing_engine = "colmap"
        simulation_delay_seconds = 0
        image_validation_enabled = False
        image_validation_min_images_required = 1
        colmap_path = None
        colmap_binary = "colmap"
        colmap_timeout_seconds = 120
        colmap_use_gpu = False
        colmap_enable_dense_stages = True
        colmap_camera_model = "SIMPLE_RADIAL"
        colmap_single_camera = True
        colmap_fallback_to_mock = False
        colmap_require_dense_reconstruction = False

    class _BoxFallbackSettings:
        processing_engine = "colmap"
        simulation_delay_seconds = 0
        image_validation_enabled = False
        image_validation_min_images_required = 1
        colmap_path = None
        colmap_binary = "colmap"
        colmap_timeout_seconds = 120
        colmap_use_gpu = False
        colmap_enable_dense_stages = True
        colmap_camera_model = "SIMPLE_RADIAL"
        colmap_single_camera = True
        colmap_fallback_to_mock = False
        colmap_require_dense_reconstruction = False
        primitive_box_fallback_enabled = True
        primitive_box_fallback_min_selected_images = 3
        primitive_box_fallback_analysis_max_width = 256
        primitive_box_fallback_min_foreground_ratio = 0.02
        primitive_box_fallback_on_incoherent_output = False
        primitive_box_fallback_incoherent_min_registered_images = 8
        primitive_box_fallback_incoherent_min_sparse_points = 1200
        primitive_box_fallback_incoherent_min_points_per_registered_image = 180
        primitive_box_fallback_incoherent_min_faces = 20
        primitive_box_fallback_incoherent_max_faces = 700
        primitive_box_fallback_incoherent_max_extent_ratio = 6.0
        primitive_box_fallback_incoherent_min_bbox_fill_ratio = 0.12
        primitive_box_fallback_incoherent_max_bbox_fill_ratio = 1.05
        primitive_box_fallback_replace_sparse_bounding_box = False

    class _BoxFallbackValidationSettings(_BoxFallbackSettings):
        image_validation_enabled = True
        image_validation_min_images_required = 6

    class _IncoherentSparseSettings(_BoxFallbackSettings):
        primitive_box_fallback_on_incoherent_output = True
        primitive_box_fallback_replace_sparse_bounding_box = True

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

    class _SparseFallbackIrregularEngine:
        name = "colmap"

        def __init__(self, model_path: Path) -> None:
            self.model_path = model_path

        def reconstruct(self, _project_id, _images_dir, _output_dir, output_format, progress_callback=None):
            self.model_path.write_text("o sparse\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
            return ReconstructionResult(
                engine_name="colmap",
                requested_output_format=output_format,
                model_path=self.model_path,
                metadata={
                    "engine": "colmap",
                    "processing_seconds": 3.1,
                    "current_stage": "completed_with_fallback",
                    "reconstruction_type": "sparse_photogrammetry_mesh_fallback",
                    "registered_image_count": 6,
                    "point_count": 640,
                    "mesh_face_count": 839,
                    "mesh_vertex_count": 351,
                    "metrics": {
                        "mesh_face_count": 839,
                        "mesh_vertex_count": 351,
                        "reconstructed_camera_count": 6,
                        "point_3d_count": 640,
                    },
                    "sparse_fallback": {
                        "used": True,
                        "mesh_method": "delaunay_mesher_sparse",
                        "shape_diagnostics": {
                            "extent_ratio_max_min": 9.4,
                            "mesh_volume_to_bbox_volume_ratio": 0.04,
                        },
                    },
                },
            )

    @staticmethod
    def _write_valid_image(path: Path, offset: int = 0) -> None:
        canvas = Image.new("RGB", (800, 600), (120, 120, 120))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((80 + offset, 60, 720, 520), outline="black", fill=(170, 170, 170), width=5)
        draw.ellipse((180, 160 + offset, 520, 460 + offset), outline="blue", fill=(90, 90, 160), width=4)
        draw.line((100, 100 + offset, 700, 500 - offset), fill="red", width=5)
        canvas.save(path)

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
            self._write_valid_image(images_dir / "sample.jpg")
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
            self._write_valid_image(images_dir / "sample.jpg")
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

    def test_processing_service_recovers_with_primitive_box_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            self._write_valid_image(images_dir / "sample_1.jpg")
            self._write_valid_image(images_dir / "sample_2.jpg", offset=10)
            self._write_valid_image(images_dir / "sample_3.jpg", offset=20)
            self._write_valid_image(images_dir / "sample_4.jpg", offset=30)
            output_dir = root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            project_service, storage_service = self._build_services(images_dir, output_dir)

            with patch(
                "app.services.processing_service.build_reconstruction_engines",
                return_value=(self._FailingEngine(), None),
            ):
                service = ProcessingService(project_service, storage_service, self._BoxFallbackSettings())

            service._run_reconstruction_job("demo-project", OutputFormat.OBJ)

            project_service.mark_completed.assert_called_once()
            project_service.mark_failed.assert_not_called()
            metadata = project_service.mark_completed.call_args.kwargs["processing_metadata"]
            self.assertEqual(metadata["current_stage"], "completed_with_fallback")
            self.assertEqual(metadata["method_used"], "primitive_box")
            self.assertTrue(metadata["fallback_used"])
            self.assertEqual(metadata["reconstruction_type"], "approximate_box_primitive_fallback")
            self.assertTrue(metadata["approximate_geometry_fallback"]["used"])
            self.assertEqual(project_service.mark_completed.call_args.args[1], OutputFormat.OBJ)
            self.assertTrue(Path(metadata["final_model_path"]).exists())
            self.assertTrue(Path(metadata["artifacts"]["box_fallback_report"]).exists())

    def test_processing_service_prioritizes_primitive_box_when_sparse_result_is_incoherent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            self._write_valid_image(images_dir / "sample_1.jpg")
            self._write_valid_image(images_dir / "sample_2.jpg", offset=10)
            self._write_valid_image(images_dir / "sample_3.jpg", offset=20)
            self._write_valid_image(images_dir / "sample_4.jpg", offset=30)
            output_dir = root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            sparse_model_path = output_dir / "demo-project_model.obj"

            project_service, storage_service = self._build_services(images_dir, output_dir)

            with patch(
                "app.services.processing_service.build_reconstruction_engines",
                return_value=(self._SparseFallbackIrregularEngine(sparse_model_path), None),
            ):
                service = ProcessingService(project_service, storage_service, self._IncoherentSparseSettings())

            fallback_model_path = output_dir / "demo-project_box_model.obj"
            fallback_model_path.write_text("o box\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
            fallback_report = output_dir / "pipeline" / "demo-project_box_fallback_report.json"
            fallback_report.parent.mkdir(parents=True, exist_ok=True)
            fallback_report.write_text("{}", encoding="utf-8")
            fallback_result = BoxPrimitiveFallbackResult(
                model_path=fallback_model_path,
                output_format=OutputFormat.OBJ,
                report_path=fallback_report,
                metadata={
                    "method_used": "primitive_box",
                    "reconstruction_type": "approximate_box_primitive_fallback",
                    "metrics": {"mesh_face_count": 12, "mesh_vertex_count": 8},
                },
            )

            with patch.object(service._box_primitive_fallback, "build_from_images", return_value=fallback_result):
                service._run_reconstruction_job("demo-project", OutputFormat.OBJ)

            project_service.mark_completed.assert_called_once()
            project_service.mark_failed.assert_not_called()
            metadata = project_service.mark_completed.call_args.kwargs["processing_metadata"]
            self.assertEqual(metadata["current_stage"], "completed_with_fallback")
            self.assertEqual(metadata["method_used"], "primitive_box")
            self.assertTrue(metadata["fallback_used"])
            self.assertEqual(metadata["reconstruction_type"], "approximate_box_primitive_fallback")
            self.assertEqual(metadata["reason_code"], "fallback_box_used")
            self.assertEqual(metadata["failed_stage"], "quality_gate_incoherent_output")
            self.assertTrue(Path(metadata["final_model_path"]).exists())
            self.assertEqual(Path(metadata["final_model_path"]).name, fallback_model_path.name)

    def test_processing_service_keeps_glb_as_canonical_when_box_fallback_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            self._write_valid_image(images_dir / "sample_1.jpg")
            self._write_valid_image(images_dir / "sample_2.jpg", offset=10)
            self._write_valid_image(images_dir / "sample_3.jpg", offset=20)
            self._write_valid_image(images_dir / "sample_4.jpg", offset=30)
            output_dir = root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            project_service, storage_service = self._build_services(images_dir, output_dir)

            with patch(
                "app.services.processing_service.build_reconstruction_engines",
                return_value=(self._FailingEngine(), None),
            ):
                service = ProcessingService(project_service, storage_service, self._BoxFallbackSettings())

            with patch.object(
                service._box_primitive_fallback,
                "build_from_images",
                side_effect=ProcessingError(
                    "No se pudo generar GLB fallback",
                    reason_code="fallback_glb_failed",
                    current_stage="primitive_box_fallback",
                ),
            ) as mocked_fallback:
                service._run_reconstruction_job("demo-project", OutputFormat.GLB)

            project_service.mark_completed.assert_not_called()
            project_service.mark_failed.assert_called_once()
            self.assertEqual(mocked_fallback.call_count, 1)
            self.assertEqual(mocked_fallback.call_args.kwargs["output_format"], OutputFormat.GLB)

    def test_processing_service_does_not_use_box_fallback_when_input_validation_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            self._write_valid_image(images_dir / "sample_1.jpg")
            output_dir = root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            project_service, storage_service = self._build_services(images_dir, output_dir)

            with patch(
                "app.services.processing_service.build_reconstruction_engines",
                return_value=(self._FailingEngine(), None),
            ):
                service = ProcessingService(project_service, storage_service, self._BoxFallbackValidationSettings())

            service._run_reconstruction_job("demo-project", OutputFormat.OBJ)

            project_service.mark_completed.assert_not_called()
            project_service.mark_failed.assert_called_once()
            metadata = project_service.mark_failed.call_args.kwargs["processing_metadata"]
            self.assertEqual(metadata["reason_code"], "input_validation_failed")
            self.assertEqual(metadata["current_stage"], "input_validation_failed")

    def test_processing_service_writes_technical_evidence_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            self._write_valid_image(images_dir / "sample_1.jpg")
            self._write_valid_image(images_dir / "sample_2.jpg", offset=14)
            output_dir = root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / "final_model.obj"

            project_service, storage_service = self._build_services(images_dir, output_dir)

            class _MetricsEnabledSettings(self._ExplicitColmapSettings):
                metrics_evidence_enabled = True
                metrics_evidence_root = root / "experiments"
                metrics_experiment_variant = "enhanced"
                metrics_experiment_scenario = "auto"

            with patch(
                "app.services.processing_service.build_reconstruction_engines",
                return_value=(self._SuccessfulEngine(model_path), None),
            ):
                service = ProcessingService(project_service, storage_service, _MetricsEnabledSettings())

            service._run_reconstruction_job("demo-project", OutputFormat.OBJ)

            project_service.mark_completed.assert_called_once()
            metadata = project_service.mark_completed.call_args.kwargs["processing_metadata"]
            evidence_path = Path(metadata["artifacts"]["technical_evidence_report"])
            self.assertTrue(evidence_path.exists())
            self.assertEqual(metadata["technical_evidence_report_path"], str(evidence_path))

            run_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(run_payload["run_info"]["project_id"], "demo-project")
            self.assertEqual(run_payload["run_info"]["variant"], "enhanced")
            self.assertEqual(run_payload["run_info"]["status"], "completed")

            history_path = root / "experiments" / "processing_runs.ndjson"
            self.assertTrue(history_path.exists())
            history_lines = [line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(history_lines), 1)

    def test_processing_service_does_not_fallback_when_registered_images_are_insufficient(self) -> None:
        class _InsufficientRegisteredImagesEngine:
            name = "colmap"

            def reconstruct(self, _project_id, _images_dir, _output_dir, _output_format, progress_callback=None):
                raise ProcessingError(
                    "COLMAP no logro registrar suficientes imagenes para reconstruir el modelo. "
                    "Intenta capturar mas fotos con mejor traslape, buena iluminacion y mas textura visual.",
                    reason_code="insufficient_registered_images",
                    current_stage="mapper_failed_insufficient_registered_images",
                    metadata={
                        "current_stage": "mapper_failed_insufficient_registered_images",
                        "reason_code": "insufficient_registered_images",
                        "registered_image_count": 0,
                        "status_message": (
                            "COLMAP no logro registrar suficientes imagenes para reconstruir el modelo. "
                            "Intenta capturar mas fotos con mejor traslape, buena iluminacion y mas textura visual."
                        ),
                    },
                    allow_fallback=False,
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            self._write_valid_image(images_dir / "sample_1.jpg")
            self._write_valid_image(images_dir / "sample_2.jpg", offset=20)
            output_dir = root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / "fallback_model.obj"

            project_service, storage_service = self._build_services(images_dir, output_dir)

            with patch(
                "app.services.processing_service.build_reconstruction_engines",
                return_value=(_InsufficientRegisteredImagesEngine(), self._SuccessfulEngine(model_path)),
            ):
                service = ProcessingService(project_service, storage_service, self._AutoSettings())

            service._run_reconstruction_job("demo-project", OutputFormat.OBJ)

            project_service.mark_completed.assert_not_called()
            project_service.mark_failed.assert_called_once()
            metadata = project_service.mark_failed.call_args.kwargs["processing_metadata"]
            self.assertEqual(metadata["current_stage"], "mapper_failed_insufficient_registered_images")
            self.assertEqual(metadata["failed_stage"], "mapper_failed_insufficient_registered_images")
            self.assertEqual(metadata["reason_code"], "insufficient_registered_images")
            self.assertEqual(metadata["registered_image_count"], 0)
            self.assertEqual(metadata["metrics"]["registered_image_count"], 0)
            self.assertIn("traslape", metadata["status_message"])
            self.assertFalse(metadata["fallback"]["used"])
            storage_service.clear_output_files.assert_called_once_with("demo-project")

    def test_processing_service_failure_includes_execution_report_and_retry_hint(self) -> None:
        class _TimeoutEngine:
            name = "colmap"

            def reconstruct(self, _project_id, _images_dir, _output_dir, _output_format, progress_callback=None):
                raise ProcessingError(
                    "COLMAP agoto el tiempo de espera en mapper.",
                    reason_code="colmap_command_timeout",
                    current_stage="mapper",
                    metadata={
                        "current_stage": "mapper",
                        "reason_code": "colmap_command_timeout",
                        "failed_command": "mapper",
                        "logs": {
                            "stdout_path": "C:/tmp/mapper.stdout.log",
                            "stderr_path": "C:/tmp/mapper.stderr.log",
                        },
                        "status_message": "Tiempo de espera agotado en etapa 'mapper'.",
                    },
                    retryable=True,
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            self._write_valid_image(images_dir / "sample_1.jpg")
            self._write_valid_image(images_dir / "sample_2.jpg", offset=20)
            output_dir = root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            project_service, storage_service = self._build_services(images_dir, output_dir)

            with patch(
                "app.services.processing_service.build_reconstruction_engines",
                return_value=(_TimeoutEngine(), None),
            ):
                service = ProcessingService(project_service, storage_service, self._ExplicitColmapSettings())

            service._run_reconstruction_job("demo-project", OutputFormat.OBJ)

            project_service.mark_completed.assert_not_called()
            project_service.mark_failed.assert_called_once()
            metadata = project_service.mark_failed.call_args.kwargs["processing_metadata"]

            self.assertEqual(metadata["current_stage"], "mapper")
            self.assertEqual(metadata["stage_status"], "failed")
            self.assertEqual(metadata["reason_code"], "colmap_command_timeout")
            self.assertTrue(metadata["can_retry"])
            self.assertEqual(metadata["error"]["code"], "colmap_command_timeout")
            self.assertEqual(metadata["error"]["stage"], "mapper")
            self.assertTrue(metadata["error"]["retryable"])
            self.assertEqual(metadata["execution_report"]["outcome"], "failed")
            self.assertIn("mapper", [item["stage"] for item in metadata["execution_report"]["stages"]])
            self.assertTrue(Path(metadata["artifacts"]["execution_report"]).exists())


    def test_processing_service_preserves_completed_with_fallback_stage(self) -> None:
        class _CompletedWithFallbackEngine:
            name = "colmap"

            def __init__(self, model_path: Path) -> None:
                self.model_path = model_path

            def reconstruct(self, _project_id, _images_dir, _output_dir, output_format, progress_callback=None):
                self.model_path.write_bytes(b"glTFfake")
                return ReconstructionResult(
                    engine_name="colmap",
                    requested_output_format=output_format,
                    model_path=self.model_path,
                    metadata={
                        "engine": "colmap",
                        "current_stage": "completed_with_fallback",
                        "status_message": "Reconstruccion completada con fallback sparse.",
                        "metrics": {"mesh_face_count": 12},
                    },
                )

        class _FakeMesh:
            def __init__(self) -> None:
                self.vertices = [(0.0, 0.0, 0.0)] * 8
                self.faces = [(0, 1, 2)] * 12

        class _FakeScene:
            def __init__(self) -> None:
                self.geometry = {"mesh_0": _FakeMesh()}

        class _FakeTrimeshModule:
            @staticmethod
            def load(_path, file_type=None, force=None):
                if file_type == "glb" and force == "scene":
                    return _FakeScene()
                raise ValueError(file_type)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            self._write_valid_image(images_dir / "sample.jpg")
            output_dir = root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / "fallback_model.glb"

            project_service, storage_service = self._build_services(images_dir, output_dir)

            with patch(
                "app.services.processing_service.build_reconstruction_engines",
                return_value=(_CompletedWithFallbackEngine(model_path), None),
            ):
                service = ProcessingService(project_service, storage_service, self._ExplicitColmapSettings())

            with patch("app.services.processing_service.importlib.import_module", return_value=_FakeTrimeshModule()):
                service._run_reconstruction_job("demo-project", OutputFormat.GLB)

            metadata = project_service.mark_completed.call_args.kwargs["processing_metadata"]
            self.assertEqual(metadata["current_stage"], "completed_with_fallback")
            self.assertEqual(metadata["status_message"], "Reconstruccion completada con fallback sparse.")


class ColmapEngineTests(unittest.TestCase):
    @staticmethod
    def _build_fake_trimesh(convex_hull_fails: bool = False, delaunay_mesh_fails: bool = False):
        class _FakeMesh:
            def __init__(self, vertex_count: int = 8, face_count: int = 12) -> None:
                self.vertices = [(0.0, 0.0, 0.0)] * vertex_count
                self.faces = [(0, 1, 2)] * face_count

            def export(self, file_type=None):
                if file_type == "glb":
                    return b"glTFfake-mesh"
                if file_type == "obj":
                    return "# OBJ generado por fake trimesh\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
                raise ValueError(file_type)

        class _FakePointCloud:
            def __init__(self) -> None:
                self.vertices = [
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ]

            @property
            def convex_hull(self):
                if convex_hull_fails:
                    raise RuntimeError("scipy/qhull missing")
                return _FakeMesh(vertex_count=6, face_count=8)

            @property
            def bounding_box(self):
                return _FakeMesh(vertex_count=8, face_count=12)

        class _FakeScene:
            def __init__(self, vertex_count: int = 8, face_count: int = 12) -> None:
                self.geometry = {"mesh_0": _FakeMesh(vertex_count=vertex_count, face_count=face_count)}

        class _FakeTrimeshModule:
            @staticmethod
            def load(path, file_type=None, force=None):
                suffix = Path(path).suffix.lower()
                if suffix == ".ply":
                    if "meshed-delaunay" in Path(path).name:
                        if delaunay_mesh_fails:
                            raise ValueError("delaunay mesh unreadable")
                        return _FakeMesh(vertex_count=20, face_count=32)
                    return _FakePointCloud()
                if suffix == ".glb":
                    if convex_hull_fails:
                        return _FakeScene(vertex_count=8, face_count=12)
                    return _FakeScene(vertex_count=6, face_count=8)
                raise ValueError(f"Unexpected asset requested from fake trimesh: {path}")

        return _FakeTrimeshModule()

    def test_colmap_engine_detects_windows_candidates_via_help_command(self) -> None:
        engine = ColmapReconstructionEngine(colmap_binary="colmap")

        def fake_probe(command, capture_output, text, encoding, errors, timeout, check):
            if command[0] == "colmap":
                raise FileNotFoundError("not found")
            return subprocess.CompletedProcess(command, 0, stdout="COLMAP 3.9", stderr="")

        with patch("app.services.engines.colmap_engine.subprocess.run", side_effect=fake_probe):
            self.assertTrue(engine.is_available())

        self.assertEqual(engine.detected_binary, "colmap.exe")

    def test_colmap_engine_run_command_timeout_has_structured_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_dir = root / "logs"
            engine = ColmapReconstructionEngine(
                colmap_binary="C:/COLMAP/colmap.exe",
                timeout_seconds=1,
            )

            timeout_error = subprocess.TimeoutExpired(
                cmd=["C:/COLMAP/colmap.exe", "mapper"],
                timeout=1,
                output="stdout parcial",
                stderr="stderr parcial",
            )
            with patch("app.services.engines.colmap_engine.subprocess.run", side_effect=timeout_error):
                with self.assertRaises(ProcessingError) as context:
                    engine._run_command(
                        project_id="demo-timeout",
                        name="mapper",
                        command=["C:/COLMAP/colmap.exe", "mapper"],
                        logs_dir=logs_dir,
                        progress_callback=None,
                        progress_value=0.5,
                        stage_message="Ejecutando mapper.",
                    )

            error = context.exception
            self.assertEqual(error.reason_code, "colmap_command_timeout")
            self.assertEqual(error.current_stage, "mapper")
            self.assertTrue(error.retryable)
            self.assertEqual(error.metadata["failed_command"], "mapper")
            self.assertEqual(error.metadata["timeout_seconds"], engine.timeout_seconds)
            self.assertTrue(Path(error.metadata["logs"]["stdout_path"]).exists())
            self.assertTrue(Path(error.metadata["logs"]["stderr_path"]).exists())

    def test_colmap_engine_uses_sparse_fallback_glb_when_cuda_is_unavailable(self) -> None:
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
                if command_name == "-h":
                    return subprocess.CompletedProcess(command, 0, stdout="COLMAP 4.0 without CUDA", stderr="")
                if command[-1] == "-h":
                    if command_name == "feature_extractor":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureExtraction.use_gpu", stderr="")
                    if command_name == "exhaustive_matcher":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureMatching.use_gpu", stderr="")
                    if command_name == "patch_match_stereo":
                        return subprocess.CompletedProcess(command, 0, stdout="COLMAP without CUDA", stderr="")
                if command_name in {"image_undistorter", "patch_match_stereo", "stereo_fusion", "poisson_mesher"}:
                    raise AssertionError(f"Dense stage should not run in sparse fallback mode: {command_name}")
                if command_name == "feature_extractor":
                    database_path = Path(command[command.index("--database_path") + 1])
                    database_path.parent.mkdir(parents=True, exist_ok=True)
                    database_path.write_bytes(b"db")
                elif command_name == "delaunay_mesher":
                    output_path = Path(command[command.index("--output_path") + 1])
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text("ply\n", encoding="utf-8")
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
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_text("ply\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with patch.object(engine, "detect_binary", return_value="C:/COLMAP/colmap.exe"):
                with patch("app.services.engines.colmap_engine.subprocess.run", side_effect=fake_run):
                    with patch(
                        "app.services.engines.colmap_engine.importlib.import_module",
                        return_value=self._build_fake_trimesh(),
                    ):
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
            self.assertEqual(result.metadata["reconstruction_type"], "sparse_photogrammetry_mesh_fallback")
            self.assertEqual(result.metadata["current_stage"], "completed_with_fallback")
            self.assertEqual(result.metadata["sparse_fallback"]["used"], True)
            self.assertEqual(result.metadata["sparse_fallback"]["mesh_method"], "delaunay_mesher_sparse")
            self.assertIn("shape_diagnostics", result.metadata["sparse_fallback"])
            self.assertEqual(result.metadata["mesh_face_count"], 8)
            self.assertIn("CUDA", result.metadata["warnings"][0])
            self.assertTrue((output_dir / "demo_model.obj").exists())
            self.assertTrue((output_dir / "workspace" / "sparse" / "0" / "points3D.ply").exists())
            self.assertEqual(
                [item["name"] for item in result.metadata["commands"]],
                [
                    "feature_extractor",
                    "exhaustive_matcher",
                    "mapper",
                    "model_converter_txt",
                    "model_converter_ply",
                    "model_converter_sparse_ply",
                    "delaunay_mesher_sparse",
                ],
            )
            self.assertIn("sparse_mesh_fallback", [update.get("current_stage") for update in progress_updates])
            self.assertEqual(progress_updates[-1].get("current_stage"), "completed_with_fallback")

    def test_colmap_engine_uses_bounding_box_when_convex_hull_fails(self) -> None:
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
                if command_name == "-h":
                    return subprocess.CompletedProcess(command, 0, stdout="COLMAP 4.0 without CUDA", stderr="")
                if command[-1] == "-h":
                    if command_name == "feature_extractor":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureExtraction.use_gpu", stderr="")
                    if command_name == "exhaustive_matcher":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureMatching.use_gpu", stderr="")
                    if command_name == "patch_match_stereo":
                        return subprocess.CompletedProcess(command, 0, stdout="COLMAP without CUDA", stderr="")
                if command_name in {"image_undistorter", "patch_match_stereo", "stereo_fusion", "poisson_mesher"}:
                    raise AssertionError(f"Dense stage should not run in sparse fallback mode: {command_name}")
                if command_name == "feature_extractor":
                    database_path = Path(command[command.index("--database_path") + 1])
                    database_path.parent.mkdir(parents=True, exist_ok=True)
                    database_path.write_bytes(b"db")
                elif command_name == "delaunay_mesher":
                    return subprocess.CompletedProcess(command, 1, stdout="", stderr="delaunay failed")
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
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_text("ply\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with patch.object(engine, "detect_binary", return_value="C:/COLMAP/colmap.exe"):
                with patch("app.services.engines.colmap_engine.subprocess.run", side_effect=fake_run):
                    with patch(
                        "app.services.engines.colmap_engine.importlib.import_module",
                        return_value=self._build_fake_trimesh(convex_hull_fails=True, delaunay_mesh_fails=True),
                    ):
                        result = engine.reconstruct("demo", images_dir, output_dir, OutputFormat.GLB)

            self.assertEqual(result.metadata["current_stage"], "completed_with_fallback")
            self.assertEqual(result.metadata["sparse_fallback"]["mesh_method"], "bounding_box")
            self.assertEqual(result.metadata["mesh_face_count"], 12)
            self.assertTrue(result.model_path.exists())

    def test_colmap_engine_fails_fast_when_dense_is_required_without_cuda(self) -> None:
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
                require_dense_reconstruction=True,
            )

            def fake_run(command, capture_output, text, encoding, errors, timeout, check):
                command_name = command[1]
                if command_name == "-h":
                    return subprocess.CompletedProcess(command, 0, stdout="COLMAP 4.0 without CUDA", stderr="")
                if command[-1] == "-h":
                    if command_name == "feature_extractor":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureExtraction.use_gpu", stderr="")
                    if command_name == "exhaustive_matcher":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureMatching.use_gpu", stderr="")
                    if command_name == "patch_match_stereo":
                        return subprocess.CompletedProcess(command, 0, stdout="COLMAP without CUDA", stderr="")
                raise AssertionError(f"No reconstruction stage should run when dense is required: {command_name}")

            with patch.object(engine, "detect_binary", return_value="C:/COLMAP/colmap.exe"):
                with patch("app.services.engines.colmap_engine.subprocess.run", side_effect=fake_run):
                    with patch(
                        "app.services.engines.colmap_engine.importlib.import_module",
                        return_value=self._build_fake_trimesh(),
                    ):
                        with self.assertRaises(ProcessingError) as context:
                            engine.reconstruct("demo", images_dir, output_dir, OutputFormat.GLB)

            error = context.exception
            self.assertIn("exige reconstruccion densa", str(error).lower())
            self.assertEqual(error.reason_code, "dense_reconstruction_unavailable")
            self.assertEqual(error.current_stage, "dense_reconstruction_unavailable")
            self.assertFalse(error.allow_fallback)

    def test_colmap_engine_skips_dense_stages_when_disabled_by_config(self) -> None:
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
                enable_dense_stages=False,
            )

            def fake_run(command, capture_output, text, encoding, errors, timeout, check):
                command_name = command[1]
                if command_name == "-h":
                    return subprocess.CompletedProcess(command, 0, stdout="COLMAP 4.0 without CUDA", stderr="")
                if command[-1] == "-h":
                    if command_name == "feature_extractor":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureExtraction.use_gpu", stderr="")
                    if command_name == "exhaustive_matcher":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureMatching.use_gpu", stderr="")
                    if command_name == "patch_match_stereo":
                        raise AssertionError("No debe consultar patch_match_stereo -h con etapas densas deshabilitadas")
                if command_name in {"image_undistorter", "patch_match_stereo", "stereo_fusion", "poisson_mesher"}:
                    raise AssertionError(f"Dense stage should not run when disabled by config: {command_name}")
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
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_text("ply\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with patch.object(engine, "detect_binary", return_value="C:/COLMAP/colmap.exe"):
                with patch("app.services.engines.colmap_engine.subprocess.run", side_effect=fake_run):
                    with patch(
                        "app.services.engines.colmap_engine.importlib.import_module",
                        return_value=self._build_fake_trimesh(),
                    ):
                        result = engine.reconstruct("demo", images_dir, output_dir, OutputFormat.GLB)

            self.assertTrue(result.model_path.exists())
            self.assertEqual(result.metadata["current_stage"], "completed_with_fallback")
            self.assertFalse(result.metadata["dense_stages_enabled"])
            self.assertFalse(result.metadata["dense_reconstruction_supported"])
            self.assertIn("deshabilitadas", " ".join(result.metadata.get("warnings", [])).lower())

    def test_colmap_engine_reports_friendly_mapper_error_when_registered_images_are_insufficient(self) -> None:
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
                if command_name == "-h":
                    return subprocess.CompletedProcess(command, 0, stdout="COLMAP 4.0 without CUDA", stderr="")
                if command[-1] == "-h":
                    if command_name == "feature_extractor":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureExtraction.use_gpu", stderr="")
                    if command_name == "exhaustive_matcher":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureMatching.use_gpu", stderr="")
                    if command_name == "patch_match_stereo":
                        return subprocess.CompletedProcess(command, 0, stdout="COLMAP without CUDA", stderr="")
                if command_name == "feature_extractor":
                    database_path = Path(command[command.index("--database_path") + 1])
                    database_path.parent.mkdir(parents=True, exist_ok=True)
                    database_path.write_bytes(b"db")
                    return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
                if command_name == "mapper":
                    return subprocess.CompletedProcess(
                        command,
                        1,
                        stdout="",
                        stderr=(
                            "E20260406 incremental_mapper.cc:1080] Check failed: ba_config.NumImages() >= 2 "
                            "(0 vs. 2) At least two images must be registered for global bundle-adjustment"
                        ),
                    )
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with patch.object(engine, "detect_binary", return_value="C:/COLMAP/colmap.exe"):
                with patch("app.services.engines.colmap_engine.subprocess.run", side_effect=fake_run):
                    with patch(
                        "app.services.engines.colmap_engine.importlib.import_module",
                        return_value=self._build_fake_trimesh(),
                    ):
                        with self.assertRaises(ProcessingError) as context:
                            engine.reconstruct("demo", images_dir, output_dir, OutputFormat.OBJ)

            error = context.exception
            self.assertIn("no logro registrar suficientes imagenes", str(error).lower())
            self.assertEqual(error.reason_code, "insufficient_registered_images")
            self.assertEqual(error.current_stage, "mapper_failed_insufficient_registered_images")
            self.assertEqual(error.metadata["registered_image_count"], 0)
            self.assertFalse(error.allow_fallback)

    def test_colmap_engine_fails_when_sparse_model_has_fewer_than_two_registered_images(self) -> None:
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
                if command_name == "-h":
                    return subprocess.CompletedProcess(command, 0, stdout="COLMAP 4.0 without CUDA", stderr="")
                if command[-1] == "-h":
                    if command_name == "feature_extractor":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureExtraction.use_gpu", stderr="")
                    if command_name == "exhaustive_matcher":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureMatching.use_gpu", stderr="")
                    if command_name == "patch_match_stereo":
                        return subprocess.CompletedProcess(command, 0, stdout="COLMAP without CUDA", stderr="")
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
                            "# images\n1 1 0 0 0 0 0 0 1 img_1.jpg\n0 0 -1\n",
                            encoding="utf-8",
                        )
                        (output_path / "points3D.txt").write_text(
                            "# points\n1 0.0 0.0 0.0 255 0 0 0.1 1 1\n",
                            encoding="utf-8",
                        )
                    elif output_type == "PLY":
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_text("ply\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with patch.object(engine, "detect_binary", return_value="C:/COLMAP/colmap.exe"):
                with patch("app.services.engines.colmap_engine.subprocess.run", side_effect=fake_run):
                    with patch(
                        "app.services.engines.colmap_engine.importlib.import_module",
                        return_value=self._build_fake_trimesh(),
                    ):
                        with self.assertRaises(ProcessingError) as context:
                            engine.reconstruct("demo", images_dir, output_dir, OutputFormat.OBJ)

            error = context.exception
            self.assertIn("no logro registrar suficientes imagenes", str(error).lower())
            self.assertEqual(error.reason_code, "insufficient_registered_images")
            self.assertEqual(error.current_stage, "mapper_failed_insufficient_registered_images")
            self.assertEqual(error.metadata["registered_image_count"], 1)
            self.assertFalse(error.allow_fallback)

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
                if command_name == "-h":
                    return subprocess.CompletedProcess(command, 0, stdout="COLMAP 4.0 without CUDA", stderr="")
                if command[-1] == "-h":
                    if command_name == "feature_extractor":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureExtraction.use_gpu", stderr="")
                    if command_name == "exhaustive_matcher":
                        return subprocess.CompletedProcess(command, 0, stdout="--FeatureMatching.use_gpu", stderr="")
                    if command_name == "patch_match_stereo":
                        return subprocess.CompletedProcess(command, 0, stdout="COLMAP without CUDA", stderr="")
                if command_name == "feature_extractor":
                    database_path = Path(command[command.index("--database_path") + 1])
                    database_path.parent.mkdir(parents=True, exist_ok=True)
                    database_path.write_bytes(b"db")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with patch.object(engine, "detect_binary", return_value="C:/COLMAP/colmap.exe"):
                with patch("app.services.engines.colmap_engine.subprocess.run", side_effect=fake_run):
                    with patch(
                        "app.services.engines.colmap_engine.importlib.import_module",
                        return_value=self._build_fake_trimesh(),
                    ):
                        with self.assertRaises(ProcessingError) as context:
                            engine.reconstruct("demo", images_dir, output_dir, OutputFormat.OBJ)

            self.assertIn("no genero ningun submodelo", str(context.exception).lower())

    def test_validate_mapper_output_selects_best_sparse_submodel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sparse_dir = root / "sparse"
            sparse_zero = sparse_dir / "0"
            sparse_one = sparse_dir / "1"
            sparse_zero.mkdir(parents=True, exist_ok=True)
            sparse_one.mkdir(parents=True, exist_ok=True)

            for model_dir in (sparse_zero, sparse_one):
                (model_dir / "cameras.bin").write_bytes(b"c")
                (model_dir / "images.bin").write_bytes(b"i")
                (model_dir / "points3D.bin").write_bytes(b"p")

            engine = ColmapReconstructionEngine(colmap_binary="C:/COLMAP/colmap.exe", timeout_seconds=30)

            def fake_analyze(model_dir: Path, _resolved_binary: str) -> tuple[int, int]:
                if model_dir.name == "0":
                    return 2, 227
                return 7, 538

            with patch.object(engine, "_analyze_sparse_model", side_effect=fake_analyze):
                selected = engine._validate_mapper_output(
                    sparse_dir=sparse_dir,
                    resolved_binary="C:/COLMAP/colmap.exe",
                    project_id="demo",
                )

            self.assertEqual(selected.name, "1")


if __name__ == "__main__":
    unittest.main()
