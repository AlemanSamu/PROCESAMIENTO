from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from app.algorithms.input_object_segmenter import (
    InputObjectSegmentationSettings,
    InputObjectSegmenter,
)
from app.models.schemas import OutputFormat
from app.services.engines.base_engine import ReconstructionResult
from app.services.processing_service import ProcessingService


class InputObjectSegmenterTests(unittest.TestCase):
    @staticmethod
    def _draw_box_scene(path: Path, *, offset: int = 0) -> None:
        canvas = Image.new("RGB", (1024, 768), (210, 212, 214))
        draw = ImageDraw.Draw(canvas)
        left = 250 + offset
        top = 220
        right = 760 + offset
        bottom = 520
        draw.rectangle((left, top, right, bottom), fill=(42, 92, 180), outline=(18, 40, 90), width=7)
        draw.rectangle((left + 16, top + 70, right - 16, bottom - 20), fill=(236, 238, 244), outline=(32, 62, 120), width=5)
        draw.text((left + 45, top + 105), "Azitromicina", fill=(24, 38, 96))
        canvas.save(path)

    def test_segmenter_detects_main_object_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            report_dir = root / "report"
            images_dir.mkdir(parents=True, exist_ok=True)

            image_a = images_dir / "img_a.jpg"
            image_b = images_dir / "img_b.jpg"
            self._draw_box_scene(image_a, offset=0)
            self._draw_box_scene(image_b, offset=22)

            segmenter = InputObjectSegmenter(
                settings=InputObjectSegmentationSettings(
                    enabled=True,
                    min_component_score=0.10,
                    block_on_low_success=False,
                )
            )
            result = segmenter.segment_images([image_a, image_b], report_dir=report_dir)

            self.assertTrue(result.allow_processing)
            self.assertEqual(len(result.processed_images), 2)
            self.assertTrue((report_dir / "input_object_segmentation_report.json").exists())
            self.assertTrue((report_dir / "segmented_images" / "img_a.jpg").exists())
            self.assertTrue((report_dir / "segmentation_masks" / "img_a_mask.png").exists())
            self.assertGreaterEqual(result.summary.get("segmented_images", 0), 1)
            self.assertIn("images", result.summary)

    def test_segmenter_falls_back_to_original_when_no_component_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            report_dir = root / "report"
            images_dir.mkdir(parents=True, exist_ok=True)

            flat = images_dir / "flat.jpg"
            Image.new("RGB", (960, 640), (185, 185, 185)).save(flat)

            segmenter = InputObjectSegmenter(
                settings=InputObjectSegmentationSettings(
                    enabled=True,
                    min_component_score=0.2,
                    block_on_low_success=False,
                )
            )
            result = segmenter.segment_images([flat], report_dir=report_dir)

            self.assertTrue(result.allow_processing)
            self.assertEqual(result.summary.get("segmented_images"), 0)
            self.assertEqual(result.summary.get("fallback_original_images"), 1)
            self.assertTrue((report_dir / "segmented_images" / "flat.jpg").exists())
            self.assertTrue((report_dir / "segmentation_masks" / "flat_mask.png").exists())

    def test_segmenter_blocks_when_success_ratio_is_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            report_dir = root / "report"
            images_dir.mkdir(parents=True, exist_ok=True)

            for index in range(3):
                Image.new("RGB", (960, 640), (185, 185, 185)).save(images_dir / f"flat_{index}.jpg")

            segmenter = InputObjectSegmenter(
                settings=InputObjectSegmentationSettings(
                    enabled=True,
                    min_component_score=0.3,
                    min_segmented_images=2,
                    min_segmented_ratio=0.8,
                    block_on_low_success=True,
                )
            )
            ordered_images = [images_dir / f"flat_{index}.jpg" for index in range(3)]
            result = segmenter.segment_images(ordered_images, report_dir=report_dir)

            self.assertFalse(result.allow_processing)
            self.assertIn("insufficient_segmented_images", result.summary.get("blocking_reasons", []))
            self.assertIn("segmented_ratio_below_threshold", result.summary.get("blocking_reasons", []))


class ProcessingSegmentationIntegrationTests(unittest.TestCase):
    class _Settings:
        processing_engine = "colmap"
        simulation_delay_seconds = 0
        processing_cleanup_workspace_on_failure = True
        processing_execution_timeline_limit = 200
        allowed_image_extensions = (".jpg", ".jpeg", ".png")
        image_validation_enabled = True
        image_validation_min_images_required = 1
        image_validation_min_width = 640
        image_validation_min_height = 480
        image_validation_min_pixels = 640 * 480
        image_validation_min_sharpness_warn = 0.01
        image_validation_min_sharpness_reject = 0.001
        image_validation_min_brightness = 0.05
        image_validation_max_brightness = 0.95
        image_validation_exposure_warn_margin = 0.07
        image_validation_near_duplicate_warn_hamming = 6
        image_validation_near_duplicate_reject_hamming = 2
        image_validation_coverage_min_unique_ratio = 0.4
        image_validation_coverage_min_median_hamming = 4
        image_validation_coverage_max_neighbor_similarity_ratio = 0.95
        image_validation_block_on_low_coverage = False
        image_selection_enabled = True
        image_selection_min_images_required = 1
        image_selection_max_images = 3
        image_selection_target_keep_ratio = 1.0
        image_selection_min_quality_score = 0.0
        image_selection_quality_weight = 0.65
        image_selection_diversity_weight = 0.35
        image_selection_near_duplicate_hamming = 3
        image_selection_diversity_min_hamming = 6
        image_object_segmentation_enabled = True
        image_object_segmentation_analysis_max_width = 384
        image_object_segmentation_min_component_area_ratio = 0.005
        image_object_segmentation_max_component_area_ratio = 0.72
        image_object_segmentation_min_component_fill_ratio = 0.14
        image_object_segmentation_expected_aspect_ratio = 1.6
        image_object_segmentation_aspect_tolerance = 1.4
        image_object_segmentation_mask_padding_ratio = 0.06
        image_object_segmentation_min_component_score = 0.12
        image_object_segmentation_min_segmented_images = 1
        image_object_segmentation_min_segmented_ratio = 0.4
        image_object_segmentation_block_on_low_success = False
        primitive_box_fallback_enabled = False
        primitive_box_fallback_min_selected_images = 3
        primitive_box_fallback_analysis_max_width = 256
        primitive_box_fallback_min_foreground_ratio = 0.03
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
        force_presentable_model_enabled = False
        force_presentable_model_glb = None
        force_presentable_model_obj = None
        metrics_evidence_enabled = False
        colmap_path = None
        colmap_binary = "colmap"
        colmap_timeout_seconds = 120
        colmap_use_gpu = False
        colmap_gpu_mode = "disabled"
        colmap_gpu_probe_timeout_seconds = 2
        colmap_enable_dense_stages = False
        colmap_camera_model = "SIMPLE_RADIAL"
        colmap_single_camera = True
        colmap_fallback_to_mock = False
        colmap_require_dense_reconstruction = False

    class _BlockingSettings(_Settings):
        image_object_segmentation_max_component_area_ratio = 0.02
        image_object_segmentation_min_segmented_images = 2
        image_object_segmentation_min_segmented_ratio = 0.8
        image_object_segmentation_block_on_low_success = True

    class _CapturingEngine:
        name = "colmap"

        def __init__(self) -> None:
            self.received_image_count = 0
            self.received_images_dir: Path | None = None

        def reconstruct(self, _project_id, images_dir, output_dir, output_format, progress_callback=None):
            self.received_images_dir = images_dir
            self.received_image_count = len([path for path in images_dir.iterdir() if path.is_file()])
            model_path = output_dir / "demo_model.obj"
            model_path.write_text("o segmented", encoding="utf-8")
            return ReconstructionResult(
                engine_name=self.name,
                requested_output_format=output_format,
                model_path=model_path,
                metadata={
                    "engine": self.name,
                    "current_stage": "completed",
                    "status_message": "completed",
                },
            )

    @staticmethod
    def _write_valid_image(path: Path, offset: int = 0) -> None:
        canvas = Image.new("RGB", (1024, 768), (200, 201, 203))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((250 + offset, 220, 760 + offset, 520), fill=(42, 92, 180), outline=(18, 40, 90), width=7)
        draw.rectangle((265 + offset, 292, 744 + offset, 500), fill=(236, 238, 244), outline=(32, 62, 120), width=5)
        draw.text((300 + offset, 330), "Azitromicina", fill=(24, 38, 96))
        canvas.save(path)

    def test_processing_service_uses_segmented_images_before_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            output_dir = root / "output"
            images_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            self._write_valid_image(images_dir / "img_1.jpg", offset=0)
            self._write_valid_image(images_dir / "img_2.jpg", offset=24)
            self._write_valid_image(images_dir / "img_3.jpg", offset=48)

            project_service = MagicMock()
            project_service.get_project.return_value.processing_metadata = {}
            storage_service = MagicMock()
            storage_service.get_images_dir.return_value = images_dir
            storage_service.get_output_dir.return_value = output_dir

            engine = self._CapturingEngine()
            with patch(
                "app.services.processing_service.build_reconstruction_engines",
                return_value=(engine, None),
            ):
                service = ProcessingService(project_service, storage_service, self._Settings())

            service._run_reconstruction_job("demo-project", OutputFormat.OBJ)

            self.assertIsNotNone(engine.received_images_dir)
            self.assertEqual(engine.received_images_dir.name, "segmented_images")
            self.assertEqual(engine.received_image_count, 3)

            project_service.mark_completed.assert_called_once()
            metadata = project_service.mark_completed.call_args.kwargs["processing_metadata"]
            self.assertEqual(metadata.get("current_stage"), "completed")
            self.assertIn("input_object_segmentation", metadata)
            self.assertEqual((metadata.get("metrics") or {}).get("image_count_segmented"), 3)
            self.assertEqual((metadata.get("metrics") or {}).get("image_count_processed"), 3)
            self.assertTrue((metadata.get("artifacts") or {}).get("input_object_segmentation_report"))

    def test_processing_service_keeps_running_when_segmentation_confidence_is_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            output_dir = root / "output"
            images_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            self._write_valid_image(images_dir / "img_1.jpg", offset=0)
            self._write_valid_image(images_dir / "img_2.jpg", offset=24)
            self._write_valid_image(images_dir / "img_3.jpg", offset=48)

            project_service = MagicMock()
            project_service.get_project.return_value.processing_metadata = {}
            storage_service = MagicMock()
            storage_service.get_images_dir.return_value = images_dir
            storage_service.get_output_dir.return_value = output_dir

            engine = self._CapturingEngine()
            with patch(
                "app.services.processing_service.build_reconstruction_engines",
                return_value=(engine, None),
            ):
                service = ProcessingService(project_service, storage_service, self._BlockingSettings())

            service._run_reconstruction_job("demo-project", OutputFormat.OBJ)

            project_service.mark_completed.assert_called_once()
            project_service.mark_failed.assert_not_called()
            metadata = project_service.mark_completed.call_args.kwargs["processing_metadata"]
            self.assertEqual(metadata.get("current_stage"), "completed")
            segmentation = metadata.get("input_object_segmentation") or {}
            policy = segmentation.get("policy_decision") or {}
            self.assertFalse(bool(policy.get("processing_blocked")))
            self.assertFalse(bool(policy.get("segmenter_allow_processing")))
            self.assertEqual(engine.received_image_count, 3)


if __name__ == "__main__":
    unittest.main()
