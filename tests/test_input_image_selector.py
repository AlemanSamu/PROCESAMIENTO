from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from app.algorithms.input_image_selector import InputImageSelectionSettings, InputImageSelector
from app.models.schemas import OutputFormat
from app.services.engines.base_engine import ReconstructionResult
from app.services.processing_service import ProcessingService


class InputImageSelectorTests(unittest.TestCase):
    @staticmethod
    def _draw_pattern(path: Path, *, offset: int = 0, base_tone: int = 120) -> None:
        canvas = Image.new("RGB", (1024, 768), (base_tone, base_tone, base_tone))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((120 + offset, 110, 900 + offset, 650), outline="black", fill=(170, 170, 170), width=7)
        draw.ellipse((220, 210 + offset, 640, 610 + offset), outline="blue", fill=(90, 90, 160), width=6)
        draw.line((160, 150 + offset, 890, 620 - offset), fill="red", width=7)
        canvas.save(path)

    @staticmethod
    def _build_validation_summary(candidate_paths: list[Path]) -> dict[str, object]:
        images_payload = []
        for path in candidate_paths:
            images_payload.append(
                {
                    "path": str(path),
                    "status": "apta",
                    "warning_reasons": [],
                    "metrics": {
                        "width": 1024,
                        "height": 768,
                        "pixel_count": 1024 * 768,
                        "brightness": 0.5,
                        "sharpness": 0.08,
                    },
                }
            )
        return {"images": images_payload}

    def test_selector_discards_near_duplicates_and_generates_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            report_dir = root / "report"
            images_dir.mkdir(parents=True, exist_ok=True)

            img_a = images_dir / "a.jpg"
            img_a_copy = images_dir / "a_copy.jpg"
            img_b = images_dir / "b.jpg"
            img_c = images_dir / "c.jpg"

            self._draw_pattern(img_a, offset=0)
            img_a_copy.write_bytes(img_a.read_bytes())
            self._draw_pattern(img_b, offset=36)
            self._draw_pattern(img_c, offset=78)

            candidates = [img_a, img_a_copy, img_b, img_c]
            validation_summary = self._build_validation_summary(candidates)

            selector = InputImageSelector(
                selection_settings=InputImageSelectionSettings(
                    enabled=True,
                    min_images_required=2,
                    max_images=3,
                    target_keep_ratio=0.6,
                    min_quality_score=0.0,
                    near_duplicate_hamming=3,
                    diversity_min_hamming=8,
                )
            )
            result = selector.select_images(validation_summary, candidates, report_dir=report_dir)

            self.assertTrue(result.allow_processing)
            self.assertTrue((report_dir / "input_image_selection_report.json").exists())
            self.assertEqual(result.summary["candidate_images"], 4)
            self.assertLess(result.summary["selected_images"], 4)
            self.assertGreaterEqual(result.summary["selected_images"], 2)
            self.assertGreaterEqual(
                result.summary["discarded_reason_counts"].get("near_duplicate_of:a.jpg", 0)
                + result.summary["discarded_reason_counts"].get("near_duplicate_of:a_copy.jpg", 0),
                1,
            )


class ProcessingSelectionIntegrationTests(unittest.TestCase):
    class _Settings:
        processing_engine = "colmap"
        simulation_delay_seconds = 0
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
        image_selection_enabled = True
        image_selection_min_images_required = 1
        image_selection_max_images = 1
        image_selection_target_keep_ratio = 0.34
        image_selection_min_quality_score = 0.0
        image_selection_quality_weight = 0.65
        image_selection_diversity_weight = 0.35
        image_selection_near_duplicate_hamming = 3
        image_selection_diversity_min_hamming = 8
        colmap_path = None
        colmap_binary = "colmap"
        colmap_timeout_seconds = 120
        colmap_use_gpu = False
        colmap_enable_dense_stages = True
        colmap_camera_model = "SIMPLE_RADIAL"
        colmap_single_camera = True
        colmap_fallback_to_mock = False
        colmap_require_dense_reconstruction = False

    class _CapturingEngine:
        name = "colmap"

        def __init__(self) -> None:
            self.received_image_count = 0

        def reconstruct(self, _project_id, images_dir, output_dir, output_format, progress_callback=None):
            self.received_image_count = len([path for path in images_dir.iterdir() if path.is_file()])
            model_path = output_dir / "demo_model.obj"
            model_path.write_text("o selected", encoding="utf-8")
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
        canvas = Image.new("RGB", (1024, 768), (120, 120, 120))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((120 + offset, 100, 920 + offset, 680), outline="black", fill=(170, 170, 170), width=7)
        draw.ellipse((230, 220 + offset, 670, 650 + offset), outline="blue", fill=(90, 90, 160), width=6)
        draw.line((180, 160 + offset, 900, 640 - offset), fill="red", width=7)
        canvas.save(path)

    def test_processing_service_uses_selected_subset_before_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            output_dir = root / "output"
            images_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            self._write_valid_image(images_dir / "img_1.jpg", offset=0)
            self._write_valid_image(images_dir / "img_2.jpg", offset=24)
            self._write_valid_image(images_dir / "img_3.jpg", offset=50)

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

            self.assertEqual(engine.received_image_count, 1)
            project_service.mark_completed.assert_called_once()
            metadata = project_service.mark_completed.call_args.kwargs["processing_metadata"]
            self.assertEqual(metadata.get("current_stage"), "completed")
            self.assertEqual((metadata.get("input_selection") or {}).get("selected_images"), 1)
            self.assertEqual((metadata.get("metrics") or {}).get("image_count_received"), 3)
            self.assertEqual((metadata.get("metrics") or {}).get("image_count_selected"), 1)
            self.assertEqual((metadata.get("metrics") or {}).get("image_count_processed"), 1)


if __name__ == "__main__":
    unittest.main()
