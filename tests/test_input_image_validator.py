from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from app.algorithms.input_image_validator import InputImageValidationSettings, InputImageValidator
from app.models.schemas import OutputFormat
from app.services.processing_service import ProcessingService


class InputImageValidatorTests(unittest.TestCase):
    @staticmethod
    def _write_pattern_image(path: Path, *, shift: int = 0, size: tuple[int, int] = (1280, 720), tone: str = "normal") -> None:
        if tone == "dark":
            canvas = Image.new("RGB", size, (12, 12, 12))
            canvas.save(path)
            return
        if tone == "bright":
            canvas = Image.new("RGB", size, (245, 245, 245))
            canvas.save(path)
            return

        canvas = Image.new("RGB", size, (120, 120, 120))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((120 + shift, 80, 1120 + shift, 640), outline="black", fill=(170, 170, 170), width=8)
        draw.ellipse((260, 180 + shift, 760, 640 + shift), outline="blue", fill=(90, 90, 160), width=6)
        draw.line((140, 120 + shift, 1140, 620 - shift), fill="red", width=7)
        canvas.save(path)

    def test_validator_rejects_bad_inputs_and_blocks_when_minimum_not_met(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            report_dir = root / "report"
            images_dir.mkdir(parents=True, exist_ok=True)

            self._write_pattern_image(images_dir / "good_1.jpg")
            self._write_pattern_image(images_dir / "good_2.jpg", shift=24)
            self._write_pattern_image(images_dir / "dark.jpg", tone="dark")
            self._write_pattern_image(images_dir / "bright.jpg", tone="bright")
            self._write_pattern_image(images_dir / "lowres.jpg", size=(320, 240))
            shutil.copy2(images_dir / "good_1.jpg", images_dir / "dup_good_1.jpg")

            validator = InputImageValidator(
                validation_settings=InputImageValidationSettings(
                    min_images_required=3,
                    min_width=640,
                    min_height=480,
                    min_pixels=640 * 480,
                    min_sharpness_warning=0.06,
                    min_sharpness_reject=0.04,
                )
            )
            result = validator.validate_batch(images_dir, report_dir=report_dir)

            self.assertFalse(result.allow_processing)
            self.assertTrue((report_dir / "input_image_validation_report.json").exists())
            self.assertEqual(result.summary["total_images"], 6)
            self.assertEqual(result.summary["accepted_images"], 2)
            self.assertEqual(result.summary["rejected_images"], 4)
            self.assertIn("insufficient_valid_images", result.summary["blocking_reasons"])
            self.assertGreaterEqual(result.summary["rejected_reason_counts"].get("underexposed", 0), 1)
            self.assertGreaterEqual(result.summary["rejected_reason_counts"].get("overexposed", 0), 1)
            self.assertGreaterEqual(result.summary["rejected_reason_counts"].get("low_resolution", 0), 1)
            self.assertGreaterEqual(result.summary["rejected_reason_counts"].get("duplicate_exact", 0), 1)


class ProcessingValidationGateTests(unittest.TestCase):
    class _Settings:
        processing_engine = "colmap"
        simulation_delay_seconds = 0
        image_validation_enabled = True
        image_validation_min_images_required = 2
        image_validation_min_width = 640
        image_validation_min_height = 480
        image_validation_min_pixels = 640 * 480
        colmap_path = None
        colmap_binary = "colmap"
        colmap_timeout_seconds = 120
        colmap_use_gpu = False
        colmap_enable_dense_stages = True
        colmap_camera_model = "SIMPLE_RADIAL"
        colmap_single_camera = True
        colmap_fallback_to_mock = False
        colmap_require_dense_reconstruction = False

    class _RecordingEngine:
        name = "colmap"

        def __init__(self) -> None:
            self.called = False

        def reconstruct(self, *_args, **_kwargs):
            self.called = True
            raise AssertionError("El motor no debe ejecutarse cuando la validacion bloquea el lote")

    @staticmethod
    def _write_valid_image(path: Path) -> None:
        canvas = Image.new("RGB", (800, 600), (120, 120, 120))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((80, 70, 720, 520), outline="black", fill=(170, 170, 170), width=5)
        draw.ellipse((180, 160, 520, 440), outline="blue", fill=(90, 90, 160), width=4)
        draw.line((100, 100, 700, 500), fill="red", width=5)
        canvas.save(path)

    def test_processing_service_stops_when_validation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            output_dir = root / "output"
            images_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._write_valid_image(images_dir / "only_one.jpg")

            project_service = MagicMock()
            project_service.get_project.return_value.processing_metadata = {}
            storage_service = MagicMock()
            storage_service.get_images_dir.return_value = images_dir
            storage_service.get_output_dir.return_value = output_dir

            engine = self._RecordingEngine()
            with patch(
                "app.services.processing_service.build_reconstruction_engines",
                return_value=(engine, None),
            ):
                service = ProcessingService(project_service, storage_service, self._Settings())

            service._run_reconstruction_job("demo-project", OutputFormat.GLB)

            self.assertFalse(engine.called)
            project_service.mark_completed.assert_not_called()
            project_service.mark_failed.assert_called_once()
            failure_metadata = project_service.mark_failed.call_args.kwargs["processing_metadata"]
            self.assertEqual(failure_metadata["current_stage"], "input_validation_failed")
            self.assertEqual(failure_metadata["reason_code"], "input_validation_failed")
            self.assertIn("insufficient_valid_images", (failure_metadata.get("input_validation") or {}).get("blocking_reasons", []))


if __name__ == "__main__":
    unittest.main()
