from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.services.engines.colmap_engine import ColmapReconstructionEngine
from app.services.processing_service import ProcessingService


class ColmapPhase3Tests(unittest.TestCase):
    def test_colmap_setup_script_exists(self) -> None:
        script_path = Path("scripts/check_colmap_setup.py")

        self.assertTrue(script_path.exists())
        self.assertIn("LOCAL3D_COLMAP_BINARY", script_path.read_text(encoding="utf-8"))

    def test_colmap_report_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            report_path = ColmapReconstructionEngine.write_failure_report(
                project_id="schema-demo",
                output_dir=output_dir,
                colmap_binary="colmap",
                profile="balanced",
                failure_reason="COLMAP no disponible",
                error_context={"reason_code": "colmap_unavailable"},
                fallback_used=True,
            )

            payload = json.loads(report_path.read_text(encoding="utf-8"))
            expected_keys = {
                "colmap_binary",
                "colmap_version",
                "gpu_detected",
                "gpu_requested",
                "gpu_used",
                "gpu_fallback_to_cpu",
                "gpu_error_message",
                "profile",
                "commands_executed",
                "command_durations",
                "stage_timings_by_phase",
                "sparse_created",
                "cameras_reconstructed",
                "images_registered",
                "points3D_count",
                "model_outputs",
                "warnings",
                "failure_reason",
                "fallback_used",
            }
            self.assertTrue(expected_keys.issubset(payload.keys()))
            self.assertTrue(payload["fallback_used"])
            self.assertFalse(payload["sparse_created"])

    def test_colmap_profile_options(self) -> None:
        conservative = ColmapReconstructionEngine.profile_options("conservative", gpu_available=True)
        balanced = ColmapReconstructionEngine.profile_options("balanced", gpu_available=True)
        quality = ColmapReconstructionEngine.profile_options("quality", gpu_available=True)
        balanced_cpu = ColmapReconstructionEngine.profile_options("balanced", gpu_available=False)

        self.assertEqual(conservative["SiftExtraction.use_gpu"], 0)
        self.assertEqual(conservative["SiftMatching.use_gpu"], 0)
        self.assertFalse(conservative["dense_enabled"])
        self.assertEqual(balanced["SiftExtraction.use_gpu"], 1)
        self.assertEqual(balanced_cpu["SiftExtraction.use_gpu"], 0)
        self.assertGreater(quality["timeout_seconds"], balanced["timeout_seconds"])

    def test_colmap_failure_generates_report_and_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            images_dir = root / "preprocessed_images"
            output_dir.mkdir()
            images_dir.mkdir()
            for index in range(3):
                (images_dir / f"img_{index + 1}.jpg").write_bytes(b"fake-image")
            model_path = output_dir / "fallback.glb"
            model_path.write_bytes(b"glb")

            service = object.__new__(ProcessingService)
            service._engine = SimpleNamespace(
                name="colmap",
                colmap_binary="missing-colmap",
                detected_binary=None,
                is_available=lambda: False,
            )
            service._requested_engine_mode = "colmap"
            service.settings = SimpleNamespace(profile="balanced")

            metadata = service._write_academic_fallback_report(
                project_id="fallback-demo",
                output_dir=output_dir,
                metadata={"fallback_used": True, "artifacts": {}},
                failure_reason="COLMAP fallo durante mapper",
                error_context={"reason_code": "colmap_command_failed", "current_stage": "mapper"},
                selected_images_dir=images_dir,
                model_path=model_path,
            )

            fallback_report = output_dir / "pipeline" / "fallback_report.json"
            colmap_report = output_dir / "pipeline" / "colmap_report.json"
            self.assertTrue(fallback_report.exists())
            self.assertTrue(colmap_report.exists())
            self.assertIn("primitive_box_academic_fallback", fallback_report.read_text(encoding="utf-8"))
            payload = json.loads(colmap_report.read_text(encoding="utf-8"))
            self.assertTrue(payload["fallback_used"])
            self.assertEqual(payload["failure_reason"], "COLMAP fallo durante mapper")
            self.assertEqual(metadata["artifacts"]["colmap_report"], str(colmap_report))


if __name__ == "__main__":
    unittest.main()
