import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.errors import ProcessingError
from app.services.engines import factory as engine_factory
from app.services.engines.colmap_engine import ColmapCommandTrace, ColmapReconstructionEngine


class ColmapGpuSelectionTests(unittest.TestCase):
    @staticmethod
    def _settings(**overrides):
        base = {
            "colmap_gpu_mode": "auto",
            "colmap_use_gpu": False,
            "colmap_gpu_probe_timeout_seconds": 3,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_resolve_gpu_mode_uses_explicit_mode(self) -> None:
        settings = self._settings(colmap_gpu_mode="disabled", colmap_use_gpu=True)
        self.assertEqual(engine_factory._resolve_gpu_mode(settings), "disabled")

    def test_resolve_gpu_mode_falls_back_to_legacy_flag(self) -> None:
        settings = self._settings(colmap_gpu_mode="legacy", colmap_use_gpu=True)
        self.assertEqual(engine_factory._resolve_gpu_mode(settings), "enabled")

    def test_resolve_gpu_request_auto_enables_when_probe_detects_gpu(self) -> None:
        settings = self._settings(colmap_gpu_mode="auto", colmap_use_gpu=False)
        with patch.object(engine_factory, "_probe_nvidia_gpu", return_value=(True, "nvidia_smi_detected_gpu")):
            use_gpu, reason = engine_factory._resolve_gpu_request("auto", settings)

        self.assertTrue(use_gpu)
        self.assertEqual(reason, "nvidia_smi_detected_gpu")

    def test_resolve_gpu_request_auto_keeps_legacy_true_when_probe_fails(self) -> None:
        settings = self._settings(colmap_gpu_mode="auto", colmap_use_gpu=True)
        with patch.object(engine_factory, "_probe_nvidia_gpu", return_value=(False, "nvidia_smi_not_found")):
            use_gpu, reason = engine_factory._resolve_gpu_request("auto", settings)

        self.assertTrue(use_gpu)
        self.assertEqual(reason, "legacy_colmap_use_gpu_true")


class ColmapGpuRuntimeFallbackTests(unittest.TestCase):
    def test_retry_stage_in_cpu_when_gpu_runtime_error_happens(self) -> None:
        engine = ColmapReconstructionEngine(
            use_gpu=True,
            gpu_mode="auto",
            gpu_probe_reason="nvidia_smi_detected_gpu",
        )
        failing_error = ProcessingError(
            "CUDA error: no device",
            reason_code="colmap_command_failed",
            current_stage="feature_extractor",
            metadata={"logs": {"stderr_path": "stderr.log"}},
        )
        successful_trace = ColmapCommandTrace(
            name="feature_extractor",
            command="colmap feature_extractor --SiftExtraction.use_gpu 0",
            duration_seconds=1.2,
            return_code=0,
            stdout_tail="ok",
            stderr_tail="",
            stdout_path="stdout.log",
            stderr_path="stderr.log",
        )

        with patch.object(engine, "_run_command", side_effect=[failing_error, successful_trace]) as mocked_run:
            trace, fell_back_to_cpu, reason = engine._run_command_with_optional_gpu_fallback(
                project_id="demo",
                name="feature_extractor",
                command=[
                    "colmap",
                    "feature_extractor",
                    "--SiftExtraction.use_gpu",
                    "1",
                ],
                logs_dir=Path("logs"),
                progress_callback=None,
                progress_value=0.10,
                stage_message="feature_extractor",
                gpu_flag_name="--SiftExtraction.use_gpu",
                allow_gpu_fallback=True,
            )

        self.assertTrue(fell_back_to_cpu)
        self.assertIsNotNone(reason)
        self.assertEqual(trace.return_code, 0)
        self.assertEqual(mocked_run.call_count, 2)
        retry_command = mocked_run.call_args_list[1].kwargs["command"]
        self.assertEqual(retry_command[3], "0")


if __name__ == "__main__":
    unittest.main()
