from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.models.schemas import OutputFormat
from app.services.engines.base_engine import ReconstructionResult
from app.services.processing_service import ProcessingService


class ForcedPresentableModelTests(unittest.TestCase):
    @staticmethod
    def _build_settings(tmp_root: Path, *, enabled: bool, glb_path: Path, obj_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            processing_engine="auto",
            force_presentable_model_enabled=enabled,
            force_presentable_model_glb=str(glb_path),
            force_presentable_model_obj=str(obj_path),
        )

    def _build_service(self, settings: SimpleNamespace) -> ProcessingService:
        engine = SimpleNamespace(name="mock")
        with (
            patch("app.services.processing_service.build_reconstruction_engines", return_value=(engine, None)),
            patch("app.services.processing_service.InputImageValidator.from_settings", return_value=MagicMock()),
            patch("app.services.processing_service.InputImageSelector.from_settings", return_value=MagicMock()),
            patch("app.services.processing_service.BoxPrimitiveFallback.from_settings", return_value=MagicMock()),
            patch("app.services.processing_service.TechnicalEvidenceService", return_value=MagicMock()),
        ):
            return ProcessingService(MagicMock(), MagicMock(), settings)

    def test_apply_forced_presentable_model_replaces_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            canonical_dir = root / "canonical"
            canonical_dir.mkdir(parents=True, exist_ok=True)

            canonical_glb = canonical_dir / "master.glb"
            canonical_obj = canonical_dir / "master.obj"
            canonical_glb.write_bytes(b"glTFdemo")
            canonical_obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")

            original = output_dir / "original_model.glb"
            original.write_bytes(b"old")
            result = ReconstructionResult(
                engine_name="mock",
                requested_output_format=OutputFormat.GLB,
                model_path=original,
                metadata={},
            )
            settings = self._build_settings(
                root,
                enabled=True,
                glb_path=canonical_glb,
                obj_path=canonical_obj,
            )
            service = self._build_service(settings)

            updated_result, updated_metadata = service._apply_forced_presentable_model_if_configured(
                project_id="demo-project",
                output_dir=output_dir,
                output_format=OutputFormat.GLB,
                result=result,
                metadata={"current_stage": "completed"},
            )

            expected_path = output_dir / "demo-project_model.glb"
            self.assertEqual(updated_result.model_path, expected_path)
            self.assertTrue(expected_path.exists())
            self.assertEqual(expected_path.read_bytes(), b"glTFdemo")
            self.assertTrue(updated_metadata["forced_presentable_model"]["applied"])
            self.assertEqual(updated_metadata["method_used"], "forced_presentable_model")

    def test_apply_forced_presentable_model_keeps_original_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            canonical_glb = root / "master.glb"
            canonical_obj = root / "master.obj"
            canonical_glb.write_bytes(b"glTFdemo")
            canonical_obj.write_text("o demo", encoding="utf-8")

            original = output_dir / "original_model.glb"
            original.write_bytes(b"old")
            result = ReconstructionResult(
                engine_name="mock",
                requested_output_format=OutputFormat.GLB,
                model_path=original,
                metadata={},
            )
            settings = self._build_settings(
                root,
                enabled=False,
                glb_path=canonical_glb,
                obj_path=canonical_obj,
            )
            service = self._build_service(settings)

            updated_result, updated_metadata = service._apply_forced_presentable_model_if_configured(
                project_id="demo-project",
                output_dir=output_dir,
                output_format=OutputFormat.GLB,
                result=result,
                metadata={"current_stage": "completed"},
            )

            self.assertEqual(updated_result.model_path, original)
            self.assertEqual(updated_metadata["current_stage"], "completed")

    def test_validate_final_artifact_skips_trimesh_if_forced_model_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical_glb = root / "master.glb"
            canonical_obj = root / "master.obj"
            canonical_glb.write_bytes(b"glTFdemo")
            canonical_obj.write_text("o demo", encoding="utf-8")
            settings = self._build_settings(
                root,
                enabled=True,
                glb_path=canonical_glb,
                obj_path=canonical_obj,
            )
            service = self._build_service(settings)

            output_dir = root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / "demo-project_model.glb"
            model_path.write_bytes(b"glTFdemo")
            result = ReconstructionResult(
                engine_name="mock",
                requested_output_format=OutputFormat.GLB,
                model_path=model_path,
                metadata={},
            )
            metadata = {
                "forced_presentable_model": {"applied": True},
                "metrics": {},
            }

            with patch("app.services.processing_service.importlib.import_module", side_effect=ImportError("missing")):
                service._validate_final_result_artifact(
                    project_id="demo-project",
                    result=result,
                    output_format=OutputFormat.GLB,
                    metadata=metadata,
                )

            self.assertTrue(metadata["metrics"]["mesh_validation_skipped"])
            self.assertEqual(
                metadata["metrics"]["mesh_validation_reason"],
                "missing_trimesh_dependency",
            )
            self.assertEqual(metadata["final_model_validation"]["status"], "skipped")
            self.assertFalse(metadata["final_model_validation"]["performed"])

    def test_forced_presentable_model_preserves_captured_texture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            canonical_glb = root / "master.glb"
            canonical_obj = root / "master.obj"
            canonical_glb.write_bytes(b"glTFdemo")
            canonical_obj.write_text("o demo", encoding="utf-8")

            original = output_dir / "original_model.glb"
            original.write_bytes(b"captured_texture_glb")
            result = ReconstructionResult(
                engine_name="mock",
                requested_output_format=OutputFormat.GLB,
                model_path=original,
                metadata={},
            )
            settings = self._build_settings(
                root,
                enabled=True,
                glb_path=canonical_glb,
                obj_path=canonical_obj,
            )
            service = self._build_service(settings)

            updated_result, updated_metadata = service._apply_forced_presentable_model_if_configured(
                project_id="demo-project",
                output_dir=output_dir,
                output_format=OutputFormat.GLB,
                result=result,
                metadata={
                    "approximate_geometry_fallback": {
                        "captured_texture": {
                            "applied": True,
                            "atlas_path": str(output_dir / "atlas.png"),
                        }
                    }
                },
            )

            self.assertEqual(updated_result.model_path, original)
            forced = updated_metadata.get("forced_presentable_model") or {}
            self.assertFalse(bool(forced.get("applied")))
            self.assertEqual(
                forced.get("reason"),
                "preserved_captured_texture_from_images",
            )

    def test_remove_non_canonical_model_variants_keeps_only_final_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            glb_path = output_dir / "demo-project_model.glb"
            obj_path = output_dir / "demo-project_model.obj"
            glb_path.write_bytes(b"glb")
            obj_path.write_text("o mesh", encoding="utf-8")

            ProcessingService._remove_non_canonical_model_variants(
                "demo-project",
                output_dir,
                glb_path,
            )

            self.assertTrue(glb_path.exists())
            self.assertFalse(obj_path.exists())


if __name__ == "__main__":
    unittest.main()
