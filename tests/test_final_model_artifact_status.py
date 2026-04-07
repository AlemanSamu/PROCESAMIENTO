import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.api.routes.projects import get_project_status
from app.core.errors import InvalidProjectStateError
from app.models.schemas import OutputFormat, ProjectMetadata, ProjectStatus
from app.services.engines.base_engine import ReconstructionResult
from app.services.processing_service import ProcessingService
from app.services.project_service import ProjectService
from app.services.storage_service import StorageService


class FinalModelStatusTests(unittest.TestCase):
    def test_project_status_exposes_final_model_fields(self) -> None:
        metadata = ProjectMetadata(
            id="demo-status",
            name="Demo",
            status=ProjectStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            image_count=12,
            output_format=OutputFormat.GLB,
            model_filename="demo_model.glb",
            processing_metadata={
                "engine": "colmap",
                "current_stage": "completed_with_fallback",
                "fallback_used": True,
                "final_model_type": "glb",
                "final_model_path": "C:/projects/demo/output/demo_model.glb",
                "method_used": "bounding_box",
            },
        )
        project_service = MagicMock()
        project_service.get_project.return_value = metadata

        response = get_project_status("demo-status", project_service=project_service)

        self.assertEqual(response.current_stage, "completed_with_fallback")
        self.assertTrue(response.fallback_used)
        self.assertEqual(response.final_model_type, "glb")
        self.assertEqual(response.final_model_path, "C:/projects/demo/output/demo_model.glb")
        self.assertEqual(response.method_used, "bounding_box")
        self.assertEqual(response.model_download_url, "/projects/demo-status/model")

    def test_project_status_exposes_specific_mapper_failure_message(self) -> None:
        friendly_message = (
            "COLMAP no logro registrar suficientes imagenes para reconstruir el modelo. "
            "Intenta capturar mas fotos con mejor traslape, buena iluminacion y mas textura visual."
        )
        metadata = ProjectMetadata(
            id="demo-failed-status",
            name="Demo failed",
            status=ProjectStatus.FAILED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            image_count=6,
            error_message=friendly_message,
            processing_metadata={
                "engine": "colmap",
                "current_stage": "mapper_failed_insufficient_registered_images",
                "status_message": friendly_message,
                "reason_code": "insufficient_registered_images",
                "registered_image_count": 0,
                "metrics": {"registered_image_count": 0},
            },
        )
        project_service = MagicMock()
        project_service.get_project.return_value = metadata

        response = get_project_status("demo-failed-status", project_service=project_service)

        self.assertEqual(response.current_stage, "mapper_failed_insufficient_registered_images")
        self.assertEqual(response.message, friendly_message)
        self.assertEqual(response.metrics["registered_image_count"], 0)
        self.assertEqual(response.processing_metadata["reason_code"], "insufficient_registered_images")



class FinalArtifactSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage_root = Path(self._tmp.name) / "projects"
        self.settings = SimpleNamespace(
            storage_root=self.storage_root,
            max_images_per_project=10,
            allowed_image_extensions=(".jpg", ".jpeg", ".png"),
            processing_engine="colmap",
            simulation_delay_seconds=0,
            colmap_path=None,
            colmap_binary="colmap",
            colmap_timeout_seconds=120,
            colmap_use_gpu=False,
            colmap_camera_model="SIMPLE_RADIAL",
            colmap_single_camera=True,
            colmap_fallback_to_mock=False,
        )
        self.storage_service = StorageService(self.settings)
        self.project_service = ProjectService(self.storage_service, self.settings)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_get_model_file_rejects_sparse_point_cloud_when_project_claims_glb(self) -> None:
        project_id = "demo-points"
        project_dir = self.storage_service.get_project_dir(project_id)
        output_dir = self.storage_service.get_output_dir(project_id)
        project_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "points3D.ply").write_text("ply", encoding="utf-8")

        metadata = ProjectMetadata(
            id=project_id,
            name="Demo",
            status=ProjectStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            image_count=4,
            output_format=OutputFormat.GLB,
            model_filename="points3D.ply",
            processing_metadata={
                "current_stage": "completed_with_fallback",
                "final_model_path": str(output_dir / "demo_model.glb"),
                "fallback_used": True,
                "method_used": "convex_hull",
            },
        )
        self.storage_service.save_project_metadata(metadata)

        with self.assertRaises(InvalidProjectStateError) as context:
            self.project_service.get_model_file(project_id)

        self.assertIn("formato esperado", str(context.exception))

    def test_processing_service_marks_failed_when_glb_has_no_faces(self) -> None:
        class _FaceLessEngine:
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
                        "sparse_fallback": {
                            "used": True,
                            "mesh_method": "convex_hull",
                        },
                    },
                )

        class _FaceLessMesh:
            def __init__(self) -> None:
                self.vertices = [(0.0, 0.0, 0.0)] * 8
                self.faces = []

        class _FaceLessScene:
            def __init__(self) -> None:
                self.geometry = {"mesh_0": _FaceLessMesh()}

        class _FakeTrimeshModule:
            @staticmethod
            def load(_path, file_type=None, force=None):
                if file_type == "glb" and force == "scene":
                    return _FaceLessScene()
                raise ValueError(file_type)

        project_id = "demo-invalid-glb"
        images_dir = self.storage_service.get_images_dir(project_id)
        output_dir = self.storage_service.get_output_dir(project_id)
        images_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        (images_dir / "img_1.jpg").write_bytes(b"img1")
        (images_dir / "img_2.jpg").write_bytes(b"img2")

        metadata = ProjectMetadata(
            id=project_id,
            name="Demo",
            status=ProjectStatus.READY,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            image_count=2,
            image_files=["img_1.jpg", "img_2.jpg"],
        )
        self.storage_service.save_project_metadata(metadata)
        self.project_service.mark_processing(
            project_id,
            OutputFormat.GLB,
            processing_metadata={
                "current_stage": "queued",
                "progress": 0.0,
            },
        )

        model_path = output_dir / "demo_model.glb"
        with patch(
            "app.services.processing_service.build_reconstruction_engines",
            return_value=(_FaceLessEngine(model_path), None),
        ):
            service = ProcessingService(self.project_service, self.storage_service, self.settings)

        with patch("app.services.processing_service.importlib.import_module", return_value=_FakeTrimeshModule()):
            service._run_reconstruction_job(project_id, OutputFormat.GLB)

        failed = self.project_service.get_project(project_id)
        self.assertEqual(failed.status, ProjectStatus.FAILED)
        self.assertIn("no contiene caras", failed.error_message or "")
        self.assertEqual((failed.processing_metadata or {}).get("current_stage"), "failed")


if __name__ == "__main__":
    unittest.main()
