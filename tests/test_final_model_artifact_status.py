import tempfile
import unittest
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

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
        self.assertEqual(response.stage_status, "completed")
        self.assertTrue(response.fallback_used)
        self.assertEqual(response.final_model_type, "glb")
        self.assertEqual(response.final_model_path, "C:/projects/demo/output/demo_model.glb")
        self.assertEqual(response.method_used, "bounding_box")
        self.assertEqual(response.model_download_url, "/projects/demo-status/model")

    def test_project_status_model_download_url_honors_api_prefix(self) -> None:
        metadata = ProjectMetadata(
            id="demo-prefixed-status",
            name="Demo prefixed",
            status=ProjectStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            image_count=4,
            output_format=OutputFormat.GLB,
            model_filename="demo_prefixed.glb",
            processing_metadata={
                "engine": "colmap",
                "current_stage": "completed",
            },
        )
        project_service = MagicMock()
        project_service.get_project.return_value = metadata

        with patch(
            "app.api.routes.projects.get_settings",
            return_value=SimpleNamespace(api_prefix="/api/v1"),
        ):
            response = get_project_status("demo-prefixed-status", project_service=project_service)

        self.assertEqual(response.model_download_url, "/api/v1/projects/demo-prefixed-status/model")

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
        self.assertEqual(response.stage_status, "failed")
        self.assertEqual(response.message, friendly_message)
        self.assertEqual(response.metrics["registered_image_count"], 0)
        self.assertEqual(response.processing_metadata["reason_code"], "insufficient_registered_images")

    def test_failed_status_hides_stale_model_fields(self) -> None:
        metadata = ProjectMetadata(
            id="demo-stale-model-on-failure",
            name="Demo stale",
            status=ProjectStatus.FAILED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            image_count=9,
            output_format=OutputFormat.GLB,
            model_filename="old_model.glb",
            error_message="Fallo de reconstruccion.",
            processing_metadata={
                "engine": "colmap",
                "current_stage": "quality_gate_incoherent_output",
                "final_model_type": "glb",
                "final_model_path": "C:/projects/demo/output/old_model.glb",
                "status_message": "Procesamiento fallido.",
            },
        )
        project_service = MagicMock()
        project_service.get_project.return_value = metadata

        response = get_project_status("demo-stale-model-on-failure", project_service=project_service)

        self.assertEqual(response.status, ProjectStatus.FAILED)
        self.assertIsNone(response.model_filename)
        self.assertIsNone(response.model_download_url)
        self.assertIsNone(response.final_model_type)
        self.assertIsNone(response.final_model_path)

    def test_failed_status_prefers_specific_error_message_over_generic_status_message(self) -> None:
        metadata = ProjectMetadata(
            id="demo-generic-failed-status",
            name="Demo failed generic",
            status=ProjectStatus.FAILED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            image_count=3,
            error_message="No se pudo validar el GLB final porque falta la dependencia 'trimesh'.",
            processing_metadata={
                "current_stage": "failed",
                "status_message": "Procesamiento fallido.",
            },
        )
        project_service = MagicMock()
        project_service.get_project.return_value = metadata

        response = get_project_status("demo-generic-failed-status", project_service=project_service)

        self.assertEqual(
            response.message,
            "No se pudo validar el GLB final porque falta la dependencia 'trimesh'.",
        )



class FinalArtifactSafetyTests(unittest.TestCase):
    @staticmethod
    def _write_valid_image(path: Path, offset: int = 0) -> None:
        canvas = Image.new("RGB", (800, 600), (120, 120, 120))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((80 + offset, 70, 720, 520), outline="black", fill=(170, 170, 170), width=5)
        draw.ellipse((180, 160 + offset, 520, 440 + offset), outline="blue", fill=(90, 90, 160), width=4)
        draw.line((100, 100 + offset, 700, 500 - offset), fill="red", width=5)
        canvas.save(path)

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage_root = Path(self._tmp.name) / "projects"
        self.settings = SimpleNamespace(
            storage_root=self.storage_root,
            max_images_per_project=10,
            allowed_image_extensions=(".jpg", ".jpeg", ".png"),
            image_validation_enabled=False,
            image_validation_min_images_required=1,
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
        self._write_valid_image(images_dir / "img_1.jpg")
        self._write_valid_image(images_dir / "img_2.jpg", offset=20)

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

    def test_processing_service_applies_presentation_profile_for_single_project(self) -> None:
        class _SparseEngine:
            name = "colmap"

            def reconstruct(self, project_id, _images_dir, output_dir, output_format, progress_callback=None):
                import numpy as np
                import trimesh

                sparse_dir = output_dir / "workspace" / "sparse" / "0"
                sparse_dir.mkdir(parents=True, exist_ok=True)
                points = np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [5.0, -5.0, 7.0],  # outlier intencional para validar limpieza
                    ],
                    dtype=float,
                )
                trimesh.points.PointCloud(points).export(sparse_dir / "points3D.ply")

                model_path = output_dir / f"{project_id}_model.glb"
                initial_mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
                model_path.write_bytes(initial_mesh.export(file_type="glb"))
                return ReconstructionResult(
                    engine_name="colmap",
                    requested_output_format=output_format,
                    model_path=model_path,
                    metadata={
                        "engine": "colmap",
                        "current_stage": "completed_with_fallback",
                        "sparse_fallback": {
                            "used": True,
                            "mesh_method": "delaunay_mesher_sparse",
                        },
                    },
                )

        project_id = "demo-presentable"
        images_dir = self.storage_service.get_images_dir(project_id)
        output_dir = self.storage_service.get_output_dir(project_id)
        images_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_valid_image(images_dir / "img_1.jpg")
        self._write_valid_image(images_dir / "img_2.jpg", offset=24)

        metadata = ProjectMetadata(
            id=project_id,
            name="Demo Presentable",
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

        profile_path = self.storage_service.get_project_dir(project_id) / "presentation_profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "enabled": True,
                    "project_id": project_id,
                    "mode": "sparse_oriented_box_cleanup",
                    "bounds_trim_quantile": 0.05,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        with patch(
            "app.services.processing_service.build_reconstruction_engines",
            return_value=(_SparseEngine(), None),
        ):
            service = ProcessingService(self.project_service, self.storage_service, self.settings)

        service._run_reconstruction_job(project_id, OutputFormat.GLB)

        completed = self.project_service.get_project(project_id)
        self.assertEqual(completed.status, ProjectStatus.COMPLETED)
        processing = completed.processing_metadata or {}
        postprocess = processing.get("presentation_postprocess") or {}
        self.assertTrue(postprocess.get("applied"))
        self.assertEqual(postprocess.get("method"), "sparse_oriented_box_cleanup")
        self.assertEqual(processing.get("status_message"), "Reconstruccion completada con ajuste de presentacion sobre malla sparse.")
        metrics = processing.get("metrics") or {}
        self.assertGreater(int(metrics.get("mesh_vertex_count") or 0), 0)
        self.assertGreater(int(metrics.get("mesh_face_count") or 0), 0)
        self.assertEqual(completed.model_filename, f"{project_id}_model.glb")
        self.assertTrue((output_dir / f"{project_id}_model.glb").exists())
        self.assertTrue((output_dir / f"{project_id}_model.obj").exists())

    def test_mark_processing_and_failed_clear_stale_model_filename(self) -> None:
        project_id = "demo-clear-stale-model"
        metadata = ProjectMetadata(
            id=project_id,
            name="Demo stale model",
            status=ProjectStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            image_count=2,
            image_files=["img_1.jpg", "img_2.jpg"],
            output_format=OutputFormat.GLB,
            model_filename=f"{project_id}_model.glb",
        )
        self.storage_service.save_project_metadata(metadata)

        processing = self.project_service.mark_processing(
            project_id,
            OutputFormat.GLB,
            processing_metadata={"current_stage": "queued", "progress": 0.0},
        )
        self.assertEqual(processing.status, ProjectStatus.PROCESSING)
        self.assertIsNone(processing.model_filename)

        failed = self.project_service.mark_failed(
            project_id,
            "Fallo intencional de prueba.",
            processing_metadata={"current_stage": "failed"},
        )
        self.assertEqual(failed.status, ProjectStatus.FAILED)
        self.assertIsNone(failed.model_filename)


if __name__ == "__main__":
    unittest.main()
