import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from fastapi import UploadFile

from app.api.routes.projects import upload_images
from app.models.schemas import OutputFormat, ProjectStatus
from app.services.project_service import ProjectService
from app.services.storage_service import StorageService


class ProjectImageReloadEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage_root = Path(self._tmp.name) / "projects"
        self.settings = SimpleNamespace(
            storage_root=self.storage_root,
            max_images_per_project=10,
            allowed_image_extensions=(".jpg", ".jpeg", ".png"),
        )
        self.storage_service = StorageService(self.settings)
        self.project_service = ProjectService(self.storage_service, self.settings)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_reloading_same_project_with_same_images_skips_duplicates(self) -> None:
        project_id = "demo-reload-same"

        first_response = self._call_upload(
            project_id,
            [
                ("room_1.jpg", b"image-room-1"),
                ("room_2.jpg", b"image-room-2"),
            ],
        )
        self.assertEqual(first_response.uploaded_count, 2)
        self.assertEqual(first_response.skipped_count, 0)

        second_response = self._call_upload(
            project_id,
            [
                ("copy_room_1.jpg", b"image-room-1"),
                ("copy_room_2.jpg", b"image-room-2"),
            ],
        )

        self.assertEqual(second_response.project_id, project_id)
        self.assertEqual(second_response.status, ProjectStatus.READY)
        self.assertEqual(second_response.uploaded_count, 0)
        self.assertEqual(second_response.skipped_count, 2)
        self.assertEqual(second_response.total_images, 2)
        self.assertIn("omitidas por duplicadas", second_response.message or "")

        metadata = self.project_service.get_project(project_id)
        self.assertEqual(metadata.image_count, 2)
        self.assertEqual(len(metadata.image_files), 2)
        self.assertEqual(len(self.storage_service.list_image_files(project_id)), 2)

    def test_reloading_same_project_with_partially_new_images_adds_only_new_ones(self) -> None:
        project_id = "demo-reload-partial"

        first_response = self._call_upload(
            project_id,
            [
                ("scan_a.jpg", b"scan-a"),
                ("scan_b.jpg", b"scan-b"),
            ],
        )
        self.assertEqual(first_response.uploaded_count, 2)

        second_response = self._call_upload(
            project_id,
            [
                ("duplicate_scan_b.jpg", b"scan-b"),
                ("scan_c.jpg", b"scan-c"),
            ],
        )

        self.assertEqual(second_response.uploaded_count, 1)
        self.assertEqual(second_response.skipped_count, 1)
        self.assertEqual(second_response.total_images, 3)
        self.assertEqual(len(second_response.uploaded_files), 1)

        metadata = self.project_service.get_project(project_id)
        self.assertEqual(metadata.image_count, 3)
        self.assertEqual(len(self.storage_service.list_image_files(project_id)), 3)

    def test_reupload_cleans_previous_processing_outputs_and_resets_project_state(self) -> None:
        project_id = "demo-reprocess-cleanup"
        initial_response = self._call_upload(project_id, [("seed.jpg", b"seed-image")])
        self.assertEqual(initial_response.uploaded_count, 1)

        output_dir = self.storage_service.get_output_dir(project_id)
        (output_dir / "workspace" / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        (output_dir / "workspace" / "dense").mkdir(parents=True, exist_ok=True)
        (output_dir / "workspace" / "database.db").write_bytes(b"db")
        (output_dir / "workspace" / "sparse" / "0" / "points3D.bin").write_bytes(b"bin")
        (output_dir / "workspace" / "dense" / "fused.ply").write_text("ply", encoding="utf-8")
        (output_dir / "demo_model.glb").write_bytes(b"glb")
        (output_dir / "logs" / "colmap").mkdir(parents=True, exist_ok=True)
        (output_dir / "logs" / "colmap" / "run.log").write_text("log", encoding="utf-8")

        self.project_service.mark_completed(
            project_id,
            OutputFormat.GLB,
            "demo_model.glb",
            processing_metadata={
                "progress": 1.0,
                "current_stage": "completed",
                "status_message": "Listo",
            },
        )

        response = self._call_upload(project_id, [("same_seed.jpg", b"seed-image")])

        self.assertEqual(response.uploaded_count, 0)
        self.assertEqual(response.skipped_count, 1)
        self.assertIn("Se limpiaron artefactos previos", response.message or "")

        metadata = self.project_service.get_project(project_id)
        self.assertEqual(metadata.status, ProjectStatus.READY)
        self.assertIsNone(metadata.output_format)
        self.assertIsNone(metadata.model_filename)
        self.assertIsNone(metadata.error_message)
        self.assertIsNone(metadata.processing_metadata)
        self.assertEqual(metadata.image_count, 1)
        self.assertEqual(len(self.storage_service.list_image_files(project_id)), 1)
        self.assertEqual(list(output_dir.iterdir()), [])

    def _call_upload(self, project_id: str, files: list[tuple[str, bytes]]):
        upload_files = [
            UploadFile(file=BytesIO(content), filename=filename)
            for filename, content in files
        ]
        return upload_images(
            project_id=project_id,
            files=upload_files,
            project_service=self.project_service,
        )


if __name__ == "__main__":
    unittest.main()
