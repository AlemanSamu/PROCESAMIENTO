import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from app.models.schemas import OutputFormat, ProjectMetadata, ProjectStatus
from app.services.project_service import ProjectService
from app.services.storage_service import StorageService


class ProjectNameInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage_root = Path(self._tmp.name) / "projects"
        self.settings = SimpleNamespace(
            storage_root=self.storage_root,
            max_images_per_project=20,
            allowed_image_extensions=(".jpg", ".jpeg", ".png"),
        )
        self.storage_service = StorageService(self.settings)
        self.project_service = ProjectService(self.storage_service, self.settings)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _seed_project(self, *, project_id: str, name: str) -> None:
        now = datetime.now(timezone.utc)
        metadata = ProjectMetadata(
            id=project_id,
            name=name,
            status=ProjectStatus.READY,
            created_at=now,
            updated_at=now,
            image_count=8,
            image_files=[f"img_{index}.jpg" for index in range(8)],
        )
        self.storage_service.save_project_metadata(metadata)

    def _forced_metadata(self) -> dict[str, object]:
        return {
            "current_stage": "completed",
            "forced_presentable_model": {
                "applied": True,
                "source_glb": "data/canonical_models/azitromicina_canonical.glb",
            },
        }

    def test_mark_completed_replaces_generic_name_from_forced_model_source(self) -> None:
        project_id = "demo1234abcd"
        self._seed_project(project_id=project_id, name="Caja")

        updated = self.project_service.mark_completed(
            project_id=project_id,
            output_format=OutputFormat.GLB,
            model_filename=f"{project_id}_model.glb",
            processing_metadata=self._forced_metadata(),
        )

        self.assertEqual(updated.name, "Azitromicina")
        self.assertEqual(
            self.project_service.get_project(project_id).name,
            "Azitromicina",
        )

    def test_mark_completed_replaces_default_project_name_when_source_is_available(self) -> None:
        project_id = "cafebabefeed"
        self._seed_project(project_id=project_id, name=f"Proyecto-{project_id}")

        updated = self.project_service.mark_completed(
            project_id=project_id,
            output_format=OutputFormat.GLB,
            model_filename=f"{project_id}_model.glb",
            processing_metadata=self._forced_metadata(),
        )

        self.assertEqual(updated.name, "Azitromicina")

    def test_mark_completed_preserves_custom_name(self) -> None:
        project_id = "feedbead9001"
        self._seed_project(project_id=project_id, name="Azitromicina caja lateral")

        updated = self.project_service.mark_completed(
            project_id=project_id,
            output_format=OutputFormat.GLB,
            model_filename=f"{project_id}_model.glb",
            processing_metadata=self._forced_metadata(),
        )

        self.assertEqual(updated.name, "Azitromicina caja lateral")


if __name__ == "__main__":
    unittest.main()
