from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw

from app.algorithms.image_preprocessor import ImagePreprocessor
from app.api.routes.projects import get_project_result
from app.core.errors import ProcessingError
from app.models.schemas import OutputFormat, ProjectMetadata, ProjectStatus
from app.services.engines.base_engine import ReconstructionEngine, ReconstructionResult
from app.services.processing_service import ProcessingService
from app.services.project_service import ProjectService
from app.services.storage_service import StorageService


class _UnavailableColmapEngine(ReconstructionEngine):
    name = "colmap"

    def is_available(self) -> bool:
        return False

    def reconstruct(self, project_id, images_dir, output_dir, output_format, progress_callback=None):
        raise ProcessingError(
            "COLMAP no esta disponible para esta prueba.",
            reason_code="colmap_unavailable",
            current_stage="starting",
            metadata={
                "current_stage": "starting",
                "reason_code": "colmap_unavailable",
                "status_message": "COLMAP no disponible.",
            },
            retryable=False,
        )


class Phase2PreprocessingFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.storage_root = self.root / "projects"
        self.settings = SimpleNamespace(
            storage_root=self.storage_root,
            profile="balanced",
            processing_engine="auto",
            simulation_delay_seconds=0,
            processing_cleanup_workspace_on_failure=True,
            processing_execution_timeline_limit=200,
            image_preprocessing_max_width=320,
            allowed_image_extensions=(".jpg", ".jpeg", ".png"),
            max_images_per_project=20,
            image_validation_enabled=True,
            image_validation_min_images_required=3,
            image_validation_min_width=120,
            image_validation_min_height=120,
            image_validation_min_pixels=120 * 120,
            image_validation_min_sharpness_warn=0.01,
            image_validation_min_sharpness_reject=0.005,
            image_validation_min_brightness=0.05,
            image_validation_max_brightness=0.98,
            image_validation_exposure_warn_margin=0.02,
            image_validation_near_duplicate_warn_hamming=0,
            image_validation_near_duplicate_reject_hamming=0,
            image_validation_coverage_min_unique_ratio=0.1,
            image_validation_coverage_min_median_hamming=0,
            image_validation_coverage_max_neighbor_similarity_ratio=1.0,
            image_validation_block_on_low_coverage=False,
            image_selection_enabled=True,
            image_selection_min_images_required=3,
            image_selection_max_images=6,
            image_selection_target_keep_ratio=1.0,
            image_selection_min_quality_score=0.0,
            image_selection_quality_weight=0.65,
            image_selection_diversity_weight=0.35,
            image_selection_near_duplicate_hamming=0,
            image_selection_diversity_min_hamming=0,
            image_object_segmentation_enabled=False,
            image_object_segmentation_analysis_max_width=256,
            image_object_segmentation_min_component_area_ratio=0.003,
            image_object_segmentation_max_component_area_ratio=0.68,
            image_object_segmentation_min_component_fill_ratio=0.10,
            image_object_segmentation_expected_aspect_ratio=1.6,
            image_object_segmentation_aspect_tolerance=2.0,
            image_object_segmentation_mask_padding_ratio=0.045,
            image_object_segmentation_min_component_score=0.16,
            image_object_segmentation_min_segmented_images=2,
            image_object_segmentation_min_segmented_ratio=0.20,
            image_object_segmentation_block_on_low_success=False,
            primitive_box_fallback_enabled=True,
            primitive_box_fallback_min_selected_images=3,
            primitive_box_fallback_analysis_max_width=256,
            primitive_box_fallback_min_foreground_ratio=0.01,
            primitive_box_fallback_texture_enabled=True,
            primitive_box_fallback_on_incoherent_output=True,
            primitive_box_fallback_incoherent_min_registered_images=8,
            primitive_box_fallback_incoherent_min_sparse_points=1200,
            primitive_box_fallback_incoherent_min_points_per_registered_image=180,
            primitive_box_fallback_incoherent_min_faces=20,
            primitive_box_fallback_incoherent_max_faces=700,
            primitive_box_fallback_incoherent_max_extent_ratio=6.0,
            primitive_box_fallback_incoherent_min_bbox_fill_ratio=0.12,
            primitive_box_fallback_incoherent_max_bbox_fill_ratio=1.05,
            primitive_box_fallback_replace_sparse_bounding_box=True,
            metrics_evidence_enabled=False,
            metrics_evidence_root=self.root / "experiments",
            metrics_experiment_variant="test",
            metrics_experiment_scenario="phase2",
            force_presentable_model_enabled=False,
            force_presentable_model_glb=None,
            force_presentable_model_obj=None,
            colmap_binary="colmap",
            colmap_timeout_seconds=60,
            colmap_use_gpu=True,
            colmap_gpu_mode="auto",
            colmap_gpu_probe_timeout_seconds=1,
            colmap_enable_dense_stages=False,
            colmap_camera_model="SIMPLE_RADIAL",
            colmap_single_camera=True,
            colmap_fallback_to_mock=False,
            colmap_require_dense_reconstruction=False,
        )
        self.storage_service = StorageService(self.settings)
        self.project_service = ProjectService(self.storage_service, self.settings)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_image_preprocessing_profile(self) -> None:
        images_dir = self.root / "images"
        images_dir.mkdir()
        self._write_image(images_dir / "large.jpg", size=(900, 500), offset=8)

        preprocessor = ImagePreprocessor(profile="conservative", max_width=300)
        processed, report = preprocessor.run(images_dir, self.root / "pipeline", output_images_dir=self.root / "preprocessed")

        self.assertEqual(len(processed), 1)
        self.assertLessEqual(processed[0].source.width, 900)
        self.assertEqual(report.metrics["profile"], "conservative")
        self.assertEqual(report.metrics["max_width"], 300)
        self.assertLessEqual(Image.open(processed[0].preprocessed_path).width, 300)

    def test_preprocessing_manifest_created(self) -> None:
        images_dir = self.root / "images_manifest"
        images_dir.mkdir()
        self._write_image(images_dir / "sample.jpg", size=(640, 360), offset=12)

        preprocessor = ImagePreprocessor(profile="balanced", max_width=320)
        _processed, report = preprocessor.run(images_dir, self.root / "pipeline_manifest", output_images_dir=self.root / "preprocessed_manifest")

        self.assertIsNotNone(report.artifact_path)
        manifest = report.artifact_path.read_text(encoding="utf-8")
        self.assertIn('"profile": "balanced"', manifest)
        self.assertIn('"processed_path"', manifest)
        self.assertIn('"transformations"', manifest)

    def test_fallback_report_created_when_colmap_unavailable(self) -> None:
        project_id = self._create_project_with_images("phase2-fallback")
        self.project_service.mark_processing(project_id, OutputFormat.GLB, processing_metadata={"current_stage": "queued"})

        service = self._build_processing_service()
        service._run_reconstruction_job(project_id, OutputFormat.GLB)

        completed = self.project_service.get_project(project_id)
        self.assertEqual(completed.status, ProjectStatus.COMPLETED)
        metadata = completed.processing_metadata or {}
        self.assertTrue(metadata.get("fallback_used"))
        fallback_report = self.storage_service.get_output_dir(project_id) / "pipeline" / "fallback_report.json"
        self.assertTrue(fallback_report.exists())
        self.assertIn("primitive_box_academic_fallback", fallback_report.read_text(encoding="utf-8"))

    def test_result_endpoint_contains_preprocessing_info(self) -> None:
        project_id = self._create_project_with_images("phase2-result")
        self.project_service.mark_processing(project_id, OutputFormat.GLB, processing_metadata={"current_stage": "queued"})
        service = self._build_processing_service()
        service._run_reconstruction_job(project_id, OutputFormat.GLB)

        payload = get_project_result(project_id, project_service=self.project_service)

        self.assertEqual(payload["project_id"], project_id)
        self.assertIsInstance(payload.get("preprocessing_summary"), dict)
        self.assertIsInstance(payload.get("fallback_report"), dict)
        self.assertIn("artifact_paths", payload)
        self.assertIn("recommended_next_action", payload)

    def _build_processing_service(self) -> ProcessingService:
        with patch(
            "app.services.processing_service.build_reconstruction_engines",
            return_value=(_UnavailableColmapEngine(), None),
        ):
            return ProcessingService(self.project_service, self.storage_service, self.settings)

    def _create_project_with_images(self, project_id: str) -> str:
        images_dir = self.storage_service.get_images_dir(project_id)
        images_dir.mkdir(parents=True, exist_ok=True)
        for index in range(3):
            self._write_image(images_dir / f"img_{index + 1}.jpg", size=(640, 480), offset=index * 16)
        metadata = ProjectMetadata(
            id=project_id,
            name=f"Proyecto {project_id}",
            status=ProjectStatus.READY,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            image_count=3,
            image_files=[f"img_{index + 1}.jpg" for index in range(3)],
        )
        self.storage_service.save_project_metadata(metadata)
        return project_id

    @staticmethod
    def _write_image(path: Path, *, size: tuple[int, int], offset: int) -> None:
        canvas = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(canvas)
        width, height = size
        draw.rectangle(
            (40 + offset, 50, width - 80 + offset // 2, height - 80),
            fill=(210, 230, 245),
            outline="black",
            width=5,
        )
        draw.line((20, 20 + offset, width - 20, height - 40), fill="crimson", width=4)
        draw.ellipse((90, 90 + offset, 220, 220 + offset), outline="navy", width=4)
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, format="JPEG", quality=95)


if __name__ == "__main__":
    unittest.main()
