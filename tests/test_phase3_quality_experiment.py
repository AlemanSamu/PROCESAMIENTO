from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw

from app.api.routes.projects import get_project_result
from app.core.errors import ProcessingError
from app.models.schemas import OutputFormat, ProjectMetadata, ProjectStatus
from app.services.engines.base_engine import ReconstructionEngine
from app.services.engines.colmap_engine import ColmapReconstructionEngine
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
                "gpu_requested": True,
                "gpu_fallback_to_cpu": False,
                "gpu_error_message": None,
            },
            retryable=False,
        )


class Phase3QualityExperimentTests(unittest.TestCase):
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
            metrics_experiment_scenario="phase3",
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

    def test_colmap_config_diagnostic(self) -> None:
        module = self._load_script_module("scripts/check_colmap_setup.py", "check_colmap_setup")
        with patch.dict("os.environ", {"LOCAL3D_COLMAP_BINARY": "colmap"}, clear=False):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = module.main()

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertIn("local3d_colmap_binary", payload)
        self.assertIn("path_colmap", payload)
        self.assertIn("gpu_probe", payload)
        self.assertIn("profiles", payload)
        self.assertIn("ready_for_colmap", payload)

    def test_quality_report_created(self) -> None:
        project_id = self._create_project_with_images("phase3-quality")
        self.project_service.mark_processing(project_id, OutputFormat.GLB, processing_metadata={"current_stage": "queued"})

        service = self._build_processing_service()
        service._run_reconstruction_job(project_id, OutputFormat.GLB)

        quality_report_path = self.storage_service.get_output_dir(project_id) / "pipeline" / "quality_report.json"
        self.assertTrue(quality_report_path.exists())
        payload = json.loads(quality_report_path.read_text(encoding="utf-8"))
        self.assertIn(payload.get("quality_classification"), {"fallback_completed", "success_sparse_only", "success_approx_surface", "success_real", "failed"})
        self.assertIsInstance(payload.get("metrics"), dict)

    def test_result_contains_quality_report(self) -> None:
        project_id = self._create_project_with_images("phase3-result")
        self.project_service.mark_processing(project_id, OutputFormat.GLB, processing_metadata={"current_stage": "queued"})
        service = self._build_processing_service()
        service._run_reconstruction_job(project_id, OutputFormat.GLB)

        payload = get_project_result(project_id, project_service=self.project_service)
        self.assertEqual(payload["project_id"], project_id)
        self.assertIsInstance(payload.get("quality_report"), dict)
        self.assertIn("quality_classification", payload["quality_report"])

    def test_gpu_metadata_fields_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            report_path = ColmapReconstructionEngine.write_failure_report(
                project_id="gpu-meta",
                output_dir=output_dir,
                colmap_binary="colmap",
                profile="balanced",
                failure_reason="COLMAP no disponible",
                error_context={"gpu_requested": True, "gpu_fallback_to_cpu": False, "gpu_error_message": None},
                fallback_used=True,
            )
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("gpu_requested", payload)
            self.assertIn("gpu_used", payload)
            self.assertIn("gpu_fallback_to_cpu", payload)
            self.assertIn("gpu_error_message", payload)

    def test_experiment_report_generation(self) -> None:
        module = self._load_script_module("scripts/run_experiment.py", "run_experiment")
        reports_dir = self.root / "reports"
        outputs = module.write_profile_comparison_reports(
            runs=[
                {
                    "profile": "conservative",
                    "project_id": "a1",
                    "status": "completed",
                    "quality_classification": "fallback_completed",
                    "total_time_seconds": 12.5,
                    "images_accepted": 12,
                    "cameras_reconstructed": 0,
                    "points_3d_count": 0,
                    "fallback_used": True,
                    "model_size_bytes": 1024,
                    "output_dir": "tmp/a1",
                },
                {
                    "profile": "balanced",
                    "project_id": "b1",
                    "status": "completed",
                    "quality_classification": "success_sparse_only",
                    "total_time_seconds": 18.2,
                    "images_accepted": 12,
                    "cameras_reconstructed": 8,
                    "points_3d_count": 1240,
                    "fallback_used": False,
                    "model_size_bytes": 4096,
                    "output_dir": "tmp/b1",
                },
            ],
            reports_dir=reports_dir,
            input_dir=self.root / "images",
            output_format="glb",
        )
        self.assertTrue(outputs["json"].exists())
        self.assertTrue(outputs["csv"].exists())
        self.assertIn("quality_classification", outputs["csv"].read_text(encoding="utf-8"))

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

    @staticmethod
    def _load_script_module(relative_path: str, module_name: str):
        script_path = Path(relative_path)
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"No se pudo cargar el script: {relative_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
