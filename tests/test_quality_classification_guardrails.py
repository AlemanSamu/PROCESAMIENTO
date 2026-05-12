from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from app.models.schemas import ProjectStatus
from app.services.processing_service import ProcessingService


class QualityClassificationGuardrailsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ProcessingService.__new__(ProcessingService)
        self.service.settings = SimpleNamespace(profile="balanced")
        self.service._engine = SimpleNamespace(name="colmap")

    def test_no_success_real_when_primitive_box(self) -> None:
        metadata = {
            "reconstruction_type": "approximate_box_primitive_fallback",
            "fallback_used": True,
            "registered_image_count": 30,
            "camera_count": 30,
            "point_count": 8000,
            "artifacts": {},
        }
        classification = self.service._classify_quality_result(
            project_status=ProjectStatus.COMPLETED,
            metadata=metadata,
        )
        self.assertEqual(classification, "fallback_completed")

    def test_sparse_low_points_is_not_success_real(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sparse_dir = Path(tmp) / "sparse"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "reconstruction_type": "sparse_photogrammetry_mesh_fallback",
                "fallback_used": False,
                "registered_image_count": 5,
                "camera_count": 5,
                "point_count": 300,
                "artifacts": {"sparse_txt_dir": str(sparse_dir)},
            }
            classification = self.service._classify_quality_result(
                project_status=ProjectStatus.COMPLETED,
                metadata=metadata,
            )
            self.assertEqual(classification, "fallback_completed")

    def test_sparse_medium_points_is_success_sparse_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sparse_dir = Path(tmp) / "sparse"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "reconstruction_type": "sparse_photogrammetry_point_cloud_fallback",
                "fallback_used": True,
                "registered_image_count": 16,
                "camera_count": 16,
                "point_count": 3000,
                "metrics": {"mesh_face_count": 0},
                "sparse_fallback": {"visualization_type": "point_spheres"},
                "artifacts": {"sparse_txt_dir": str(sparse_dir)},
            }
            classification = self.service._classify_quality_result(
                project_status=ProjectStatus.COMPLETED,
                metadata=metadata,
            )
            self.assertEqual(classification, "success_sparse_only")

    def test_sparse_high_points_is_success_sparse_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sparse_dir = Path(tmp) / "sparse"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "reconstruction_type": "sparse_photogrammetry_point_cloud_fallback",
                "fallback_used": True,
                "registered_image_count": 16,
                "camera_count": 16,
                "point_count": 8000,
                "metrics": {"mesh_face_count": 0},
                "sparse_fallback": {"visualization_type": "point_spheres"},
                "artifacts": {"sparse_txt_dir": str(sparse_dir)},
            }
            classification = self.service._classify_quality_result(
                project_status=ProjectStatus.COMPLETED,
                metadata=metadata,
            )
            self.assertEqual(classification, "success_sparse_only")

    def test_dense_mesh_with_faces_can_be_success_real(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dense_dir = Path(tmp) / "dense"
            dense_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "reconstruction_type": "dense_photogrammetry_mesh",
                "fallback_used": False,
                "registered_image_count": 16,
                "camera_count": 16,
                "point_count": 8000,
                "metrics": {
                    "mesh_face_count": 500,
                    "dense_faces_count_real": 500,
                    "dense_vertices_count_real": 300,
                },
                "artifacts": {"fused_ply_path": str(dense_dir / "fused.ply")},
            }
            (dense_dir / "fused.ply").write_text("ply", encoding="utf-8")
            classification = self.service._classify_quality_result(
                project_status=ProjectStatus.COMPLETED,
                metadata=metadata,
            )
            self.assertEqual(classification, "success_real")

    def test_sparse_surface_with_faces_can_be_success_approx_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sparse_dir = Path(tmp) / "sparse"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "reconstruction_type": "sparse_surface_reconstruction",
                "fallback_used": True,
                "registered_image_count": 26,
                "camera_count": 26,
                "point_count": 1630,
                "metrics": {
                    "mesh_face_count": 1200,
                    "surface_faces_count_real": 1200,
                    "surface_vertices_count_real": 320,
                },
                "surface_success": True,
                "artifacts": {"sparse_txt_dir": str(sparse_dir)},
            }
            classification = self.service._classify_quality_result(
                project_status=ProjectStatus.COMPLETED,
                metadata=metadata,
            )
            self.assertEqual(classification, "success_approx_surface")

    def test_sparse_surface_with_8_faces_is_fallback_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sparse_dir = Path(tmp) / "sparse"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "reconstruction_type": "sparse_surface_reconstruction",
                "registered_image_count": 26,
                "camera_count": 26,
                "point_count": 1637,
                "metrics": {"mesh_face_count": 8},
                "surface_attempted": True,
                "surface_success": False,
                "sparse_fallback": {"visualization_type": "point_spheres"},
                "artifacts": {"sparse_txt_dir": str(sparse_dir)},
            }
            classification = self.service._classify_quality_result(
                project_status=ProjectStatus.COMPLETED,
                metadata=metadata,
            )
            self.assertEqual(classification, "fallback_completed")

    def test_visual_faces_do_not_trigger_success_approx_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sparse_dir = Path(tmp) / "sparse"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "reconstruction_type": "sparse_photogrammetry_point_cloud_fallback",
                "registered_image_count": 26,
                "camera_count": 26,
                "point_count": 1637,
                "surface_attempted": True,
                "surface_success": False,
                "metrics": {
                    "mesh_face_count": 120000,
                    "mesh_face_count_is_visual_only": True,
                    "surface_faces_count_real": 76,
                    "surface_vertices_count_real": 40,
                },
                "sparse_fallback": {"visualization_type": "point_spheres"},
                "artifacts": {"sparse_txt_dir": str(sparse_dir)},
            }
            classification = self.service._classify_quality_result(
                project_status=ProjectStatus.COMPLETED,
                metadata=metadata,
            )
            self.assertEqual(classification, "success_sparse_only")

    def test_quality_report_includes_geometry_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metadata = {
                "engine": "colmap",
                "reconstruction_type": "sparse_photogrammetry_mesh_fallback",
                "registered_image_count": 8,
                "camera_count": 8,
                "point_count": 1200,
                "dense_stages_enabled": True,
                "sparse_fallback": {"final_mesh_method": "convex_hull"},
                "metrics": {"total_processing_seconds": 1.23},
                "artifacts": {"sparse_txt_dir": str(output_dir / "sparse")},
            }
            (output_dir / "sparse").mkdir(parents=True, exist_ok=True)
            updated = self.service._write_quality_report(
                project_id="demo",
                output_dir=output_dir,
                project_status=ProjectStatus.COMPLETED,
                metadata=metadata,
                model_path=None,
            )
            payload = updated["quality_report"]
            self.assertEqual(payload["geometry_source"], "colmap_sparse")
            self.assertIn("texture_source", payload)
            self.assertIn("mesh_quality_score", payload)
            self.assertIn("sparse_quality_score", payload)
            self.assertIn("sparse_density_level", payload)
            self.assertIn("visualization_type", payload)
            self.assertIn("segmentation_report", payload)
            self.assertIn("geometric_prior_report", payload)
            self.assertIn("texture_report", payload)
            self.assertIn("final_model_texture_quality", payload)
            self.assertIn("final_model_visual_score", payload)
            self.assertIn("final_model_is_presentable", payload)

    def test_sparse_886_points_quality_is_fallback_low_density(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            sparse_dir = output_dir / "sparse"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "engine": "colmap",
                "reconstruction_type": "sparse_photogrammetry_point_cloud_fallback",
                "registered_image_count": 16,
                "camera_count": 16,
                "point_count": 886,
                "dense_stages_enabled": True,
                "sparse_fallback": {"visualization_type": "point_spheres", "final_mesh_method": "point_cloud_from_sparse"},
                "metrics": {"total_processing_seconds": 1.23, "mesh_face_count": 0},
                "artifacts": {"sparse_txt_dir": str(sparse_dir)},
            }
            updated = self.service._write_quality_report(
                project_id="demo",
                output_dir=output_dir,
                project_status=ProjectStatus.COMPLETED,
                metadata=metadata,
                model_path=None,
            )
            payload = updated["quality_report"]
            self.assertEqual(payload["quality_classification"], "fallback_completed")
            self.assertEqual(payload["geometry_source"], "colmap_sparse_point_cloud")
            self.assertEqual(payload["sparse_density_level"], "low")

    def test_geometric_prior_never_becomes_success_real(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sparse_dir = Path(tmp) / "sparse"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "reconstruction_type": "sparse_geometric_prior_reconstruction",
                "registered_image_count": 24,
                "camera_count": 24,
                "point_count": 2000,
                "metrics": {"mesh_face_count": 1200},
                "artifacts": {"sparse_txt_dir": str(sparse_dir)},
            }
            classification = self.service._classify_quality_result(
                project_status=ProjectStatus.COMPLETED,
                metadata=metadata,
            )
            self.assertNotEqual(classification, "success_real")

    def test_surface_under_500_faces_is_not_success_approx_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sparse_dir = Path(tmp) / "sparse"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "reconstruction_type": "sparse_surface_reconstruction",
                "registered_image_count": 24,
                "camera_count": 24,
                "point_count": 2600,
                "surface_attempted": True,
                "surface_success": False,
                "metrics": {
                    "surface_faces_count_real": 420,
                    "surface_vertices_count_real": 130,
                    "mesh_face_count": 420,
                },
                "artifacts": {"sparse_txt_dir": str(sparse_dir)},
            }
            classification = self.service._classify_quality_result(
                project_status=ProjectStatus.COMPLETED,
                metadata=metadata,
            )
            self.assertEqual(classification, "fallback_completed")


class InspectReconstructionOutputScriptTests(unittest.TestCase):
    def test_script_runs_with_simulated_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_id = "demo123"
            project_dir = root / project_id
            pipeline = project_dir / "output" / "pipeline"
            sparse_txt = project_dir / "output" / "colmap_sparse_txt"
            sparse0 = project_dir / "output" / "workspace" / "sparse" / "0"
            logs_dir = project_dir / "output" / "logs" / "colmap"
            for path in (pipeline, sparse_txt, sparse0, logs_dir):
                path.mkdir(parents=True, exist_ok=True)

            (project_dir / "meta.json").write_text(
                json.dumps({"status": "completed", "processing_metadata": {"fallback_used": True}}),
                encoding="utf-8",
            )
            (pipeline / "quality_report.json").write_text(
                json.dumps(
                    {
                        "quality_classification": "fallback_completed",
                        "geometry_source": "primitive_box",
                        "cameras_reconstructed": 2,
                        "images_registered": 2,
                        "points3D_count": 40,
                        "dense_available": False,
                    }
                ),
                encoding="utf-8",
            )
            (pipeline / "colmap_report.json").write_text(
                json.dumps({"cameras_reconstructed": 2, "images_registered": 2, "points3D_count": 40}),
                encoding="utf-8",
            )
            (sparse_txt / "points3D.txt").write_text("# header\n1 0 0 0 255 255 255 0", encoding="utf-8")

            spec = importlib.util.spec_from_file_location(
                "inspect_reconstruction_output",
                Path("scripts/inspect_reconstruction_output.py"),
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            argv_backup = __import__("sys").argv
            __import__("sys").argv = [
                "inspect_reconstruction_output.py",
                "--project-id",
                project_id,
                "--projects-root",
                str(root),
            ]
            try:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = module.main()
            finally:
                __import__("sys").argv = argv_backup

            output = buffer.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("colmap_real_or_fallback: fallback", output)
            self.assertIn("is_primitive_box: True", output)
            self.assertIn("visualization_generated:", output)
            self.assertIn("defendible_as:", output)


if __name__ == "__main__":
    unittest.main()
