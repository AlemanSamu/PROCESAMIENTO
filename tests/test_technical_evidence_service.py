from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.services.technical_evidence_service import (
    TechnicalEvidenceService,
    build_experiment_summary,
    load_run_records,
    write_experiment_reports,
)


class TechnicalEvidenceServiceTests(unittest.TestCase):
    def _build_settings(self, root: Path) -> SimpleNamespace:
        return SimpleNamespace(
            metrics_evidence_enabled=True,
            metrics_evidence_root=root,
            metrics_experiment_variant="enhanced",
            metrics_experiment_scenario="auto",
        )

    def test_write_run_evidence_generates_report_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "project" / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            final_model = output_dir / "demo_model.glb"
            final_model.write_bytes(b"glTFdemo")

            service = TechnicalEvidenceService(self._build_settings(root / "experiments"))
            metadata = {
                "metrics": {
                    "image_count_received": 24,
                    "image_count_accepted": 20,
                    "image_count_rejected": 4,
                    "image_count_warned": 3,
                    "image_count_selected": 12,
                    "image_count_discarded_selection": 8,
                    "total_processing_seconds": 42.7,
                },
                "input_validation": {
                    "allow_processing": True,
                    "rejected_reason_counts": {"underexposed": 2, "blurry": 2},
                    "warning_reason_counts": {"slightly_blurry": 3},
                    "coverage": {"possible_low_coverage": False},
                },
                "input_selection": {
                    "allow_processing": True,
                    "discarded_reason_counts": {"near_duplicate_of:img_01.jpg": 5},
                    "comparison": {"reduction_ratio": 0.4},
                },
                "stage_timings_seconds": {"mapper": 9.5, "feature_extractor": 7.0},
                "artifacts": {"model_path": str(final_model)},
                "current_stage": "completed",
            }
            report_path = service.write_run_evidence(
                project_id="demo",
                output_dir=output_dir,
                processing_metadata=metadata,
                project_status="completed",
            )

            self.assertIsNotNone(report_path)
            self.assertTrue(report_path.exists())
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report_payload["run_info"]["project_id"], "demo")
            self.assertEqual(report_payload["run_info"]["status"], "completed")
            self.assertEqual(report_payload["input_metrics"]["total_images_loaded"], 24)
            self.assertAlmostEqual(report_payload["input_metrics"]["input_reduction_pct"], 40.0)
            self.assertEqual(report_payload["artifact_metrics"]["final_artifact_type"], "glb")
            self.assertEqual(report_payload["artifact_metrics"]["final_artifact_size_bytes"], len(b"glTFdemo"))

            history_path = root / "experiments" / "processing_runs.ndjson"
            self.assertTrue(history_path.exists())
            records = load_run_records(history_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["run_info"]["variant"], "enhanced")

    def test_build_run_record_marks_blocked_input_as_bad_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = TechnicalEvidenceService(self._build_settings(root / "experiments"))
            record = service.build_run_record(
                project_id="blocked-case",
                project_status="failed",
                processing_metadata={
                    "reason_code": "input_validation_failed",
                    "current_stage": "input_validation_failed",
                    "metrics": {
                        "image_count_received": 8,
                        "image_count_accepted": 1,
                        "image_count_rejected": 7,
                        "image_count_selected": 0,
                    },
                    "input_validation": {
                        "allow_processing": False,
                        "blocking_reasons": ["insufficient_valid_images"],
                        "coverage": {"possible_low_coverage": True},
                    },
                },
            )

            self.assertEqual(record["run_info"]["scenario_label"], "bad")
            self.assertTrue(record["quality_gates"]["blocked_by_input_deficiency"])
            self.assertEqual(record["execution_metrics"]["reason_code"], "input_validation_failed")

    def test_experiment_summary_and_tables_are_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [
                {
                    "run_id": "r1",
                    "generated_at": "2026-01-01T00:00:00+00:00",
                    "run_info": {
                        "project_id": "p1",
                        "status": "completed",
                        "variant": "baseline",
                        "scenario_label": "good",
                    },
                    "input_metrics": {
                        "total_images_loaded": 20,
                        "accepted_images": 20,
                        "rejected_images": 0,
                        "warning_images": 0,
                        "selected_images": 20,
                        "selection_discarded_images": 0,
                        "input_reduction_pct": 0.0,
                    },
                    "execution_metrics": {
                        "total_processing_seconds": 90.0,
                        "stage_timings_seconds": {"mapper": 35.0},
                        "fallback_used": False,
                    },
                    "quality_gates": {"blocked_by_input_deficiency": False},
                    "reason_frequencies": {"validation_rejected_reason_counts": {}},
                    "artifact_metrics": {"final_artifact_type": "glb", "final_artifact_size_bytes": 1024},
                },
                {
                    "run_id": "r2",
                    "generated_at": "2026-01-01T00:10:00+00:00",
                    "run_info": {
                        "project_id": "p2",
                        "status": "completed",
                        "variant": "enhanced",
                        "scenario_label": "good",
                    },
                    "input_metrics": {
                        "total_images_loaded": 20,
                        "accepted_images": 18,
                        "rejected_images": 2,
                        "warning_images": 1,
                        "selected_images": 12,
                        "selection_discarded_images": 6,
                        "input_reduction_pct": 33.3,
                    },
                    "execution_metrics": {
                        "total_processing_seconds": 61.0,
                        "stage_timings_seconds": {"mapper": 22.0},
                        "fallback_used": False,
                    },
                    "quality_gates": {"blocked_by_input_deficiency": False},
                    "reason_frequencies": {"validation_rejected_reason_counts": {"blurry": 2}},
                    "artifact_metrics": {"final_artifact_type": "glb", "final_artifact_size_bytes": 1200},
                },
            ]
            summary = build_experiment_summary(runs, before_variant="baseline", after_variant="enhanced")
            self.assertEqual(summary["run_count"], 2)
            self.assertIn("before_vs_after", summary)
            self.assertIsNotNone(summary["before_vs_after"])
            self.assertLess(summary["before_vs_after"]["avg_total_processing_seconds_delta"], 0)

            outputs = write_experiment_reports(
                runs=runs,
                output_dir=root / "reports",
                before_variant="baseline",
                after_variant="enhanced",
            )
            for path in outputs.values():
                self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
