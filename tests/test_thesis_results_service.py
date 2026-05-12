from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from app.services.thesis_results_service import (
    build_scenario_variant_rows,
    build_variant_metrics_rows,
    generate_thesis_results_package,
)


class ThesisResultsServiceTests(unittest.TestCase):
    def test_build_variant_and_scenario_tables(self) -> None:
        runs_rows = [
            {
                "status": "completed",
                "variant": "baseline",
                "scenario_label": "good",
                "total_images_loaded": "10",
                "accepted_images": "9",
                "rejected_images": "1",
                "selected_images": "9",
                "selection_discarded_images": "0",
                "input_reduction_pct": "0",
                "total_processing_seconds": "5.0",
                "blocked_by_input_deficiency": "False",
            },
            {
                "status": "completed",
                "variant": "enhanced",
                "scenario_label": "good",
                "total_images_loaded": "10",
                "accepted_images": "6",
                "rejected_images": "4",
                "selected_images": "4",
                "selection_discarded_images": "2",
                "input_reduction_pct": "33.3",
                "total_processing_seconds": "3.0",
                "blocked_by_input_deficiency": "False",
            },
            {
                "status": "failed",
                "variant": "enhanced",
                "scenario_label": "bad",
                "total_images_loaded": "5",
                "accepted_images": "0",
                "rejected_images": "5",
                "selected_images": "0",
                "selection_discarded_images": "0",
                "input_reduction_pct": "0",
                "total_processing_seconds": "0.2",
                "blocked_by_input_deficiency": "True",
            },
        ]

        variant_rows = build_variant_metrics_rows(runs_rows)
        baseline = next(row for row in variant_rows if row["variant"] == "baseline")
        enhanced = next(row for row in variant_rows if row["variant"] == "enhanced")

        self.assertEqual(baseline["run_count"], 1)
        self.assertEqual(baseline["success_rate_pct"], 100.0)
        self.assertEqual(enhanced["run_count"], 2)
        self.assertEqual(enhanced["failure_rate_pct"], 50.0)
        self.assertEqual(enhanced["blocked_rate_pct"], 50.0)

        scenario_rows = build_scenario_variant_rows(runs_rows)
        enhanced_bad = next(
            row for row in scenario_rows if row["scenario"] == "bad" and row["variant"] == "enhanced"
        )
        self.assertEqual(enhanced_bad["failure_rate_pct"], 100.0)
        self.assertEqual(enhanced_bad["blocked_rate_pct"], 100.0)

    def test_generate_thesis_package_writes_files(self) -> None:
        summary_payload = {
            "overall": {
                "run_count": 2,
                "completed_count": 1,
                "failed_count": 1,
                "success_rate": 0.5,
                "failure_rate": 0.5,
                "input_blocked_rate": 0.5,
                "fallback_rate": 0.0,
            },
            "by_variant": {
                "baseline": {
                    "run_count": 1,
                    "completed_count": 1,
                    "failed_count": 0,
                    "success_rate": 1.0,
                    "failure_rate": 0.0,
                    "input_blocked_rate": 0.0,
                    "fallback_rate": 0.0,
                    "avg_total_processing_seconds": 5.0,
                    "median_total_processing_seconds": 5.0,
                    "avg_input_reduction_pct": 0.0,
                    "top_failed_stage": None,
                    "top_reason_code": None,
                },
                "enhanced": {
                    "run_count": 1,
                    "completed_count": 0,
                    "failed_count": 1,
                    "success_rate": 0.0,
                    "failure_rate": 1.0,
                    "input_blocked_rate": 1.0,
                    "fallback_rate": 0.0,
                    "avg_total_processing_seconds": 0.3,
                    "median_total_processing_seconds": 0.3,
                    "avg_input_reduction_pct": 40.0,
                    "top_failed_stage": "input_validation_failed",
                    "top_reason_code": "input_validation_failed",
                },
            },
            "by_scenario": {
                "good": {
                    "run_count": 1,
                    "completed_count": 1,
                    "failed_count": 0,
                    "success_rate": 1.0,
                    "failure_rate": 0.0,
                    "input_blocked_rate": 0.0,
                    "fallback_rate": 0.0,
                    "avg_total_processing_seconds": 5.0,
                    "median_total_processing_seconds": 5.0,
                    "avg_input_reduction_pct": 0.0,
                    "top_failed_stage": None,
                    "top_reason_code": None,
                },
                "bad": {
                    "run_count": 1,
                    "completed_count": 0,
                    "failed_count": 1,
                    "success_rate": 0.0,
                    "failure_rate": 1.0,
                    "input_blocked_rate": 1.0,
                    "fallback_rate": 0.0,
                    "avg_total_processing_seconds": 0.3,
                    "median_total_processing_seconds": 0.3,
                    "avg_input_reduction_pct": 40.0,
                    "top_failed_stage": "input_validation_failed",
                    "top_reason_code": "input_validation_failed",
                },
            },
        }
        runs_rows = [
            {
                "status": "completed",
                "variant": "baseline",
                "scenario_label": "good",
                "total_images_loaded": "10",
                "accepted_images": "9",
                "rejected_images": "1",
                "selected_images": "9",
                "selection_discarded_images": "0",
                "input_reduction_pct": "0",
                "total_processing_seconds": "5.0",
                "blocked_by_input_deficiency": "False",
            },
            {
                "status": "failed",
                "variant": "enhanced",
                "scenario_label": "bad",
                "total_images_loaded": "5",
                "accepted_images": "0",
                "rejected_images": "5",
                "selected_images": "0",
                "selection_discarded_images": "0",
                "input_reduction_pct": "0",
                "total_processing_seconds": "0.3",
                "blocked_by_input_deficiency": "True",
            },
        ]
        reason_rows = [
            {
                "variant": "enhanced",
                "reason_type": "validation_rejected_reason_counts",
                "reason": "blurry",
                "count": "5",
            }
        ]
        stage_rows: list[dict[str, str]] = []

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "thesis"
            outputs = generate_thesis_results_package(
                summary_payload=summary_payload,
                runs_rows=runs_rows,
                reason_rows=reason_rows,
                stage_rows=stage_rows,
                output_dir=output_dir,
                baseline_variant="baseline",
                enhanced_variant="enhanced",
                source_paths={
                    "summary_json": Path("summary.json"),
                    "runs_csv": Path("runs.csv"),
                    "reasons_csv": Path("reasons.csv"),
                    "stage_csv": Path("stage.csv"),
                },
            )

            self.assertIn("chapter_markdown", outputs)
            for path in outputs.values():
                self.assertTrue(path.exists())

            chapter_text = outputs["chapter_markdown"].read_text(encoding="utf-8")
            self.assertIn("## 1. Introduccion de pruebas", chapter_text)
            self.assertIn("No se registraron tiempos por etapa", chapter_text)

            with outputs["table_baseline_vs_enhanced"].open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["metric"] == "success_rate" for row in rows))


if __name__ == "__main__":
    unittest.main()
