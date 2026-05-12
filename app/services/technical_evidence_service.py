from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from app.algorithms.artifacts import write_json

logger = logging.getLogger(__name__)

INPUT_BLOCKING_REASON_CODES = {"input_validation_failed", "input_selection_failed"}


class TechnicalEvidenceService:
    """Collects lightweight, normalized metrics for thesis-grade evidence."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.enabled = bool(getattr(settings, "metrics_evidence_enabled", False))
        configured_root = getattr(settings, "metrics_evidence_root", Path("data/experiments"))
        self.metrics_root = Path(configured_root)
        self.experiment_variant = str(getattr(settings, "metrics_experiment_variant", "enhanced")).strip() or "enhanced"
        self.configured_scenario = str(getattr(settings, "metrics_experiment_scenario", "auto")).strip().lower() or "auto"

    def write_run_evidence(
        self,
        *,
        project_id: str,
        output_dir: Path,
        processing_metadata: dict[str, Any],
        project_status: str,
    ) -> Path | None:
        if not self.enabled:
            return None

        run_record = self.build_run_record(
            project_id=project_id,
            processing_metadata=processing_metadata,
            project_status=project_status,
        )
        report_path = write_json(
            output_dir / "pipeline" / f"{project_id}_technical_evidence.json",
            run_record,
        )
        self._append_run_to_history(run_record)
        return report_path

    def build_run_record(
        self,
        *,
        project_id: str,
        processing_metadata: dict[str, Any],
        project_status: str,
    ) -> dict[str, Any]:
        metadata = dict(processing_metadata or {})
        metrics = dict(metadata.get("metrics") or {})
        input_validation = dict(metadata.get("input_validation") or {})
        input_selection = dict(metadata.get("input_selection") or {})
        execution_report = dict(metadata.get("execution_report") or {})
        stage_timings = self._normalize_stage_timings(metadata.get("stage_timings_seconds"))
        artifacts = dict(metadata.get("artifacts") or {})
        reason_code = str(metadata.get("reason_code") or "").strip() or None
        failed_stage = (
            str(metadata.get("failed_stage") or "").strip()
            or str(execution_report.get("failed_stage") or "").strip()
            or None
        )

        total_images = int(metrics.get("image_count_received") or input_validation.get("total_images") or 0)
        accepted_images = int(metrics.get("image_count_accepted") or input_validation.get("accepted_images") or 0)
        rejected_images = int(metrics.get("image_count_rejected") or input_validation.get("rejected_images") or 0)
        warning_images = int(metrics.get("image_count_warned") or input_validation.get("warning_images") or 0)
        selected_images = int(metrics.get("image_count_selected") or input_selection.get("selected_images") or 0)
        discarded_selection = int(
            metrics.get("image_count_discarded_selection") or input_selection.get("discarded_images") or 0
        )
        reduction_ratio = self._resolve_reduction_ratio(
            input_selection=input_selection,
            accepted_images=accepted_images,
            selected_images=selected_images,
        )
        reduction_pct = round(reduction_ratio * 100.0, 2)
        blocked_by_input = bool(
            reason_code in INPUT_BLOCKING_REASON_CODES
            or str(metadata.get("current_stage") or "").strip() in INPUT_BLOCKING_REASON_CODES
        )

        fallback_used = bool(
            metadata.get("fallback_used")
            or (metadata.get("fallback") or {}).get("used")
            or (metadata.get("sparse_fallback") or {}).get("used")
        )
        retryable = metadata.get("retryable")
        if not isinstance(retryable, bool):
            retryable = bool(metadata.get("can_retry")) if metadata.get("can_retry") is not None else None

        output_artifacts = self._collect_output_artifacts(metadata)
        final_artifact = self._resolve_final_artifact(output_artifacts)
        scenario_label, scenario_source = self._resolve_scenario(
            total_images=total_images,
            rejected_images=rejected_images,
            accepted_images=accepted_images,
            selected_images=selected_images,
            blocked_by_input=blocked_by_input,
            coverage=input_validation.get("coverage"),
        )

        total_processing_seconds = self._safe_float(metrics.get("total_processing_seconds"))
        stage_observed_count = len(list(execution_report.get("stages") or []))
        generated_at = self._utc_now_iso()
        run_id = f"{project_id}-{generated_at.replace(':', '').replace('-', '')}"

        return {
            "schema_version": "1.0",
            "run_id": run_id,
            "generated_at": generated_at,
            "run_info": {
                "project_id": project_id,
                "status": str(project_status or "").strip().lower(),
                "variant": self.experiment_variant,
                "scenario_label": scenario_label,
                "scenario_source": scenario_source,
            },
            "input_metrics": {
                "total_images_loaded": total_images,
                "accepted_images": accepted_images,
                "rejected_images": rejected_images,
                "warning_images": warning_images,
                "selected_images": selected_images,
                "selection_discarded_images": discarded_selection,
                "input_reduction_ratio": round(reduction_ratio, 4),
                "input_reduction_pct": reduction_pct,
            },
            "reason_frequencies": {
                "validation_rejected_reason_counts": dict(input_validation.get("rejected_reason_counts") or {}),
                "validation_warning_reason_counts": dict(input_validation.get("warning_reason_counts") or {}),
                "selection_discarded_reason_counts": dict(input_selection.get("discarded_reason_counts") or {}),
                "pipeline_blocking_reasons": {
                    str(reason): int(count)
                    for reason, count in self._to_counter(
                        list(input_validation.get("blocking_reasons") or [])
                        + list(input_selection.get("blocking_reasons") or [])
                    ).items()
                },
            },
            "execution_metrics": {
                "total_processing_seconds": total_processing_seconds,
                "stage_timings_seconds": stage_timings,
                "stage_observed_count": stage_observed_count,
                "current_stage": metadata.get("current_stage"),
                "failed_stage": failed_stage,
                "reason_code": reason_code,
                "retryable": retryable,
                "fallback_used": fallback_used,
                "completed_with_fallback": str(metadata.get("current_stage") or "").strip() == "completed_with_fallback",
            },
            "quality_gates": {
                "blocked_by_input_deficiency": blocked_by_input,
                "allow_processing_after_validation": bool(input_validation.get("allow_processing", True)),
                "allow_processing_after_selection": bool(input_selection.get("allow_processing", True)),
                "coverage_possible_low": bool((input_validation.get("coverage") or {}).get("possible_low_coverage")),
            },
            "artifact_metrics": {
                "final_artifact_type": final_artifact.get("file_type"),
                "final_artifact_size_bytes": final_artifact.get("size_bytes"),
                "output_artifacts": output_artifacts,
            },
            "sources": {
                "execution_report_path": artifacts.get("execution_report"),
                "input_validation_report_path": artifacts.get("input_validation_report"),
                "input_selection_report_path": artifacts.get("input_selection_report"),
            },
        }

    def _append_run_to_history(self, run_record: dict[str, Any]) -> None:
        history_path = self.metrics_root / "processing_runs.ndjson"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(run_record, ensure_ascii=False))
            handle.write("\n")

    def _resolve_scenario(
        self,
        *,
        total_images: int,
        rejected_images: int,
        accepted_images: int,
        selected_images: int,
        blocked_by_input: bool,
        coverage: Any,
    ) -> tuple[str, str]:
        configured = self.configured_scenario
        if configured and configured not in {"", "auto", "automatic", "inferred"}:
            return configured, "configured"

        if blocked_by_input:
            return "bad", "inferred"
        if total_images <= 0:
            return "unknown", "inferred"

        rejection_ratio = rejected_images / max(total_images, 1)
        selected_ratio = selected_images / max(accepted_images, 1) if accepted_images > 0 else 0.0
        coverage_low = bool((coverage or {}).get("possible_low_coverage")) if isinstance(coverage, dict) else False

        if rejection_ratio >= 0.45 or selected_ratio < 0.45:
            return "bad", "inferred"
        if rejection_ratio >= 0.2 or selected_ratio < 0.7 or coverage_low:
            return "mixed", "inferred"
        return "good", "inferred"

    @staticmethod
    def _resolve_reduction_ratio(
        *,
        input_selection: dict[str, Any],
        accepted_images: int,
        selected_images: int,
    ) -> float:
        comparison = input_selection.get("comparison") if isinstance(input_selection.get("comparison"), dict) else {}
        reported_ratio = comparison.get("reduction_ratio")
        if isinstance(reported_ratio, (int, float)):
            return max(0.0, min(1.0, float(reported_ratio)))
        if accepted_images <= 0:
            return 0.0
        removed = max(0, accepted_images - selected_images)
        return max(0.0, min(1.0, removed / accepted_images))

    @staticmethod
    def _normalize_stage_timings(stage_timings_raw: Any) -> dict[str, float]:
        if not isinstance(stage_timings_raw, dict):
            return {}
        normalized: dict[str, float] = {}
        for stage, seconds in stage_timings_raw.items():
            key = str(stage or "").strip()
            if not key:
                continue
            numeric = TechnicalEvidenceService._safe_float(seconds)
            if numeric is None:
                continue
            normalized[key] = round(max(0.0, numeric), 3)
        return normalized

    @staticmethod
    def _collect_output_artifacts(metadata: dict[str, Any]) -> list[dict[str, Any]]:
        artifacts = dict(metadata.get("artifacts") or {})
        candidates: dict[str, str] = {}
        for key in (
            "model_path",
            "obj_model_path",
            "glb_model_path",
            "raw_sparse_ply",
            "fused_ply_path",
            "poisson_mesh_ply",
            "execution_report",
            "input_validation_report",
            "input_selection_report",
        ):
            value = artifacts.get(key)
            if value:
                candidates[key] = str(value)

        metadata_output = metadata.get("output_path")
        if metadata_output:
            candidates.setdefault("output_path", str(metadata_output))

        summaries: list[dict[str, Any]] = []
        for name, path_value in sorted(candidates.items(), key=lambda item: item[0]):
            path = Path(path_value)
            exists = path.exists() and path.is_file()
            size_bytes = path.stat().st_size if exists else None
            summaries.append(
                {
                    "name": name,
                    "path": str(path),
                    "exists": exists,
                    "size_bytes": size_bytes,
                    "file_type": path.suffix.lower().lstrip(".") if path.suffix else None,
                }
            )
        return summaries

    @staticmethod
    def _resolve_final_artifact(output_artifacts: list[dict[str, Any]]) -> dict[str, Any]:
        priority = ["model_path", "glb_model_path", "obj_model_path", "output_path"]
        for name in priority:
            found = next((item for item in output_artifacts if item.get("name") == name), None)
            if found is not None:
                return found
        return output_artifacts[0] if output_artifacts else {}

    @staticmethod
    def _to_counter(values: list[Any]) -> dict[str, int]:
        counter: dict[str, int] = {}
        for value in values:
            key = str(value).strip()
            if not key:
                continue
            counter[key] = counter.get(key, 0) + 1
        return counter

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


def load_run_records(runs_file: Path) -> list[dict[str, Any]]:
    if not runs_file.exists() or not runs_file.is_file():
        return []

    records: list[dict[str, Any]] = []
    for raw_line in runs_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed NDJSON line in %s", runs_file)
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def build_experiment_summary(
    runs: list[dict[str, Any]],
    *,
    before_variant: str | None = None,
    after_variant: str | None = None,
) -> dict[str, Any]:
    normalized_runs = [record for record in runs if isinstance(record, dict)]
    by_variant = _group_by(normalized_runs, lambda record: _nested_str(record, "run_info", "variant") or "unknown")
    by_scenario = _group_by(normalized_runs, lambda record: _nested_str(record, "run_info", "scenario_label") or "unknown")

    variant_stats = {variant: _compute_group_stats(group) for variant, group in by_variant.items()}
    scenario_stats = {scenario: _compute_group_stats(group) for scenario, group in by_scenario.items()}
    global_stats = _compute_group_stats(normalized_runs)

    delta = None
    baseline = (before_variant or "").strip() or None
    improved = (after_variant or "").strip() or None
    if baseline and improved and baseline in variant_stats and improved in variant_stats:
        delta = _build_variant_delta(variant_stats[baseline], variant_stats[improved], baseline, improved)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_count": len(normalized_runs),
        "overall": global_stats,
        "by_variant": variant_stats,
        "by_scenario": scenario_stats,
        "before_vs_after": delta,
    }


def write_experiment_reports(
    *,
    runs: list[dict[str, Any]],
    output_dir: Path,
    before_variant: str | None = None,
    after_variant: str | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = build_experiment_summary(
        runs,
        before_variant=before_variant,
        after_variant=after_variant,
    )

    summary_path = write_json(output_dir / "processing_experiment_summary.json", summary_payload)
    runs_csv_path = output_dir / "processing_runs_table.csv"
    stage_csv_path = output_dir / "processing_stage_timings_table.csv"
    reason_csv_path = output_dir / "processing_reason_frequencies_table.csv"

    _write_runs_csv(runs_csv_path, runs)
    _write_stage_timings_csv(stage_csv_path, runs)
    _write_reason_frequencies_csv(reason_csv_path, runs)

    return {
        "summary_json": summary_path,
        "runs_csv": runs_csv_path,
        "stage_timings_csv": stage_csv_path,
        "reason_frequencies_csv": reason_csv_path,
    }


def _group_by(
    runs: list[dict[str, Any]],
    key_fn,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in runs:
        key = str(key_fn(record) or "unknown").strip() or "unknown"
        grouped.setdefault(key, []).append(record)
    return grouped


def _compute_group_stats(group: list[dict[str, Any]]) -> dict[str, Any]:
    if not group:
        return {
            "run_count": 0,
            "completed_count": 0,
            "failed_count": 0,
            "success_rate": 0.0,
            "failure_rate": 0.0,
            "input_blocked_count": 0,
            "input_blocked_rate": 0.0,
            "fallback_count": 0,
            "fallback_rate": 0.0,
            "avg_total_processing_seconds": None,
            "median_total_processing_seconds": None,
            "avg_input_reduction_pct": 0.0,
            "top_failed_stage": None,
            "top_reason_code": None,
        }

    completed_count = sum(
        1 for record in group if _nested_str(record, "run_info", "status") == "completed"
    )
    failed_count = sum(
        1 for record in group if _nested_str(record, "run_info", "status") == "failed"
    )
    input_blocked_count = sum(
        1
        for record in group
        if bool(_nested_value(record, "quality_gates", "blocked_by_input_deficiency"))
    )
    fallback_count = sum(
        1 for record in group if bool(_nested_value(record, "execution_metrics", "fallback_used"))
    )

    duration_values = [
        float(value)
        for value in (
            _nested_value(record, "execution_metrics", "total_processing_seconds")
            for record in group
        )
        if isinstance(value, (int, float))
    ]
    reduction_values = [
        float(value)
        for value in (
            _nested_value(record, "input_metrics", "input_reduction_pct")
            for record in group
        )
        if isinstance(value, (int, float))
    ]
    failed_stage_counter = _counter(
        _nested_str(record, "execution_metrics", "failed_stage")
        for record in group
    )
    reason_code_counter = _counter(
        _nested_str(record, "execution_metrics", "reason_code")
        for record in group
    )

    run_count = len(group)
    return {
        "run_count": run_count,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "success_rate": round(completed_count / run_count, 4),
        "failure_rate": round(failed_count / run_count, 4),
        "input_blocked_count": input_blocked_count,
        "input_blocked_rate": round(input_blocked_count / run_count, 4),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / run_count, 4),
        "avg_total_processing_seconds": round(mean(duration_values), 3) if duration_values else None,
        "median_total_processing_seconds": round(median(duration_values), 3) if duration_values else None,
        "avg_input_reduction_pct": round(mean(reduction_values), 3) if reduction_values else 0.0,
        "top_failed_stage": _top_counter_key(failed_stage_counter),
        "top_reason_code": _top_counter_key(reason_code_counter),
    }


def _build_variant_delta(
    baseline_stats: dict[str, Any],
    improved_stats: dict[str, Any],
    baseline_name: str,
    improved_name: str,
) -> dict[str, Any]:
    def diff(metric_key: str, ndigits: int = 4) -> float | None:
        left = baseline_stats.get(metric_key)
        right = improved_stats.get(metric_key)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return None
        return round(float(right) - float(left), ndigits)

    return {
        "baseline_variant": baseline_name,
        "improved_variant": improved_name,
        "success_rate_delta": diff("success_rate"),
        "failure_rate_delta": diff("failure_rate"),
        "input_blocked_rate_delta": diff("input_blocked_rate"),
        "fallback_rate_delta": diff("fallback_rate"),
        "avg_total_processing_seconds_delta": diff("avg_total_processing_seconds", 3),
        "avg_input_reduction_pct_delta": diff("avg_input_reduction_pct", 3),
    }


def _write_runs_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "project_id",
                "generated_at",
                "status",
                "variant",
                "scenario_label",
                "total_images_loaded",
                "accepted_images",
                "rejected_images",
                "warning_images",
                "selected_images",
                "selection_discarded_images",
                "input_reduction_pct",
                "total_processing_seconds",
                "failed_stage",
                "reason_code",
                "fallback_used",
                "blocked_by_input_deficiency",
                "final_artifact_type",
                "final_artifact_size_bytes",
            ],
        )
        writer.writeheader()
        for record in runs:
            writer.writerow(
                {
                    "run_id": record.get("run_id"),
                    "project_id": _nested_str(record, "run_info", "project_id"),
                    "generated_at": record.get("generated_at"),
                    "status": _nested_str(record, "run_info", "status"),
                    "variant": _nested_str(record, "run_info", "variant"),
                    "scenario_label": _nested_str(record, "run_info", "scenario_label"),
                    "total_images_loaded": _nested_value(record, "input_metrics", "total_images_loaded"),
                    "accepted_images": _nested_value(record, "input_metrics", "accepted_images"),
                    "rejected_images": _nested_value(record, "input_metrics", "rejected_images"),
                    "warning_images": _nested_value(record, "input_metrics", "warning_images"),
                    "selected_images": _nested_value(record, "input_metrics", "selected_images"),
                    "selection_discarded_images": _nested_value(record, "input_metrics", "selection_discarded_images"),
                    "input_reduction_pct": _nested_value(record, "input_metrics", "input_reduction_pct"),
                    "total_processing_seconds": _nested_value(record, "execution_metrics", "total_processing_seconds"),
                    "failed_stage": _nested_str(record, "execution_metrics", "failed_stage"),
                    "reason_code": _nested_str(record, "execution_metrics", "reason_code"),
                    "fallback_used": _nested_value(record, "execution_metrics", "fallback_used"),
                    "blocked_by_input_deficiency": _nested_value(record, "quality_gates", "blocked_by_input_deficiency"),
                    "final_artifact_type": _nested_str(record, "artifact_metrics", "final_artifact_type"),
                    "final_artifact_size_bytes": _nested_value(record, "artifact_metrics", "final_artifact_size_bytes"),
                }
            )


def _write_stage_timings_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "project_id",
                "variant",
                "stage",
                "duration_seconds",
            ],
        )
        writer.writeheader()
        for record in runs:
            timings = _nested_value(record, "execution_metrics", "stage_timings_seconds")
            if not isinstance(timings, dict):
                continue
            for stage_name, seconds in sorted(timings.items(), key=lambda item: str(item[0])):
                if not isinstance(seconds, (int, float)):
                    continue
                writer.writerow(
                    {
                        "run_id": record.get("run_id"),
                        "project_id": _nested_str(record, "run_info", "project_id"),
                        "variant": _nested_str(record, "run_info", "variant"),
                        "stage": stage_name,
                        "duration_seconds": round(float(seconds), 3),
                    }
                )


def _write_reason_frequencies_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "project_id",
                "variant",
                "reason_type",
                "reason",
                "count",
            ],
        )
        writer.writeheader()
        for record in runs:
            frequencies = _nested_value(record, "reason_frequencies")
            if not isinstance(frequencies, dict):
                continue
            for reason_type, counter in frequencies.items():
                if not isinstance(counter, dict):
                    continue
                for reason, count in sorted(counter.items(), key=lambda item: str(item[0])):
                    if not isinstance(count, int):
                        continue
                    writer.writerow(
                        {
                            "run_id": record.get("run_id"),
                            "project_id": _nested_str(record, "run_info", "project_id"),
                            "variant": _nested_str(record, "run_info", "variant"),
                            "reason_type": reason_type,
                            "reason": reason,
                            "count": count,
                        }
                    )


def _nested_value(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for step in path:
        if not isinstance(current, dict):
            return None
        current = current.get(step)
    return current


def _nested_str(payload: dict[str, Any], *path: str) -> str | None:
    value = _nested_value(payload, *path)
    if value is None:
        return None
    return str(value).strip() or None


def _counter(values) -> dict[str, int]:
    counter: dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        counter[key] = counter.get(key, 0) + 1
    return counter


def _top_counter_key(counter: dict[str, int]) -> str | None:
    if not counter:
        return None
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
