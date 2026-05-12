from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.schemas import OutputFormat, ProjectMetadata, ProjectStatus  # noqa: E402
from app.services.processing_service import ProcessingService  # noqa: E402
from app.services.project_service import ProjectService  # noqa: E402
from app.services.storage_service import StorageService  # noqa: E402
from config import Settings, _apply_profile_defaults  # noqa: E402


ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Corre experimento comparativo por perfiles de LOCAL3D.")
    parser.add_argument("--input", required=True, type=Path, help="Carpeta con imagenes de entrada.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["conservative", "balanced", "quality"],
        help="Perfiles a comparar. Ejemplo: conservative balanced quality",
    )
    parser.add_argument("--output-format", choices=["glb", "obj"], default="glb")
    parser.add_argument("--reports-dir", type=Path, default=Path("data/experiments/reports"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/experiments/profile_runs"))
    return parser


def _collect_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError(f"La carpeta de entrada no existe: {input_dir}")
    images = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"No se encontraron imagenes soportadas en: {input_dir}")
    return images


def _build_profile_settings(profile: str, storage_root: Path) -> Settings:
    settings = Settings()
    settings.profile = str(profile).strip().lower()
    settings.storage_root = storage_root
    settings.processing_engine = "colmap"
    settings.colmap_fallback_to_mock = False
    settings.metrics_evidence_enabled = False
    settings.force_presentable_model_enabled = False
    settings = _apply_profile_defaults(settings)
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    return settings


def _prepare_project_with_images(
    *,
    project_service: ProjectService,
    storage_service: StorageService,
    project_id: str,
    profile: str,
    images: list[Path],
) -> None:
    images_dir = storage_service.get_images_dir(project_id)
    images_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    for index, source in enumerate(images, start=1):
        destination_name = f"{index:03d}_{source.name}"
        destination = images_dir / destination_name
        shutil.copy2(source, destination)
        copied_files.append(destination_name)

    now = datetime.now(timezone.utc)
    metadata = ProjectMetadata(
        id=project_id,
        name=f"Experiment-{profile}-{project_id}",
        status=ProjectStatus.READY,
        created_at=now,
        updated_at=now,
        image_count=len(copied_files),
        image_files=copied_files,
    )
    storage_service.save_project_metadata(metadata)


def _extract_run_row(
    *,
    profile: str,
    project_id: str,
    metadata: ProjectMetadata,
    output_dir: Path,
) -> dict[str, Any]:
    processing_metadata = metadata.processing_metadata or {}
    metrics = processing_metadata.get("metrics") if isinstance(processing_metadata.get("metrics"), dict) else {}
    quality_report = processing_metadata.get("quality_report") if isinstance(processing_metadata.get("quality_report"), dict) else {}
    quality_metrics = quality_report.get("metrics") if isinstance(quality_report.get("metrics"), dict) else {}

    final_model_path = processing_metadata.get("final_model_path") or (processing_metadata.get("artifacts") or {}).get("model_path")
    model_size_bytes = 0
    if final_model_path:
        try:
            model_size_bytes = Path(str(final_model_path)).stat().st_size
        except OSError:
            model_size_bytes = 0

    return {
        "profile": profile,
        "project_id": project_id,
        "status": metadata.status.value,
        "quality_classification": str(
            quality_report.get("quality_classification")
            or processing_metadata.get("quality_classification")
            or "failed"
        ),
        "total_time_seconds": float(metrics.get("total_processing_seconds") or 0.0),
        "images_accepted": int(
            metrics.get("image_count_accepted")
            or quality_metrics.get("image_count_processed")
            or metrics.get("image_count_selected")
            or 0
        ),
        "cameras_reconstructed": int(
            quality_metrics.get("cameras_reconstructed")
            or processing_metadata.get("registered_image_count")
            or metrics.get("reconstructed_camera_count")
            or 0
        ),
        "points_3d_count": int(
            quality_metrics.get("points_3d_count")
            or metrics.get("point_3d_count")
            or metrics.get("sparse_point_cloud_count")
            or processing_metadata.get("point_count")
            or 0
        ),
        "fallback_used": bool(
            processing_metadata.get("fallback_used")
            or (processing_metadata.get("fallback") or {}).get("used")
            or (processing_metadata.get("sparse_fallback") or {}).get("used")
        ),
        "model_size_bytes": int(quality_metrics.get("model_size_bytes") or model_size_bytes),
        "output_dir": str(output_dir),
    }


def write_profile_comparison_reports(
    *,
    runs: list[dict[str, Any]],
    reports_dir: Path,
    input_dir: Path,
    output_format: str,
) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at_utc": generated_at,
        "input_dir": str(input_dir),
        "output_format": output_format,
        "profiles": [str(item.get("profile")) for item in runs],
        "runs": runs,
    }

    json_path = reports_dir / "profile_comparison.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = reports_dir / "profile_comparison.csv"
    fieldnames = [
        "profile",
        "project_id",
        "status",
        "quality_classification",
        "total_time_seconds",
        "images_accepted",
        "cameras_reconstructed",
        "points_3d_count",
        "fallback_used",
        "model_size_bytes",
        "output_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow({key: run.get(key) for key in fieldnames})

    return {"json": json_path, "csv": csv_path}


def run_experiment(
    *,
    input_dir: Path,
    profiles: list[str],
    output_format: OutputFormat,
    reports_dir: Path,
    work_dir: Path,
) -> dict[str, Path]:
    images = _collect_images(input_dir)
    runs: list[dict[str, Any]] = []

    for raw_profile in profiles:
        profile = str(raw_profile).strip().lower()
        if profile not in {"conservative", "balanced", "quality"}:
            raise RuntimeError(f"Perfil no soportado: {raw_profile}")

        project_id = f"{profile}-{uuid.uuid4().hex[:8]}"
        profile_storage_root = work_dir / profile / "projects"
        settings = _build_profile_settings(profile, profile_storage_root)
        storage_service = StorageService(settings)
        project_service = ProjectService(storage_service, settings)
        processing_service = ProcessingService(project_service, storage_service, settings)

        _prepare_project_with_images(
            project_service=project_service,
            storage_service=storage_service,
            project_id=project_id,
            profile=profile,
            images=images,
        )
        project_service.mark_processing(project_id, output_format, processing_metadata={"current_stage": "queued"})

        started_at = time.perf_counter()
        processing_service._run_reconstruction_job(project_id, output_format)
        elapsed_seconds = round(time.perf_counter() - started_at, 3)

        metadata = project_service.get_project(project_id)
        output_dir = storage_service.get_output_dir(project_id)
        row = _extract_run_row(
            profile=profile,
            project_id=project_id,
            metadata=metadata,
            output_dir=output_dir,
        )
        if not row.get("total_time_seconds"):
            row["total_time_seconds"] = elapsed_seconds
        runs.append(row)

    return write_profile_comparison_reports(
        runs=runs,
        reports_dir=reports_dir,
        input_dir=input_dir,
        output_format=output_format.value,
    )


def main() -> int:
    args = _build_parser().parse_args()
    outputs = run_experiment(
        input_dir=args.input.resolve(),
        profiles=[str(item) for item in args.profiles],
        output_format=OutputFormat(args.output_format),
        reports_dir=args.reports_dir,
        work_dir=args.work_dir,
    )
    summary = {
        "profile_comparison_json": str(outputs["json"]),
        "profile_comparison_csv": str(outputs["csv"]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
