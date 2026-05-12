from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import UploadFile

from app.models.schemas import OutputFormat, ProjectStatus
from app.services.processing_service import ProcessingService
from app.services.project_service import ProjectService
from app.services.storage_service import StorageService
from config import get_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta una prueba real end-to-end desde una carpeta de imagenes "
            "y genera un resumen JSON para evidencia."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Carpeta local con imagenes de entrada (ej: CAJA_PASTILLAS).",
    )
    parser.add_argument(
        "--output-format",
        choices=["glb", "obj"],
        default="glb",
        help="Formato de salida solicitado al pipeline.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="Prueba carpeta local",
        help="Nombre del proyecto temporal de prueba.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limite de imagenes a usar (0 = todas).",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="Intervalo de consulta de estado en segundos.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=20 * 60,
        help="Timeout maximo de espera de procesamiento.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("tmp_folder_trial_report.json"),
        help="Ruta del JSON resumen de la corrida.",
    )
    return parser


def _collect_images(images_dir: Path, allowed_extensions: tuple[str, ...], max_images: int) -> list[Path]:
    normalized_allowed = {ext.lower() for ext in allowed_extensions}
    images = sorted(
        [path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in normalized_allowed],
        key=lambda item: item.name.lower(),
    )
    if max_images > 0:
        images = images[:max_images]
    return images


def _build_summary(
    *,
    project_id: str,
    status: ProjectStatus,
    model_filename: str | None,
    error_message: str | None,
    processing_metadata: dict,
) -> dict:
    segmentation = processing_metadata.get("input_object_segmentation")
    if not isinstance(segmentation, dict):
        segmentation = {}

    captured_texture = (
        (processing_metadata.get("approximate_geometry_fallback") or {}).get("captured_texture")
        if isinstance(processing_metadata.get("approximate_geometry_fallback"), dict)
        else None
    )
    if not isinstance(captured_texture, dict):
        captured_texture = None

    return {
        "project_id": project_id,
        "status": str(status),
        "model_filename": model_filename,
        "error_message": error_message,
        "current_stage": processing_metadata.get("current_stage"),
        "stage_status": processing_metadata.get("stage_status"),
        "reason_code": processing_metadata.get("reason_code"),
        "method_used": processing_metadata.get("method_used"),
        "fallback_used": processing_metadata.get("fallback_used"),
        "final_model_path": processing_metadata.get("final_model_path"),
        "metrics": processing_metadata.get("metrics"),
        "input_object_segmentation": {
            "segmented_images": segmentation.get("segmented_images"),
            "fallback_original_images": segmentation.get("fallback_original_images"),
            "segmentation_ratio": segmentation.get("segmentation_ratio"),
            "policy_decision": segmentation.get("policy_decision"),
        },
        "captured_texture": captured_texture,
        "artifacts": processing_metadata.get("artifacts"),
    }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    settings = get_settings()
    images_dir = args.images_dir.resolve()
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"[error] Carpeta invalida: {images_dir}", file=sys.stderr)
        return 2

    image_paths = _collect_images(images_dir, settings.allowed_image_extensions, max(0, int(args.max_images)))
    if not image_paths:
        print(f"[error] No se encontraron imagenes validas en: {images_dir}", file=sys.stderr)
        return 2

    storage = StorageService(settings)
    project_service = ProjectService(storage, settings)
    processing_service = ProcessingService(project_service, storage, settings)

    project = project_service.create_project(args.project_name)
    project_id = project.id

    uploads = [UploadFile(filename=path.name, file=path.open("rb")) for path in image_paths]
    try:
        upload_result = project_service.add_images(project_id, uploads)
    finally:
        for upload in uploads:
            try:
                upload.file.close()
            except Exception:
                pass

    print(f"[trial] project_id={project_id}")
    print(f"[trial] images_dir={images_dir}")
    print(f"[trial] uploaded={upload_result.uploaded_count} skipped={upload_result.skipped_count}")

    requested_output_format = OutputFormat(str(args.output_format).lower())
    engine_name = processing_service.start_processing(project_id, requested_output_format)
    print(f"[trial] engine={engine_name} output_format={requested_output_format.value}")

    started_at = time.perf_counter()
    timeout_seconds = max(60, int(args.timeout_seconds))
    poll_seconds = max(0.5, float(args.poll_seconds))
    last_status: ProjectStatus | None = None

    while True:
        metadata = project_service.get_project(project_id)
        if metadata.status != last_status:
            print(f"[trial] status={metadata.status}")
            last_status = metadata.status

        if metadata.status in {ProjectStatus.COMPLETED, ProjectStatus.FAILED}:
            break

        if (time.perf_counter() - started_at) > timeout_seconds:
            print("[error] Timeout esperando fin de procesamiento.", file=sys.stderr)
            return 3
        time.sleep(poll_seconds)

    metadata = project_service.get_project(project_id)
    processing_metadata = dict(metadata.processing_metadata or {})
    summary = _build_summary(
        project_id=project_id,
        status=metadata.status,
        model_filename=metadata.model_filename,
        error_message=metadata.error_message,
        processing_metadata=processing_metadata,
    )

    summary_out = args.summary_out.resolve()
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[trial] summary={summary_out}")
    print(f"[trial] current_stage={summary.get('current_stage')} method={summary.get('method_used')}")
    print(f"[trial] final_model_path={summary.get('final_model_path')}")

    return 0 if metadata.status == ProjectStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
