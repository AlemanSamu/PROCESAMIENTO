import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import FileResponse

from app.core.dependencies import get_processing_service, get_project_service, require_api_key
from app.models.schemas import (
    ImageUploadResponse,
    ProcessRequest,
    ProcessStartResponse,
    ProjectCreateRequest,
    ProjectResponse,
    ProjectStatus,
    ProjectStatusResponse,
)
from app.services.processing_service import ProcessingService
from app.services.project_service import ProjectService
from config import get_settings

router = APIRouter(prefix="/projects", tags=["projects"], dependencies=[Depends(require_api_key)])
logger = logging.getLogger(__name__)


def _normalized_api_prefix() -> str:
    raw_prefix = str(get_settings().api_prefix or "").strip()
    if not raw_prefix:
        return ""
    if not raw_prefix.startswith("/"):
        raw_prefix = f"/{raw_prefix}"
    return raw_prefix.rstrip("/")


def _build_model_download_url(project_id: str) -> str:
    prefix = _normalized_api_prefix()
    if prefix:
        return f"{prefix}/projects/{project_id}/model"
    return f"/projects/{project_id}/model"


def _to_project_response(metadata) -> ProjectResponse:
    model_filename = metadata.model_filename if metadata.status == ProjectStatus.COMPLETED else None
    model_download_url = None
    if metadata.status == ProjectStatus.COMPLETED and metadata.model_filename:
        model_download_url = _build_model_download_url(metadata.id)

    status_details = _build_status_details(metadata)

    return ProjectResponse(
        id=metadata.id,
        name=metadata.name,
        status=metadata.status,
        created_at=metadata.created_at,
        updated_at=metadata.updated_at,
        image_count=metadata.image_count,
        output_format=metadata.output_format,
        model_filename=model_filename,
        model_download_url=model_download_url,
        error_message=metadata.error_message,
        current_stage=status_details["current_stage"],
        stage_status=status_details["stage_status"],
        fallback_used=status_details["fallback_used"],
        final_model_type=status_details["final_model_type"],
        final_model_path=status_details["final_model_path"],
        method_used=status_details["method_used"],
        processing_metadata=metadata.processing_metadata,
    )


def _build_status_details(metadata) -> dict[str, Any]:
    processing_metadata = metadata.processing_metadata or {}
    progress = processing_metadata.get("progress")
    if isinstance(progress, (int, float)):
        progress = max(0.0, min(1.0, float(progress)))
    elif metadata.status == ProjectStatus.COMPLETED:
        progress = 1.0
    elif metadata.status in {ProjectStatus.CREATED, ProjectStatus.READY}:
        progress = 0.0
    else:
        progress = None

    current_stage = processing_metadata.get("current_stage")
    if not current_stage and metadata.status == ProjectStatus.COMPLETED:
        current_stage = "completed"
    if not current_stage and metadata.status == ProjectStatus.FAILED:
        current_stage = "failed"

    stage_status = processing_metadata.get("stage_status")
    if not stage_status:
        if metadata.status == ProjectStatus.PROCESSING:
            stage_status = "running"
        elif metadata.status == ProjectStatus.COMPLETED:
            stage_status = "completed"
        elif metadata.status == ProjectStatus.FAILED:
            stage_status = "failed"
        else:
            stage_status = "idle"

    message = processing_metadata.get("status_message") or metadata.error_message
    if metadata.status == ProjectStatus.FAILED and metadata.error_message:
        normalized_message = str(message or "").strip().lower()
        if normalized_message in {
            "",
            "procesamiento fallido.",
            "processing failed.",
            "failed",
            "error",
        }:
            message = metadata.error_message
    if not message and metadata.status == ProjectStatus.COMPLETED:
        message = "Reconstruccion completada."
    if not message and metadata.status == ProjectStatus.PROCESSING:
        message = "Procesamiento en curso."

    metrics = processing_metadata.get("metrics")
    if not isinstance(metrics, dict):
        metrics = None

    artifacts = processing_metadata.get("artifacts") or {}
    sparse_fallback = processing_metadata.get("sparse_fallback") or {}
    fallback = processing_metadata.get("fallback") or {}
    final_model_path = (
        processing_metadata.get("final_model_path")
        or processing_metadata.get("output_path")
        or artifacts.get("model_path")
    )
    final_model_type = processing_metadata.get("final_model_type")
    if not final_model_type and final_model_path:
        suffix = str(final_model_path).rsplit('.', 1)
        final_model_type = suffix[-1].lower() if len(suffix) > 1 else None
    if metadata.status != ProjectStatus.COMPLETED:
        final_model_path = None
        final_model_type = None

    return {
        "engine": processing_metadata.get("engine") or processing_metadata.get("engine_requested"),
        "current_stage": current_stage,
        "stage_status": stage_status,
        "progress": progress,
        "message": message,
        "metrics": metrics,
        "fallback_used": bool(
            processing_metadata.get("fallback_used")
            or sparse_fallback.get("used")
            or fallback.get("used")
        ),
        "final_model_type": final_model_type,
        "final_model_path": final_model_path,
        "method_used": (
            processing_metadata.get("method_used")
            or sparse_fallback.get("mesh_method")
            or processing_metadata.get("meshing_method")
        ),
    }


def _build_result_payload(metadata) -> dict[str, Any]:
    status_details = _build_status_details(metadata)
    processing_metadata = metadata.processing_metadata or {}
    artifacts = processing_metadata.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    preprocessing_summary = processing_metadata.get("preprocessing")
    if not isinstance(preprocessing_summary, dict):
        preprocessing_summary = _load_json_artifact(artifacts.get("preprocessing_manifest"))
    fallback_report = processing_metadata.get("fallback_report")
    if not isinstance(fallback_report, dict):
        fallback_report = _load_json_artifact(artifacts.get("fallback_report"))
    quality_report = processing_metadata.get("quality_report")
    if not isinstance(quality_report, dict):
        quality_report = _load_json_artifact(artifacts.get("quality_report"))
    warnings = _collect_result_warnings(processing_metadata, preprocessing_summary, fallback_report, quality_report)
    model_download_url = None
    if metadata.status == ProjectStatus.COMPLETED and metadata.model_filename:
        model_download_url = _build_model_download_url(metadata.id)

    return {
        "project_id": metadata.id,
        "status": metadata.status,
        "engine": status_details["engine"],
        "current_stage": status_details["current_stage"],
        "workflow_stage": processing_metadata.get("workflow_stage"),
        "output_format": metadata.output_format,
        "model_download_url": model_download_url,
        "fallback_used": status_details["fallback_used"],
        "error_message": metadata.error_message,
        "metrics": status_details["metrics"],
        "preprocessing_summary": preprocessing_summary,
        "fallback_report": fallback_report,
        "quality_report": quality_report,
        "artifact_paths": artifacts,
        "warnings": warnings,
        "recommended_next_action": _recommended_next_action(metadata.status, status_details["fallback_used"], warnings),
        "artifacts": {
            **artifacts,
            "model_filename": metadata.model_filename,
            "final_model_path": status_details["final_model_path"],
            "final_model_type": status_details["final_model_type"],
        },
    }


def _load_json_artifact(raw_path: object) -> dict[str, Any] | None:
    if raw_path is None:
        return None
    try:
        path = Path(str(raw_path))
        if not path.exists() or not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _collect_result_warnings(
    processing_metadata: dict[str, Any],
    preprocessing_summary: dict[str, Any] | None,
    fallback_report: dict[str, Any] | None,
    quality_report: dict[str, Any] | None,
) -> list[str]:
    warnings: list[str] = []
    metadata_warnings = processing_metadata.get("warnings")
    if isinstance(metadata_warnings, list):
        warnings.extend(str(item) for item in metadata_warnings if item)
    if fallback_report:
        warnings.append("fallback_academico_usado")
    if preprocessing_summary:
        metrics = preprocessing_summary.get("metrics") if isinstance(preprocessing_summary.get("metrics"), dict) else {}
        warning_count = metrics.get("warning_images") or preprocessing_summary.get("warning_images")
        if isinstance(warning_count, int) and warning_count > 0:
            warnings.append(f"preprocesamiento_con_advertencias:{warning_count}")
    if quality_report:
        quality_classification = str(quality_report.get("quality_classification") or "").strip().lower()
        if quality_classification:
            warnings.append(f"quality_classification:{quality_classification}")
    return sorted(set(warnings))


def _recommended_next_action(status: ProjectStatus, fallback_used: bool, warnings: list[str]) -> str:
    if status == ProjectStatus.FAILED:
        return "Revisar logs y capturar mas imagenes con mejor overlap, nitidez e iluminacion."
    if fallback_used:
        return "Usar el modelo como evidencia minima y recapturar dataset para intentar COLMAP real."
    if warnings:
        return "Revisar advertencias de preprocesamiento antes de usar el resultado como evidencia final."
    if status == ProjectStatus.COMPLETED:
        return "Validar visualmente el modelo y anexar reportes JSON al informe."
    return "Esperar finalizacion del procesamiento."


@router.get("", response_model=list[ProjectResponse])
def list_projects(project_service: ProjectService = Depends(get_project_service)) -> list[ProjectResponse]:
    return [_to_project_response(item) for item in project_service.list_projects()]


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(
    payload: ProjectCreateRequest,
    project_service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    metadata = project_service.create_project(payload.name)
    return _to_project_response(metadata)


@router.post("/{project_id}/images", response_model=ImageUploadResponse)
def upload_images(
    project_id: str,
    files: list[UploadFile] = File(...),
    project_service: ProjectService = Depends(get_project_service),
) -> ImageUploadResponse:
    try:
        upload_result = project_service.add_images(project_id, files)
    finally:
        for file in files:
            file.file.close()

    return ImageUploadResponse(
        project_id=project_id,
        status=upload_result.metadata.status,
        uploaded_count=upload_result.uploaded_count,
        skipped_count=upload_result.skipped_count,
        total_images=upload_result.metadata.image_count,
        uploaded_files=upload_result.uploaded_files,
        message=upload_result.message,
    )


@router.post("/{project_id}/process", response_model=ProcessStartResponse, status_code=status.HTTP_202_ACCEPTED)
def start_processing(
    project_id: str,
    payload: ProcessRequest,
    processing_service: ProcessingService = Depends(get_processing_service),
) -> ProcessStartResponse:
    engine = processing_service.start_processing(project_id, payload.output_format)
    return ProcessStartResponse(
        project_id=project_id,
        status=ProjectStatus.PROCESSING,
        engine=engine,
        message="Procesamiento iniciado en segundo plano.",
    )


@router.get("/{project_id}/status", response_model=ProjectStatusResponse)
def get_project_status(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
) -> ProjectStatusResponse:
    metadata = project_service.get_project(project_id)
    model_filename = metadata.model_filename if metadata.status == ProjectStatus.COMPLETED else None
    model_download_url = None
    if metadata.status == ProjectStatus.COMPLETED and metadata.model_filename:
        model_download_url = _build_model_download_url(project_id)

    status_details = _build_status_details(metadata)

    return ProjectStatusResponse(
        project_id=metadata.id,
        status=metadata.status,
        image_count=metadata.image_count,
        output_format=metadata.output_format,
        model_filename=model_filename,
        model_download_url=model_download_url,
        error_message=metadata.error_message,
        engine=status_details["engine"],
        current_stage=status_details["current_stage"],
        stage_status=status_details["stage_status"],
        progress=status_details["progress"],
        message=status_details["message"],
        metrics=status_details["metrics"],
        fallback_used=status_details["fallback_used"],
        final_model_type=status_details["final_model_type"],
        final_model_path=status_details["final_model_path"],
        method_used=status_details["method_used"],
        processing_metadata=metadata.processing_metadata,
    )


@router.get("/{project_id}/result")
def get_project_result(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
) -> dict[str, Any]:
    metadata = project_service.get_project(project_id)
    return _build_result_payload(metadata)


@router.get("/{project_id}/model")
def download_model(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
) -> FileResponse:
    metadata = project_service.get_project(project_id)
    model_path = project_service.get_model_file(project_id)
    processing_metadata = metadata.processing_metadata or {}
    logger.info(
        "Serving /projects/%s/model artifact=%s current_stage=%s method_used=%s fallback_used=%s",
        project_id,
        model_path,
        processing_metadata.get("current_stage"),
        processing_metadata.get("method_used") or (processing_metadata.get("sparse_fallback") or {}).get("mesh_method"),
        processing_metadata.get("fallback_used"),
    )
    media_type = "model/gltf-binary" if model_path.suffix.lower() == ".glb" else "text/plain"
    return FileResponse(path=model_path, filename=model_path.name, media_type=media_type)
