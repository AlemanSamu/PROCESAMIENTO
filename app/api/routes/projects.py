import logging
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

router = APIRouter(prefix="/projects", tags=["projects"], dependencies=[Depends(require_api_key)])
logger = logging.getLogger(__name__)


def _to_project_response(metadata) -> ProjectResponse:
    model_download_url = None
    if metadata.status == ProjectStatus.COMPLETED and metadata.model_filename:
        model_download_url = f"/projects/{metadata.id}/model"

    status_details = _build_status_details(metadata)

    return ProjectResponse(
        id=metadata.id,
        name=metadata.name,
        status=metadata.status,
        created_at=metadata.created_at,
        updated_at=metadata.updated_at,
        image_count=metadata.image_count,
        output_format=metadata.output_format,
        model_filename=metadata.model_filename,
        model_download_url=model_download_url,
        error_message=metadata.error_message,
        current_stage=status_details["current_stage"],
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

    message = processing_metadata.get("status_message") or metadata.error_message
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

    return {
        "engine": processing_metadata.get("engine") or processing_metadata.get("engine_requested"),
        "current_stage": current_stage,
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
    model_download_url = None
    if metadata.status == ProjectStatus.COMPLETED and metadata.model_filename:
        model_download_url = f"/projects/{project_id}/model"

    status_details = _build_status_details(metadata)

    return ProjectStatusResponse(
        project_id=metadata.id,
        status=metadata.status,
        image_count=metadata.image_count,
        output_format=metadata.output_format,
        model_filename=metadata.model_filename,
        model_download_url=model_download_url,
        error_message=metadata.error_message,
        engine=status_details["engine"],
        current_stage=status_details["current_stage"],
        progress=status_details["progress"],
        message=status_details["message"],
        metrics=status_details["metrics"],
        fallback_used=status_details["fallback_used"],
        final_model_type=status_details["final_model_type"],
        final_model_path=status_details["final_model_path"],
        method_used=status_details["method_used"],
        processing_metadata=metadata.processing_metadata,
    )


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
