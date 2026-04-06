from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import FileResponse

from app.core.dependencies import get_processing_service, get_project_service
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

router = APIRouter(prefix="/projects", tags=["projects"])


def _to_project_response(metadata) -> ProjectResponse:
    model_download_url = None
    if metadata.status == ProjectStatus.COMPLETED and metadata.model_filename:
        model_download_url = f"/projects/{metadata.id}/model"

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
        processing_metadata=metadata.processing_metadata,
    )


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
        metadata, saved_files = project_service.add_images(project_id, files)
    finally:
        for file in files:
            file.file.close()

    return ImageUploadResponse(
        project_id=project_id,
        status=metadata.status,
        uploaded_count=len(saved_files),
        total_images=metadata.image_count,
        uploaded_files=saved_files,
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

    return ProjectStatusResponse(
        project_id=metadata.id,
        status=metadata.status,
        image_count=metadata.image_count,
        output_format=metadata.output_format,
        model_filename=metadata.model_filename,
        model_download_url=model_download_url,
        error_message=metadata.error_message,
        processing_metadata=metadata.processing_metadata,
    )


@router.get("/{project_id}/model")
def download_model(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
) -> FileResponse:
    model_path = project_service.get_model_file(project_id)
    media_type = "model/gltf-binary" if model_path.suffix.lower() == ".glb" else "text/plain"
    return FileResponse(path=model_path, filename=model_path.name, media_type=media_type)