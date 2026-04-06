import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from fastapi import UploadFile

from app.core.errors import BadRequestError, InvalidProjectStateError
from app.models.schemas import OutputFormat, ProjectMetadata, ProjectStatus
from app.services.storage_service import StorageService


class ProjectService:
    def __init__(self, storage_service: StorageService, settings) -> None:
        self.storage_service = storage_service
        self.settings = settings

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def create_project(self, name: str | None) -> ProjectMetadata:
        project_id = uuid.uuid4().hex[:12]
        normalized_name = (name or "").strip() or f"Proyecto-{project_id}"
        now = self._utc_now()

        metadata = ProjectMetadata(
            id=project_id,
            name=normalized_name,
            status=ProjectStatus.CREATED,
            created_at=now,
            updated_at=now,
        )
        self.storage_service.save_project_metadata(metadata)
        return metadata

    def list_projects(self) -> list[ProjectMetadata]:
        return self.storage_service.list_project_metadata()

    def get_project(self, project_id: str) -> ProjectMetadata:
        return self.storage_service.load_project_metadata(project_id)

    def add_images(self, project_id: str, files: Iterable[UploadFile]) -> tuple[ProjectMetadata, list[str]]:
        metadata = self.get_project(project_id)
        incoming_files = list(files)

        if not incoming_files:
            raise BadRequestError("Debes enviar al menos una imagen.")

        if metadata.status == ProjectStatus.PROCESSING:
            raise InvalidProjectStateError("No puedes subir imagenes mientras el proyecto esta procesando.")

        if metadata.image_count + len(incoming_files) > self.settings.max_images_per_project:
            raise BadRequestError(
                f"Se excede el maximo de imagenes por proyecto ({self.settings.max_images_per_project})."
            )

        saved_files = self.storage_service.save_images(project_id, incoming_files)
        self.storage_service.clear_output_files(project_id)

        metadata.image_files.extend(saved_files)
        metadata.image_count = len(metadata.image_files)
        metadata.status = ProjectStatus.READY
        metadata.output_format = None
        metadata.model_filename = None
        metadata.error_message = None
        metadata.processing_metadata = None
        metadata.updated_at = self._utc_now()

        self.storage_service.save_project_metadata(metadata)
        return metadata, saved_files

    def mark_processing(
        self,
        project_id: str,
        output_format: OutputFormat,
        processing_metadata: dict[str, Any] | None = None,
    ) -> ProjectMetadata:
        metadata = self.get_project(project_id)
        if metadata.status == ProjectStatus.PROCESSING:
            raise InvalidProjectStateError("El proyecto ya se encuentra en procesamiento.")

        if metadata.image_count == 0:
            raise InvalidProjectStateError("Primero debes cargar imagenes antes de procesar.")

        metadata.status = ProjectStatus.PROCESSING
        metadata.output_format = output_format
        metadata.error_message = None
        metadata.processing_metadata = processing_metadata
        metadata.updated_at = self._utc_now()

        self.storage_service.save_project_metadata(metadata)
        return metadata

    def update_processing_metadata(
        self,
        project_id: str,
        processing_metadata: dict[str, Any] | None,
    ) -> ProjectMetadata:
        metadata = self.get_project(project_id)
        if metadata.status != ProjectStatus.PROCESSING:
            return metadata

        metadata.processing_metadata = processing_metadata
        metadata.updated_at = self._utc_now()
        self.storage_service.save_project_metadata(metadata)
        return metadata

    def mark_completed(
        self,
        project_id: str,
        output_format: OutputFormat,
        model_filename: str,
        processing_metadata: dict[str, Any] | None = None,
    ) -> ProjectMetadata:
        metadata = self.get_project(project_id)
        metadata.status = ProjectStatus.COMPLETED
        metadata.output_format = output_format
        metadata.model_filename = model_filename
        metadata.error_message = None
        metadata.processing_metadata = processing_metadata
        metadata.updated_at = self._utc_now()
        self.storage_service.save_project_metadata(metadata)
        return metadata

    def mark_failed(
        self,
        project_id: str,
        reason: str,
        processing_metadata: dict[str, Any] | None = None,
    ) -> ProjectMetadata:
        metadata = self.get_project(project_id)
        metadata.status = ProjectStatus.FAILED
        metadata.error_message = reason
        metadata.processing_metadata = processing_metadata
        metadata.updated_at = self._utc_now()
        self.storage_service.save_project_metadata(metadata)
        return metadata

    def get_model_file(self, project_id: str):
        metadata = self.get_project(project_id)
        if metadata.status != ProjectStatus.COMPLETED or not metadata.model_filename:
            raise InvalidProjectStateError("El modelo aun no esta disponible para descarga.")
        return self.storage_service.get_model_path(project_id, metadata.model_filename)