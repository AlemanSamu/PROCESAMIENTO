import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Iterable

from fastapi import UploadFile

from app.core.errors import BadRequestError, InvalidProjectStateError, ProjectNotFoundError
from app.models.schemas import OutputFormat, ProjectMetadata, ProjectStatus
from app.services.storage_service import StorageService

logger = logging.getLogger(__name__)


@dataclass
class ProjectImageUploadResult:
    metadata: ProjectMetadata
    uploaded_files: list[str]
    skipped_count: int
    message: str
    project_created: bool
    reset_processing_state: bool

    @property
    def uploaded_count(self) -> int:
        return len(self.uploaded_files)


class ProjectService:
    _GENERIC_PROJECT_NAMES = {
        "caja",
        "box",
        "objeto",
        "object",
        "modelo",
        "model",
        "proyecto",
        "untitled",
        "sin nombre",
        "proyecto sin nombre",
    }
    _GENERIC_NAME_PATTERN = re.compile(r"^proyecto-[a-f0-9]{8,}$")

    def __init__(self, storage_service: StorageService, settings) -> None:
        self.storage_service = storage_service
        self.settings = settings

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def create_project(self, name: str | None) -> ProjectMetadata:
        project_id = uuid.uuid4().hex[:12]
        return self._create_project_with_id(project_id, name)

    def list_projects(self) -> list[ProjectMetadata]:
        return self.storage_service.list_project_metadata()

    def get_project(self, project_id: str) -> ProjectMetadata:
        return self.storage_service.load_project_metadata(project_id)

    def add_images(self, project_id: str, files: Iterable[UploadFile]) -> ProjectImageUploadResult:
        incoming_files = list(files)
        if not incoming_files:
            raise BadRequestError("Debes enviar al menos una imagen.")

        metadata, project_created = self._get_or_create_project(project_id)
        if metadata.status == ProjectStatus.PROCESSING:
            raise InvalidProjectStateError("No puedes subir imagenes mientras el proyecto esta procesando.")

        reset_processing_state = self._has_previous_processing_state(metadata, project_id)
        save_result = self.storage_service.save_images(
            project_id,
            incoming_files,
            max_total_images=self.settings.max_images_per_project,
        )
        self.storage_service.clear_processing_artifacts(project_id)

        metadata.image_files = self.storage_service.list_image_files(project_id)
        metadata.image_count = len(metadata.image_files)
        metadata.status = ProjectStatus.READY if metadata.image_count > 0 else ProjectStatus.CREATED
        metadata.output_format = None
        metadata.model_filename = None
        metadata.error_message = None
        metadata.processing_metadata = (
            {
                "current_stage": "uploading",
                "workflow_stage": "uploading",
                "stage_status": "completed",
                "progress": 0.0,
                "status_message": "Imagenes cargadas y proyecto listo para procesamiento.",
                "metrics": {
                    "uploaded_images": save_result.uploaded_count,
                    "skipped_duplicate_images": save_result.skipped_count,
                    "total_images": metadata.image_count,
                },
                "artifacts": {
                    "images_dir": str(self.storage_service.get_images_dir(project_id)),
                },
            }
            if save_result.uploaded_count > 0
            else None
        )
        metadata.updated_at = self._utc_now()

        self.storage_service.save_project_metadata(metadata)
        return ProjectImageUploadResult(
            metadata=metadata,
            uploaded_files=save_result.saved_files,
            skipped_count=save_result.skipped_count,
            message=self._build_upload_message(
                uploaded_count=save_result.uploaded_count,
                skipped_count=save_result.skipped_count,
                project_created=project_created,
                reset_processing_state=reset_processing_state,
            ),
            project_created=project_created,
            reset_processing_state=reset_processing_state,
        )

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
        metadata.model_filename = None
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
        metadata.name = self._resolve_completed_project_name(
            current_name=metadata.name,
            project_id=metadata.id,
            processing_metadata=processing_metadata,
        )
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
        metadata.model_filename = None
        metadata.error_message = reason
        metadata.processing_metadata = processing_metadata
        metadata.updated_at = self._utc_now()
        self.storage_service.save_project_metadata(metadata)
        return metadata

    def get_model_file(self, project_id: str):
        metadata = self.get_project(project_id)
        if metadata.status != ProjectStatus.COMPLETED or not metadata.model_filename:
            raise InvalidProjectStateError("El modelo aun no esta disponible para descarga.")

        model_path = self.storage_service.get_model_path(project_id, metadata.model_filename)
        processing_metadata = metadata.processing_metadata or {}
        final_model_path = (
            processing_metadata.get("final_model_path")
            or processing_metadata.get("output_path")
            or (processing_metadata.get("artifacts") or {}).get("model_path")
        )
        expected_extension = f".{metadata.output_format.value}" if metadata.output_format is not None else None

        if expected_extension and model_path.suffix.lower() != expected_extension.lower():
            raise InvalidProjectStateError(
                "El proyecto figura como completado, pero el artefacto final no coincide con el formato esperado. "
                f"Esperado: {expected_extension}. Encontrado: {model_path.suffix.lower() or 'sin extension'}."
            )

        if final_model_path:
            final_model_name = Path(str(final_model_path)).name
            if final_model_name and final_model_name != model_path.name:
                raise InvalidProjectStateError(
                    "El proyecto figura como completado, pero el archivo final registrado no coincide con el artefacto servido. "
                    f"Registrado: {final_model_name}. Servido: {model_path.name}."
                )

        logger.info(
            "Serving model artifact for project %s. current_stage=%s fallback_used=%s model_path=%s",
            project_id,
            processing_metadata.get("current_stage"),
            processing_metadata.get("fallback_used"),
            model_path,
        )
        return model_path

    def _create_project_with_id(self, project_id: str, name: str | None) -> ProjectMetadata:
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

    def _get_or_create_project(self, project_id: str) -> tuple[ProjectMetadata, bool]:
        try:
            return self.get_project(project_id), False
        except ProjectNotFoundError:
            return self._create_project_with_id(project_id, None), True

    def _has_previous_processing_state(self, metadata: ProjectMetadata, project_id: str) -> bool:
        if metadata.status in {ProjectStatus.COMPLETED, ProjectStatus.FAILED}:
            return True
        if metadata.output_format is not None or metadata.model_filename or metadata.error_message:
            return True
        if metadata.processing_metadata:
            return True

        output_dir = self.storage_service.get_output_dir(project_id)
        return output_dir.exists() and any(output_dir.iterdir())

    def _resolve_completed_project_name(
        self,
        *,
        current_name: str,
        project_id: str,
        processing_metadata: dict[str, Any] | None,
    ) -> str:
        if not self._is_generic_project_name(current_name, project_id):
            return current_name

        inferred_name = self._infer_name_from_processing_metadata(processing_metadata)
        if inferred_name is None:
            return current_name

        logger.info(
            "Project %s renamed from generic label '%s' to '%s' using processing metadata.",
            project_id,
            current_name,
            inferred_name,
        )
        return inferred_name

    @classmethod
    def _is_generic_project_name(cls, name: str, project_id: str) -> bool:
        normalized = cls._normalize_name(name)
        if not normalized:
            return True

        if normalized in cls._GENERIC_PROJECT_NAMES:
            return True

        if normalized == f"proyecto-{project_id}".lower():
            return True

        if cls._GENERIC_NAME_PATTERN.fullmatch(normalized):
            return True

        return False

    @classmethod
    def _infer_name_from_processing_metadata(
        cls,
        processing_metadata: dict[str, Any] | None,
    ) -> str | None:
        if not isinstance(processing_metadata, dict):
            return None

        direct_candidates = [
            processing_metadata.get("object_display_name"),
            processing_metadata.get("detected_object_name"),
            processing_metadata.get("object_name"),
        ]
        for raw_candidate in direct_candidates:
            candidate = cls._normalize_inferred_candidate(raw_candidate)
            if candidate is not None:
                return candidate

        forced = processing_metadata.get("forced_presentable_model")
        if not isinstance(forced, dict):
            return None

        forced_candidates = [
            forced.get("display_name"),
            forced.get("source_glb"),
            forced.get("source_obj"),
        ]
        for raw_candidate in forced_candidates:
            candidate = cls._normalize_inferred_candidate(raw_candidate)
            if candidate is not None:
                return candidate
        return None

    @classmethod
    def _normalize_inferred_candidate(cls, raw_candidate: object) -> str | None:
        text = str(raw_candidate or "").strip()
        if not text:
            return None

        stem = Path(text).stem if "/" in text or "\\" in text or "." in text else text
        normalized = re.sub(r"[_-]+", " ", stem)
        normalized = re.sub(
            r"\b(canonical|canonico|modelo?|mesh|reconstruido|reconstruction|output|final)\b",
            " ",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return None

        candidate = " ".join(word.capitalize() for word in normalized.split(" "))
        if cls._is_generic_project_name(candidate, project_id=""):
            return None
        return candidate

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(str(name or "").strip().lower().split())

    @staticmethod
    def _build_upload_message(
        *,
        uploaded_count: int,
        skipped_count: int,
        project_created: bool,
        reset_processing_state: bool,
    ) -> str:
        action = "Proyecto creado" if project_created else "Proyecto actualizado"
        details = (
            f"{uploaded_count} imagenes agregadas, "
            f"{skipped_count} omitidas por duplicadas."
        )
        if reset_processing_state:
            return f"{action}: {details} Se limpiaron artefactos previos y el proyecto quedo listo para reprocesarse."
        return f"{action}: {details} El proyecto quedo listo para procesarse."

