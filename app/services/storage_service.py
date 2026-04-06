import shutil
import threading
import uuid
from pathlib import Path
from typing import Iterable

from fastapi import UploadFile

from app.core.errors import BadRequestError, ProjectNotFoundError, StorageError
from app.models.schemas import ProjectMetadata


class StorageService:
    METADATA_FILE = "meta.json"

    def __init__(self, settings) -> None:
        self.settings = settings
        self.root = settings.storage_root
        self._lock = threading.Lock()
        self._allowed_extensions = {ext.lower() for ext in settings.allowed_image_extensions}
        self.root.mkdir(parents=True, exist_ok=True)

    def get_project_dir(self, project_id: str) -> Path:
        return self.root / project_id

    def get_images_dir(self, project_id: str) -> Path:
        return self.get_project_dir(project_id) / "images"

    def get_output_dir(self, project_id: str) -> Path:
        return self.get_project_dir(project_id) / "output"

    def get_metadata_path(self, project_id: str) -> Path:
        return self.get_project_dir(project_id) / self.METADATA_FILE

    def ensure_project_structure(self, project_id: str) -> None:
        self.get_images_dir(project_id).mkdir(parents=True, exist_ok=True)
        self.get_output_dir(project_id).mkdir(parents=True, exist_ok=True)

    def save_project_metadata(self, metadata: ProjectMetadata) -> None:
        try:
            self.ensure_project_structure(metadata.id)
            with self._lock:
                self.get_metadata_path(metadata.id).write_text(
                    metadata.model_dump_json(indent=2),
                    encoding="utf-8",
                )
        except Exception as exc:
            raise StorageError(f"No se pudo guardar metadata del proyecto {metadata.id}.") from exc

    def load_project_metadata(self, project_id: str) -> ProjectMetadata:
        metadata_path = self.get_metadata_path(project_id)
        if not metadata_path.exists():
            raise ProjectNotFoundError(f"Proyecto '{project_id}' no encontrado.")

        try:
            raw = metadata_path.read_text(encoding="utf-8")
            return ProjectMetadata.model_validate_json(raw)
        except ProjectNotFoundError:
            raise
        except Exception as exc:
            raise StorageError(f"No se pudo leer metadata del proyecto {project_id}.") from exc

    def list_project_metadata(self) -> list[ProjectMetadata]:
        projects: list[ProjectMetadata] = []
        if not self.root.exists():
            return projects

        for project_dir in self.root.iterdir():
            if not project_dir.is_dir():
                continue

            metadata_path = project_dir / self.METADATA_FILE
            if not metadata_path.exists():
                continue

            try:
                projects.append(ProjectMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8")))
            except Exception:
                continue

        return sorted(projects, key=lambda item: item.created_at, reverse=True)

    def save_images(self, project_id: str, files: Iterable[UploadFile]) -> list[str]:
        images_dir = self.get_images_dir(project_id)
        if not images_dir.exists():
            raise ProjectNotFoundError(f"Proyecto '{project_id}' no encontrado.")

        saved_files: list[str] = []
        for file in files:
            original_name = (file.filename or "").strip()
            if not original_name:
                raise BadRequestError("Todos los archivos deben tener nombre.")

            extension = Path(original_name).suffix.lower()
            if extension not in self._allowed_extensions:
                raise BadRequestError(
                    f"Formato de imagen no soportado: '{extension}'. "
                    f"Permitidos: {sorted(self._allowed_extensions)}"
                )

            generated_name = f"{uuid.uuid4().hex}{extension}"
            destination = images_dir / generated_name

            try:
                with destination.open("wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            except Exception as exc:
                raise StorageError(f"No se pudo guardar imagen '{original_name}'.") from exc

            saved_files.append(generated_name)

        return saved_files

    def clear_output_files(self, project_id: str) -> None:
        output_dir = self.get_output_dir(project_id)
        if not output_dir.exists():
            return

        for path in output_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    def get_model_path(self, project_id: str, model_filename: str) -> Path:
        model_path = self.get_output_dir(project_id) / model_filename
        if not model_path.exists():
            raise ProjectNotFoundError(
                f"El archivo de salida '{model_filename}' no existe para el proyecto '{project_id}'."
            )
        return model_path