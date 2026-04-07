import hashlib
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from fastapi import UploadFile

from app.core.errors import BadRequestError, ProjectNotFoundError, StorageError
from app.models.schemas import ProjectMetadata


@dataclass
class ImageSaveResult:
    saved_files: list[str]
    skipped_files: list[str]
    total_image_count: int

    @property
    def uploaded_count(self) -> int:
        return len(self.saved_files)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_files)


class StorageService:
    METADATA_FILE = "meta.json"
    _UPLOAD_TMP_DIR = ".upload_tmp"

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

    def get_upload_temp_dir(self, project_id: str) -> Path:
        return self.get_project_dir(project_id) / self._UPLOAD_TMP_DIR

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

    def save_images(self, project_id: str, files: Iterable[UploadFile], max_total_images: int | None = None) -> ImageSaveResult:
        images_dir = self.get_images_dir(project_id)
        if not images_dir.exists():
            raise ProjectNotFoundError(f"Proyecto '{project_id}' no encontrado.")

        temp_dir = self.get_upload_temp_dir(project_id)
        temp_dir.mkdir(parents=True, exist_ok=True)

        existing_files = self.list_image_files(project_id)
        existing_hashes = self._build_existing_hash_index(images_dir)
        staged_uploads: list[tuple[Path, str, str]] = []
        skipped_files: list[str] = []

        try:
            for file in files:
                original_name, extension = self._validate_upload_file(file)
                temp_path, content_hash = self._stage_upload_file(project_id, file, extension, original_name)

                if content_hash in existing_hashes:
                    temp_path.unlink(missing_ok=True)
                    skipped_files.append(original_name)
                    continue

                existing_hashes.add(content_hash)
                staged_uploads.append((temp_path, extension, original_name))

            predicted_total = len(existing_files) + len(staged_uploads)
            if max_total_images is not None and predicted_total > max_total_images:
                raise BadRequestError(
                    f"Se excede el maximo de imagenes por proyecto ({max_total_images})."
                )

            saved_files: list[str] = []
            for temp_path, extension, original_name in staged_uploads:
                generated_name = f"{uuid.uuid4().hex}{extension}"
                destination = images_dir / generated_name
                try:
                    temp_path.replace(destination)
                except Exception as exc:
                    raise StorageError(f"No se pudo guardar imagen '{original_name}'.") from exc
                saved_files.append(generated_name)

            return ImageSaveResult(
                saved_files=saved_files,
                skipped_files=skipped_files,
                total_image_count=len(existing_files) + len(saved_files),
            )
        finally:
            for temp_path, _, _ in staged_uploads:
                temp_path.unlink(missing_ok=True)
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def clear_output_files(self, project_id: str) -> None:
        self.clear_processing_artifacts(project_id)

    def clear_processing_artifacts(self, project_id: str) -> None:
        output_dir = self.get_output_dir(project_id)
        if output_dir.exists():
            for path in output_dir.iterdir():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)

        project_dir = self.get_project_dir(project_id)
        stale_paths = [
            project_dir / "database.db",
            project_dir / "database.db-shm",
            project_dir / "database.db-wal",
            project_dir / "workspace",
            project_dir / "sparse",
            project_dir / "dense",
            self.get_upload_temp_dir(project_id),
        ]
        for path in stale_paths:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink(missing_ok=True)

    def get_model_path(self, project_id: str, model_filename: str) -> Path:
        model_path = self.get_output_dir(project_id) / model_filename
        if not model_path.exists():
            raise ProjectNotFoundError(
                f"El archivo de salida '{model_filename}' no existe para el proyecto '{project_id}'."
            )
        return model_path

    def list_image_files(self, project_id: str) -> list[str]:
        images_dir = self.get_images_dir(project_id)
        if not images_dir.exists():
            return []
        return sorted(
            path.name
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self._allowed_extensions
        )

    def _build_existing_hash_index(self, images_dir: Path) -> set[str]:
        hashes: set[str] = set()
        for path in images_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in self._allowed_extensions:
                continue
            try:
                hashes.add(self._hash_file(path))
            except Exception as exc:
                raise StorageError(f"No se pudo calcular el hash de la imagen existente '{path.name}'.") from exc
        return hashes

    def _stage_upload_file(
        self,
        project_id: str,
        file: UploadFile,
        extension: str,
        original_name: str,
    ) -> tuple[Path, str]:
        temp_dir = self.get_upload_temp_dir(project_id)
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{uuid.uuid4().hex}{extension}.upload"
        digest = hashlib.sha256()

        try:
            with temp_path.open("wb") as buffer:
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    buffer.write(chunk)
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            raise StorageError(f"No se pudo procesar imagen '{original_name}'.") from exc

        return temp_path, digest.hexdigest()

    def _validate_upload_file(self, file: UploadFile) -> tuple[str, str]:
        original_name = (file.filename or "").strip()
        if not original_name:
            raise BadRequestError("Todos los archivos deben tener nombre.")

        extension = Path(original_name).suffix.lower()
        if extension not in self._allowed_extensions:
            raise BadRequestError(
                f"Formato de imagen no soportado: '{extension}'. "
                f"Permitidos: {sorted(self._allowed_extensions)}"
            )
        return original_name, extension

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
