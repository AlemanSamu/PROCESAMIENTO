from functools import lru_cache

from app.services.processing_service import ProcessingService
from app.services.project_service import ProjectService
from app.services.storage_service import StorageService
from config import get_settings


@lru_cache
def get_storage_service() -> StorageService:
    return StorageService(settings=get_settings())


@lru_cache
def get_project_service() -> ProjectService:
    return ProjectService(storage_service=get_storage_service(), settings=get_settings())


@lru_cache
def get_processing_service() -> ProcessingService:
    return ProcessingService(
        project_service=get_project_service(),
        storage_service=get_storage_service(),
        settings=get_settings(),
    )
