import hmac
import logging
from functools import lru_cache
from typing import Any

from fastapi import Header

from app.core.errors import AuthenticationError
from app.services.processing_service import ProcessingService
from app.services.project_service import ProjectService
from app.services.storage_service import StorageService
from config import Settings, get_settings

API_KEY_HEADER = "X-API-Key"
logger = logging.getLogger(__name__)


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


def inspect_api_key(
    x_api_key: str | None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    active_settings = settings or get_settings()
    expected_api_key = (active_settings.api_key or "").strip()
    provided_api_key = x_api_key.strip() if x_api_key and x_api_key.strip() else None
    required = bool(expected_api_key)
    valid = (not required) or (
        provided_api_key is not None and hmac.compare_digest(provided_api_key, expected_api_key)
    )
    return {
        "required": required,
        "valid": valid,
        "provided": bool(provided_api_key),
        "header_name": API_KEY_HEADER,
    }


def require_api_key(
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    inspection = inspect_api_key(x_api_key)
    if inspection["required"] and not inspection["valid"]:
        logger.warning(
            "API key rechazada para backend local. header=%s provided=%s",
            API_KEY_HEADER,
            inspection["provided"],
        )
        raise AuthenticationError("API key faltante o incorrecta.")
