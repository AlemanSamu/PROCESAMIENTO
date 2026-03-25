from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuracion principal del backend local."""

    app_name: str = "Local 3D Processing Module"
    api_prefix: str = ""
    storage_root: Path = Field(default=Path("data/projects"))
    processing_engine: str = "auto"  # auto | mock | colmap
    simulation_delay_seconds: int = 5
    max_images_per_project: int = 250
    allowed_image_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    )
    colmap_binary: str = "colmap"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LOCAL3D_",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    return settings
