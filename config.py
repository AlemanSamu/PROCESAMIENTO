import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuracion principal del backend local."""

    app_name: str = "Local 3D Processing Module"
    api_prefix: str = ""
    storage_root: Path = Field(default=Path("data/projects"))
    profile: str = "balanced"  # conservative | balanced | quality
    processing_engine: str = "auto"  # colmap | auto | mock
    simulation_delay_seconds: int = 5
    processing_cleanup_workspace_on_failure: bool = True
    processing_execution_timeline_limit: int = 200
    image_preprocessing_max_width: int = 1920
    image_validation_level: str = "standard"  # relaxed | standard | strict
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
    image_validation_enabled: bool = True
    image_validation_min_images_required: int = 6
    image_validation_min_width: int = 640
    image_validation_min_height: int = 480
    image_validation_min_pixels: int = 307200
    image_validation_min_sharpness_warn: float = 0.06
    image_validation_min_sharpness_reject: float = 0.04
    image_validation_min_brightness: float = 0.15
    image_validation_max_brightness: float = 0.9
    image_validation_exposure_warn_margin: float = 0.07
    image_validation_near_duplicate_warn_hamming: int = 6
    image_validation_near_duplicate_reject_hamming: int = 2
    image_validation_coverage_min_unique_ratio: float = 0.55
    image_validation_coverage_min_median_hamming: int = 8
    image_validation_coverage_max_neighbor_similarity_ratio: float = 0.7
    image_validation_block_on_low_coverage: bool = False
    image_selection_enabled: bool = True
    image_selection_min_images_required: int = 6
    image_selection_max_images: int = 60
    image_selection_target_keep_ratio: float = 0.75
    image_selection_min_quality_score: float = 0.35
    image_selection_quality_weight: float = 0.65
    image_selection_diversity_weight: float = 0.35
    image_selection_near_duplicate_hamming: int = 3
    image_selection_diversity_min_hamming: int = 8
    image_object_segmentation_enabled: bool = True
    image_object_segmentation_analysis_max_width: int = 512
    image_object_segmentation_min_component_area_ratio: float = 0.003
    image_object_segmentation_max_component_area_ratio: float = 0.68
    image_object_segmentation_min_component_fill_ratio: float = 0.10
    image_object_segmentation_expected_aspect_ratio: float = 1.6
    image_object_segmentation_aspect_tolerance: float = 2.0
    image_object_segmentation_mask_padding_ratio: float = 0.045
    image_object_segmentation_min_component_score: float = 0.16
    image_object_segmentation_min_segmented_images: int = 2
    image_object_segmentation_min_segmented_ratio: float = 0.20
    image_object_segmentation_block_on_low_success: bool = False
    primitive_box_fallback_enabled: bool = True
    primitive_box_fallback_min_selected_images: int = 3
    primitive_box_fallback_analysis_max_width: int = 256
    primitive_box_fallback_min_foreground_ratio: float = 0.03
    primitive_box_fallback_texture_enabled: bool = True
    primitive_box_fallback_on_incoherent_output: bool = True
    primitive_box_fallback_incoherent_min_registered_images: int = 8
    primitive_box_fallback_incoherent_min_sparse_points: int = 1200
    primitive_box_fallback_incoherent_min_points_per_registered_image: int = 180
    primitive_box_fallback_incoherent_min_faces: int = 20
    primitive_box_fallback_incoherent_max_faces: int = 700
    primitive_box_fallback_incoherent_max_extent_ratio: float = 6.0
    primitive_box_fallback_incoherent_min_bbox_fill_ratio: float = 0.12
    primitive_box_fallback_incoherent_max_bbox_fill_ratio: float = 1.05
    primitive_box_fallback_replace_sparse_bounding_box: bool = True
    metrics_evidence_enabled: bool = True
    metrics_evidence_root: Path = Field(default=Path("data/experiments"))
    metrics_experiment_variant: str = "enhanced"
    metrics_experiment_scenario: str = "auto"
    force_presentable_model_enabled: bool = False
    force_presentable_model_glb: Path | None = None
    force_presentable_model_obj: Path | None = None
    api_key: str | None = None
    colmap_path: str | None = None
    colmap_binary: str = "colmap"
    colmap_timeout_seconds: int = 1800
    colmap_use_gpu: bool = True
    colmap_gpu_mode: str = "auto"  # auto | enabled | disabled
    colmap_gpu_probe_timeout_seconds: int = 3
    colmap_enable_dense_stages: bool = False
    colmap_camera_model: str = "SIMPLE_RADIAL"
    colmap_single_camera: bool = True
    colmap_fallback_to_mock: bool = False
    colmap_require_dense_reconstruction: bool = False
    cors_allowed_origins: str = "*"
    cors_allow_credentials: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LOCAL3D_",
        case_sensitive=False,
    )


_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "conservative": {
        "image_preprocessing_max_width": 1280,
        "image_selection_max_images": 32,
        "colmap_use_gpu": False,
        "colmap_gpu_mode": "disabled",
        "colmap_enable_dense_stages": False,
        "colmap_timeout_seconds": 900,
        "image_validation_level": "strict",
        "image_validation_min_images_required": 6,
        "image_selection_min_images_required": 6,
        "image_validation_min_sharpness_reject": 0.05,
        "primitive_box_fallback_enabled": True,
        "colmap_fallback_to_mock": False,
    },
    "balanced": {
        "image_preprocessing_max_width": 1920,
        "image_selection_max_images": 60,
        "colmap_use_gpu": True,
        "colmap_gpu_mode": "auto",
        "colmap_enable_dense_stages": False,
        "colmap_timeout_seconds": 1800,
        "image_validation_level": "standard",
        "image_validation_min_images_required": 6,
        "image_selection_min_images_required": 6,
        "image_validation_min_sharpness_reject": 0.04,
        "primitive_box_fallback_enabled": True,
        "colmap_fallback_to_mock": False,
    },
    "quality": {
        "image_preprocessing_max_width": 2560,
        "image_selection_max_images": 100,
        "colmap_use_gpu": True,
        "colmap_gpu_mode": "auto",
        "colmap_enable_dense_stages": False,
        "colmap_timeout_seconds": 3600,
        "image_validation_level": "relaxed",
        "image_validation_min_images_required": 8,
        "image_selection_min_images_required": 8,
        "image_validation_min_sharpness_reject": 0.035,
        "primitive_box_fallback_enabled": True,
        "colmap_fallback_to_mock": False,
    },
}


def _env_name(field_name: str) -> str:
    return f"LOCAL3D_{field_name.upper()}"


def _configured_env_keys() -> set[str]:
    keys = {key.upper() for key in os.environ}
    env_path = Path(".env")
    if not env_path.exists():
        return keys

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key = line.split("=", 1)[0].strip()
            if key:
                keys.add(key.upper())
    except OSError:
        return keys
    return keys


def _env_is_set(field_name: str, configured_keys: set[str]) -> bool:
    return _env_name(field_name) in configured_keys


def _apply_profile_defaults(settings: Settings) -> Settings:
    profile = str(settings.profile or "balanced").strip().lower()
    if profile not in _PROFILE_DEFAULTS:
        profile = "balanced"
    settings.profile = profile
    configured_keys = _configured_env_keys()

    for field_name, value in _PROFILE_DEFAULTS[profile].items():
        if not _env_is_set(field_name, configured_keys):
            setattr(settings, field_name, value)

    return settings


@lru_cache
def get_settings() -> Settings:
    settings = _apply_profile_defaults(Settings())
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    return settings
