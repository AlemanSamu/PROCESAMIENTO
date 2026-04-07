from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProjectStatus(str, Enum):
    CREATED = "created"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class OutputFormat(str, Enum):
    GLB = "glb"
    OBJ = "obj"


class ProjectMetadata(BaseModel):
    id: str
    name: str
    status: ProjectStatus = ProjectStatus.CREATED
    created_at: datetime
    updated_at: datetime
    image_count: int = 0
    image_files: list[str] = Field(default_factory=list)
    output_format: OutputFormat | None = None
    model_filename: str | None = None
    error_message: str | None = None
    processing_metadata: dict[str, Any] | None = None


class ProjectCreateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=120)


class ProjectResponse(BaseModel):
    id: str
    name: str
    status: ProjectStatus
    created_at: datetime
    updated_at: datetime
    image_count: int
    output_format: OutputFormat | None = None
    model_filename: str | None = None
    model_download_url: str | None = None
    error_message: str | None = None
    current_stage: str | None = None
    fallback_used: bool | None = None
    final_model_type: str | None = None
    final_model_path: str | None = None
    method_used: str | None = None
    processing_metadata: dict[str, Any] | None = None


class ImageUploadResponse(BaseModel):
    project_id: str
    status: ProjectStatus
    uploaded_count: int
    skipped_count: int = 0
    total_images: int
    uploaded_files: list[str]
    message: str | None = None


class ProcessRequest(BaseModel):
    output_format: OutputFormat = OutputFormat.GLB


class ProcessStartResponse(BaseModel):
    project_id: str
    status: ProjectStatus
    engine: str
    message: str


class ProjectStatusResponse(BaseModel):
    project_id: str
    status: ProjectStatus
    image_count: int
    output_format: OutputFormat | None = None
    model_filename: str | None = None
    model_download_url: str | None = None
    error_message: str | None = None
    engine: str | None = None
    current_stage: str | None = None
    progress: float | None = None
    message: str | None = None
    metrics: dict[str, Any] | None = None
    fallback_used: bool | None = None
    final_model_type: str | None = None
    final_model_path: str | None = None
    method_used: str | None = None
    processing_metadata: dict[str, Any] | None = None
