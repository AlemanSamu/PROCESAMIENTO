from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.models.schemas import OutputFormat


@dataclass(frozen=True)
class ValidatedImage:
    source_path: Path
    normalized_name: str
    index: int
    size_bytes: int
    sha256: str
    extension: str
    width: int = 0
    height: int = 0
    pixel_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "normalized_name": self.normalized_name,
            "index": self.index,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "extension": self.extension,
            "width": self.width,
            "height": self.height,
            "pixel_count": self.pixel_count,
        }


@dataclass(frozen=True)
class PreprocessedImage:
    source: ValidatedImage
    preprocessed_path: Path
    brightness: float
    contrast: float
    sharpness: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "preprocessed_path": str(self.preprocessed_path),
            "brightness": self.brightness,
            "contrast": self.contrast,
            "sharpness": self.sharpness,
        }


@dataclass(frozen=True)
class FeaturePoint:
    x: float
    y: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "score": self.score}


@dataclass(frozen=True)
class ImageFeatures:
    image: PreprocessedImage
    keypoints: list[FeaturePoint]
    descriptor_seed: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image.to_dict(),
            "keypoints": [item.to_dict() for item in self.keypoints],
            "descriptor_seed": self.descriptor_seed,
            "score": self.score,
        }


@dataclass(frozen=True)
class FeatureMatch:
    left_image: str
    right_image: str
    matched_pairs: int
    confidence: float
    correspondences: list[tuple[FeaturePoint, FeaturePoint]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_image": self.left_image,
            "right_image": self.right_image,
            "matched_pairs": self.matched_pairs,
            "confidence": self.confidence,
            "correspondences": [
                {
                    "left": left.to_dict(),
                    "right": right.to_dict(),
                }
                for left, right in self.correspondences
            ],
        }


@dataclass(frozen=True)
class CameraPose:
    image_name: str
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    matrix: list[list[float]]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_name": self.image_name,
            "translation": list(self.translation),
            "rotation": list(self.rotation),
            "matrix": self.matrix,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float
    intensity: float
    source_image: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "intensity": self.intensity,
            "source_image": self.source_image,
        }


@dataclass(frozen=True)
class PointCloud:
    points: list[Point3D]
    bounds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [item.to_dict() for item in self.points],
            "bounds": self.bounds,
        }


@dataclass(frozen=True)
class MeshModel:
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]
    centroid: tuple[float, float, float]
    source_point_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "vertices": [list(vertex) for vertex in self.vertices],
            "faces": [list(face) for face in self.faces],
            "centroid": list(self.centroid),
            "source_point_count": self.source_point_count,
        }


@dataclass(frozen=True)
class ExportResult:
    model_path: Path
    output_format: OutputFormat
    bytes_written: int
    vertex_count: int
    face_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "output_format": self.output_format.value,
            "bytes_written": self.bytes_written,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
        }


@dataclass(frozen=True)
class PipelineStageResult:
    name: str
    status: str
    summary: str
    mode: str = "synthetic"
    artifact_path: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "mode": self.mode,
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class ReconstructionPipelineResult:
    project_id: str
    output_format: OutputFormat
    model_path: Path
    report_path: Path
    stage_results: list[PipelineStageResult]
    image_count: int
    feature_count: int
    match_count: int
    point_count: int
    face_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "output_format": self.output_format.value,
            "model_path": str(self.model_path),
            "report_path": str(self.report_path),
            "stage_results": [item.to_dict() for item in self.stage_results],
            "image_count": self.image_count,
            "feature_count": self.feature_count,
            "match_count": self.match_count,
            "point_count": self.point_count,
            "face_count": self.face_count,
        }


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
