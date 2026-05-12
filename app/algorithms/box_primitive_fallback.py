from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

from app.core.errors import ProcessingError
from app.models.schemas import OutputFormat

from .artifacts import MeshModel, write_json
from .exporter import ModelExporter

try:  # Optional: keep runtime stable on limited environments without OpenCV.
    import cv2  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


@dataclass(frozen=True)
class BoxPrimitiveFallbackSettings:
    enabled: bool = False
    min_selected_images: int = 3
    analysis_max_width: int = 256
    min_foreground_ratio: float = 0.03
    texture_enabled: bool = True

    @classmethod
    def from_settings(cls, settings: Any) -> "BoxPrimitiveFallbackSettings":
        return cls(
            enabled=bool(getattr(settings, "primitive_box_fallback_enabled", False)),
            min_selected_images=max(1, int(getattr(settings, "primitive_box_fallback_min_selected_images", 3))),
            analysis_max_width=max(64, int(getattr(settings, "primitive_box_fallback_analysis_max_width", 256))),
            min_foreground_ratio=max(
                0.0,
                min(1.0, float(getattr(settings, "primitive_box_fallback_min_foreground_ratio", 0.03))),
            ),
            texture_enabled=bool(getattr(settings, "primitive_box_fallback_texture_enabled", True)),
        )


@dataclass(frozen=True)
class BoxPrimitiveFallbackResult:
    model_path: Path
    output_format: OutputFormat
    report_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _ImageShapeObservation:
    path: Path
    filename: str
    width_height_ratio: float
    foreground_ratio: float
    rectangularity: float
    bbox_relative: tuple[float, float, float, float]
    bbox_absolute: tuple[int, int, int, int]
    quad_relative: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    quad_absolute: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
    brightness: float
    sharpness: float
    tilt_degrees: float
    center_offset: float
    perspective_ratio: float
    frontal_score: float
    side_score: float
    top_score: float
    lighting_score: float
    face_ratio_score: float
    selection_score: float
    segmentation_assisted: bool


@dataclass(frozen=True)
class _MaskComponent:
    pixel_count: int
    bbox: tuple[int, int, int, int]
    corners: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class _TextureAtlas:
    atlas_path: Path
    source_image: str
    crop_box: tuple[int, int, int, int]
    quad_box: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
    atlas_size: tuple[int, int]
    perspective_corrected: bool
    selection_score: float
    source_brightness: float
    source_sharpness: float
    source_tilt_degrees: float
    face_sources: dict[str, str | None]
    face_rectification: dict[str, str]


class BoxPrimitiveFallback:
    name = "primitive_box_fallback"

    def __init__(
        self,
        settings: BoxPrimitiveFallbackSettings | None = None,
        exporter: ModelExporter | None = None,
    ) -> None:
        self.settings = settings or BoxPrimitiveFallbackSettings()
        self.exporter = exporter or ModelExporter()

    @classmethod
    def from_settings(cls, settings: Any) -> "BoxPrimitiveFallback":
        return cls(settings=BoxPrimitiveFallbackSettings.from_settings(settings))

    def build_from_images(
        self,
        *,
        project_id: str,
        selected_images_dir: Path,
        output_dir: Path,
        output_format: OutputFormat,
        source_reason: str,
        segmentation_summary: dict[str, Any] | None = None,
    ) -> BoxPrimitiveFallbackResult:
        image_paths = self._collect_images(selected_images_dir)
        if len(image_paths) < self.settings.min_selected_images:
            raise ProcessingError(
                "No hay imagenes seleccionadas suficientes para construir fallback geometrico tipo box.",
                reason_code="box_fallback_insufficient_images",
                current_stage="primitive_box_fallback",
                metadata={
                    "current_stage": "primitive_box_fallback",
                    "reason_code": "box_fallback_insufficient_images",
                    "selected_images_count": len(image_paths),
                    "min_selected_images_required": self.settings.min_selected_images,
                },
                allow_fallback=False,
                retryable=False,
            )

        segmentation_records = self._index_segmentation_records(segmentation_summary)
        observations = self._observe_images(
            image_paths,
            segmentation_records=segmentation_records,
        )
        if not observations:
            raise ProcessingError(
                "No fue posible estimar siluetas utiles para fallback geometrico tipo box.",
                reason_code="box_fallback_no_valid_observations",
                current_stage="primitive_box_fallback",
                metadata={
                    "current_stage": "primitive_box_fallback",
                    "reason_code": "box_fallback_no_valid_observations",
                    "selected_images_count": len(image_paths),
                },
                allow_fallback=False,
                retryable=False,
            )

        pipeline_dir = output_dir / "pipeline"
        dimensions, sparse_reference = self._estimate_box_dimensions(output_dir, observations)
        mesh = self._build_box_mesh(dimensions)
        texture_atlas: _TextureAtlas | None = None
        if self.settings.texture_enabled and output_format == OutputFormat.GLB:
            texture_atlas = self._build_texture_atlas(project_id, pipeline_dir, observations)
        texture_export_error: str | None = None

        if output_format == OutputFormat.GLB and texture_atlas is not None:
            try:
                export_result = self.exporter.export_textured_box(
                    project_id=project_id,
                    dimensions=dimensions,
                    texture_atlas_path=texture_atlas.atlas_path,
                    output_dir=output_dir,
                    output_format=output_format,
                    work_dir=pipeline_dir,
                )
            except Exception as exc:
                texture_export_error = str(exc)
                export_result = self.exporter.export(
                    project_id=project_id,
                    mesh=mesh,
                    output_dir=output_dir,
                    output_format=output_format,
                    work_dir=pipeline_dir,
                )
        else:
            export_result = self.exporter.export(
                project_id=project_id,
                mesh=mesh,
                output_dir=output_dir,
                output_format=output_format,
                work_dir=pipeline_dir,
            )

        captured_texture_report = self._build_captured_texture_payload(
            texture_atlas=texture_atlas,
            texture_export_error=texture_export_error,
            output_format=output_format,
            include_report_fields=True,
        )
        captured_texture_metadata = self._build_captured_texture_payload(
            texture_atlas=texture_atlas,
            texture_export_error=texture_export_error,
            output_format=output_format,
            include_report_fields=False,
        )

        report_payload = {
            "fallback_name": self.name,
            "project_id": project_id,
            "output_format": output_format.value,
            "source_reason": source_reason,
            "selected_images_dir": str(selected_images_dir),
            "selected_images_count": len(image_paths),
            "estimated_box_dimensions": {
                "width": round(dimensions[0], 4),
                "height": round(dimensions[1], 4),
                "depth": round(dimensions[2], 4),
            },
            "observations": [
                {
                    "filename": item.filename,
                    "width_height_ratio": round(item.width_height_ratio, 4),
                    "foreground_ratio": round(item.foreground_ratio, 4),
                    "rectangularity": round(item.rectangularity, 4),
                    "brightness": round(item.brightness, 6),
                    "sharpness": round(item.sharpness, 6),
                    "tilt_degrees": round(item.tilt_degrees, 4),
                    "center_offset": round(item.center_offset, 6),
                    "perspective_ratio": round(item.perspective_ratio, 6),
                    "frontal_score": round(item.frontal_score, 6),
                    "side_score": round(item.side_score, 6),
                    "top_score": round(item.top_score, 6),
                    "lighting_score": round(item.lighting_score, 6),
                    "face_ratio_score": round(item.face_ratio_score, 6),
                    "selection_score": round(item.selection_score, 6),
                    "segmentation_assisted": bool(item.segmentation_assisted),
                }
                for item in observations
            ],
            "sparse_reference": sparse_reference,
            "export_result": export_result.to_dict(),
            "captured_texture": captured_texture_report,
            "note": (
                "Este artefacto es una aproximacion geometrica simplificada tipo box, "
                "no una reconstruccion fotogrametrica exacta."
            ),
        }
        report_path = write_json(pipeline_dir / f"{project_id}_box_fallback_report.json", report_payload)

        metadata = {
            "reconstruction_type": "approximate_box_primitive_fallback",
            "method_used": "primitive_box",
            "current_stage": "completed_with_fallback",
            "status_message": "Reconstruccion completada con fallback geometrico tipo box.",
            "progress": 1.0,
            "approximate_geometry_fallback": {
                "used": True,
                "type": "box",
                "source_reason": source_reason,
                "report_path": str(report_path),
                "selected_images_dir": str(selected_images_dir),
                "selected_images_count": len(image_paths),
                "estimated_box_dimensions": {
                    "width": round(dimensions[0], 4),
                    "height": round(dimensions[1], 4),
                    "depth": round(dimensions[2], 4),
                },
                "sparse_reference": sparse_reference,
                "captured_texture": captured_texture_metadata,
            },
            "metrics": {
                "image_count_processed": len(image_paths),
                "mesh_vertex_count": len(mesh.vertices),
                "mesh_face_count": len(mesh.faces),
            },
            "artifacts": {
                "box_fallback_report": str(report_path),
                "model_path": str(export_result.model_path),
            },
        }

        return BoxPrimitiveFallbackResult(
            model_path=export_result.model_path,
            output_format=output_format,
            report_path=report_path,
            metadata=metadata,
        )

    @staticmethod
    def _collect_images(images_dir: Path) -> list[Path]:
        if not images_dir.exists() or not images_dir.is_dir():
            return []
        return sorted((path for path in images_dir.iterdir() if path.is_file()), key=lambda item: item.name.lower())

    def _build_captured_texture_payload(
        self,
        *,
        texture_atlas: _TextureAtlas | None,
        texture_export_error: str | None,
        output_format: OutputFormat,
        include_report_fields: bool,
    ) -> dict[str, Any]:
        applied = bool(texture_atlas is not None and texture_export_error is None and output_format == OutputFormat.GLB)
        payload: dict[str, Any] = {
            "enabled": bool(self.settings.texture_enabled),
            "applied": applied,
            "atlas_path": str(texture_atlas.atlas_path) if texture_atlas is not None else None,
            "source_image": texture_atlas.source_image if texture_atlas is not None else None,
            "quad_box": [list(item) for item in texture_atlas.quad_box] if texture_atlas is not None else None,
            "perspective_corrected": bool(texture_atlas.perspective_corrected) if texture_atlas is not None else False,
            "selection_score": round(float(texture_atlas.selection_score), 6) if texture_atlas is not None else None,
            "source_brightness": round(float(texture_atlas.source_brightness), 6) if texture_atlas is not None else None,
            "source_sharpness": round(float(texture_atlas.source_sharpness), 6) if texture_atlas is not None else None,
            "source_tilt_degrees": round(float(texture_atlas.source_tilt_degrees), 4) if texture_atlas is not None else None,
            "face_sources": texture_atlas.face_sources if texture_atlas is not None else {},
            "face_rectification": texture_atlas.face_rectification if texture_atlas is not None else {},
            "error": texture_export_error,
        }
        if include_report_fields:
            payload["atlas_size"] = list(texture_atlas.atlas_size) if texture_atlas is not None else None
            payload["crop_box"] = list(texture_atlas.crop_box) if texture_atlas is not None else None
        return payload

    @staticmethod
    def _index_segmentation_records(
        segmentation_summary: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(segmentation_summary, dict):
            return {}
        image_records = segmentation_summary.get("images")
        if not isinstance(image_records, list):
            return {}

        indexed: dict[str, dict[str, Any]] = {}
        for item in image_records:
            if not isinstance(item, dict):
                continue
            filename = str(item.get("filename") or "").strip()
            if not filename:
                continue
            indexed[filename.lower()] = item
        return indexed

    def _observe_images(
        self,
        image_paths: list[Path],
        *,
        segmentation_records: dict[str, dict[str, Any]] | None = None,
    ) -> list[_ImageShapeObservation]:
        observations: list[_ImageShapeObservation] = []
        indexed_records = segmentation_records or {}
        for image_path in image_paths:
            record = indexed_records.get(image_path.name.lower())
            observation = self._observe_single_image(
                image_path,
                segmentation_record=record,
            )
            if observation is not None:
                observations.append(observation)
        return observations

    def _observe_single_image(
        self,
        path: Path,
        *,
        segmentation_record: dict[str, Any] | None = None,
    ) -> _ImageShapeObservation | None:
        try:
            with Image.open(path) as image:
                color_image = ImageOps.exif_transpose(image).convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError):
            return None

        grayscale = color_image.convert("L")
        segmentation_observation = self._build_observation_from_segmentation(
            path=path,
            grayscale=grayscale,
            segmentation_record=segmentation_record,
        )
        if segmentation_observation is not None:
            return segmentation_observation

        normalized = grayscale
        resized = self._resize_for_analysis(normalized)
        mask = self._build_foreground_mask(resized)
        component = self._largest_component(mask)
        if component is None:
            return None

        left, top, right, bottom = component.bbox
        bbox_width = max(1, right - left)
        bbox_height = max(1, bottom - top)
        bbox_area = bbox_width * bbox_height
        total_area = max(1, resized.width * resized.height)
        foreground_ratio = bbox_area / total_area
        if foreground_ratio < self.settings.min_foreground_ratio:
            return None

        brightness, sharpness = self._estimate_region_quality(resized, component.bbox)
        rectangularity = component.pixel_count / float(max(1, bbox_area))
        quad_relative = self._corners_to_relative(component.corners, resized.size)
        bbox_absolute = self._analysis_bbox_to_absolute(component.bbox, normalized.size, resized.size)
        quad_absolute = self._analysis_quad_to_absolute(component.corners, normalized.size, resized.size)
        tilt_degrees = self._estimate_tilt_degrees(component.corners)
        center_offset = self._estimate_center_offset(component.bbox, resized.size)
        perspective_ratio = self._estimate_perspective_ratio(component.corners)
        frontal_score = self._score_frontal(tilt_degrees, center_offset, foreground_ratio)
        side_score = self._score_side(
            width_height_ratio=float(bbox_width) / float(max(bbox_height, 1)),
            perspective_ratio=perspective_ratio,
            center_offset=center_offset,
        )
        top_score = self._score_top(
            corners=component.corners,
            width_height_ratio=float(bbox_width) / float(max(bbox_height, 1)),
            center_offset=center_offset,
        )
        lighting_score = self._score_lighting(brightness)
        sharpness_score = self._score_sharpness(sharpness)
        face_ratio_score = self._score_face_ratio(float(bbox_width) / float(max(bbox_height, 1)))
        rectangularity_score = max(0.0, min(1.0, rectangularity / 0.75))
        selection_score = (
            sharpness_score * 0.28
            + frontal_score * 0.26
            + face_ratio_score * 0.20
            + rectangularity_score * 0.16
            + lighting_score * 0.10
        )

        return _ImageShapeObservation(
            path=path,
            filename=path.name,
            width_height_ratio=bbox_width / max(bbox_height, 1),
            foreground_ratio=foreground_ratio,
            rectangularity=rectangularity,
            bbox_relative=(
                left / float(resized.width),
                top / float(resized.height),
                right / float(resized.width),
                bottom / float(resized.height),
            ),
            bbox_absolute=bbox_absolute,
            quad_relative=quad_relative,
            quad_absolute=quad_absolute,
            brightness=round(brightness, 6),
            sharpness=round(sharpness, 6),
            tilt_degrees=round(tilt_degrees, 4),
            center_offset=round(center_offset, 6),
            perspective_ratio=round(perspective_ratio, 6),
            frontal_score=round(frontal_score, 6),
            side_score=round(side_score, 6),
            top_score=round(top_score, 6),
            lighting_score=round(lighting_score, 6),
            face_ratio_score=round(face_ratio_score, 6),
            selection_score=round(selection_score, 6),
            segmentation_assisted=False,
        )

    def _build_observation_from_segmentation(
        self,
        *,
        path: Path,
        grayscale: Image.Image,
        segmentation_record: dict[str, Any] | None,
    ) -> _ImageShapeObservation | None:
        if not isinstance(segmentation_record, dict):
            return None
        if str(segmentation_record.get("status") or "").strip().lower() != "segmented":
            return None

        image_size = grayscale.size
        bbox = self._record_bbox_to_absolute(segmentation_record, image_size)
        if bbox is None:
            return None
        left, top, right, bottom = bbox
        bbox_width = max(1, right - left)
        bbox_height = max(1, bottom - top)
        bbox_area = bbox_width * bbox_height
        total_area = max(1, image_size[0] * image_size[1])
        foreground_ratio = bbox_area / float(total_area)
        if foreground_ratio < self.settings.min_foreground_ratio:
            return None

        contour = self._record_quad_to_absolute(segmentation_record, image_size)
        if contour is None:
            contour = (
                (left, top),
                (right, top),
                (right, bottom),
                (left, bottom),
            )

        bbox_relative = (
            left / float(max(1, image_size[0])),
            top / float(max(1, image_size[1])),
            right / float(max(1, image_size[0])),
            bottom / float(max(1, image_size[1])),
        )
        quad_relative = self._corners_to_relative(contour, image_size)
        contour_area = self._quad_area(contour)
        rectangularity = contour_area / float(max(1, bbox_area))
        object_metrics = segmentation_record.get("object_metrics")
        if isinstance(object_metrics, dict):
            maybe_rectangularity = object_metrics.get("rectangularity")
            if isinstance(maybe_rectangularity, (int, float)):
                rectangularity = float(maybe_rectangularity)

        brightness, sharpness = self._estimate_region_quality(grayscale, bbox)
        tilt_value = self._estimate_tilt_degrees(contour)
        if isinstance(object_metrics, dict):
            maybe_tilt = object_metrics.get("tilt_degrees")
            if isinstance(maybe_tilt, (int, float)):
                tilt_value = float(maybe_tilt)
        center_offset = self._estimate_center_offset(bbox, image_size)
        ratio = float(bbox_width) / float(max(bbox_height, 1))
        perspective_ratio = self._estimate_perspective_ratio(contour)
        frontal_score = self._score_frontal(tilt_value, center_offset, foreground_ratio)
        side_score = self._score_side(
            width_height_ratio=ratio,
            perspective_ratio=perspective_ratio,
            center_offset=center_offset,
        )
        top_score = self._score_top(
            corners=contour,
            width_height_ratio=ratio,
            center_offset=center_offset,
        )
        lighting_score = self._score_lighting(brightness)
        sharpness_score = self._score_sharpness(sharpness)
        face_ratio_score = self._score_face_ratio(ratio)
        rectangularity_score = max(0.0, min(1.0, rectangularity / 0.75))
        selection_score = (
            sharpness_score * 0.25
            + frontal_score * 0.25
            + face_ratio_score * 0.20
            + rectangularity_score * 0.20
            + lighting_score * 0.10
        )
        return _ImageShapeObservation(
            path=path,
            filename=path.name,
            width_height_ratio=ratio,
            foreground_ratio=foreground_ratio,
            rectangularity=rectangularity,
            bbox_relative=bbox_relative,
            bbox_absolute=bbox,
            quad_relative=quad_relative,
            quad_absolute=contour,
            brightness=round(brightness, 6),
            sharpness=round(sharpness, 6),
            tilt_degrees=round(tilt_value, 4),
            center_offset=round(center_offset, 6),
            perspective_ratio=round(perspective_ratio, 6),
            frontal_score=round(frontal_score, 6),
            side_score=round(side_score, 6),
            top_score=round(top_score, 6),
            lighting_score=round(lighting_score, 6),
            face_ratio_score=round(face_ratio_score, 6),
            selection_score=round(selection_score, 6),
            segmentation_assisted=True,
        )

    @staticmethod
    def _record_bbox_to_absolute(
        segmentation_record: dict[str, Any],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int] | None:
        width, height = image_size
        if width <= 1 or height <= 1:
            return None

        bbox_value = segmentation_record.get("bbox")
        if isinstance(bbox_value, list | tuple) and len(bbox_value) == 4:
            try:
                left = int(round(float(bbox_value[0])))
                top = int(round(float(bbox_value[1])))
                right = int(round(float(bbox_value[2])))
                bottom = int(round(float(bbox_value[3])))
            except (TypeError, ValueError):
                left = top = right = bottom = 0
            left = max(0, min(width, left))
            top = max(0, min(height, top))
            right = max(left + 1, min(width, right))
            bottom = max(top + 1, min(height, bottom))
            if right - left >= 4 and bottom - top >= 4:
                return left, top, right, bottom

        bbox_relative = segmentation_record.get("bbox_relative")
        if isinstance(bbox_relative, list | tuple) and len(bbox_relative) == 4:
            try:
                left_rel = max(0.0, min(1.0, float(bbox_relative[0])))
                top_rel = max(0.0, min(1.0, float(bbox_relative[1])))
                right_rel = max(0.0, min(1.0, float(bbox_relative[2])))
                bottom_rel = max(0.0, min(1.0, float(bbox_relative[3])))
            except (TypeError, ValueError):
                return None
            left = int(round(left_rel * width))
            top = int(round(top_rel * height))
            right = int(round(right_rel * width))
            bottom = int(round(bottom_rel * height))
            left = max(0, min(width, left))
            top = max(0, min(height, top))
            right = max(left + 1, min(width, right))
            bottom = max(top + 1, min(height, bottom))
            if right - left >= 4 and bottom - top >= 4:
                return left, top, right, bottom
        return None

    @staticmethod
    def _record_quad_to_absolute(
        segmentation_record: dict[str, Any],
        image_size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
        width, height = image_size
        if width <= 1 or height <= 1:
            return None

        contour_value = segmentation_record.get("contour_points")
        if isinstance(contour_value, list) and len(contour_value) >= 4:
            points: list[tuple[int, int]] = []
            for raw_point in contour_value[:4]:
                if not isinstance(raw_point, list | tuple) or len(raw_point) < 2:
                    continue
                try:
                    px = int(round(float(raw_point[0])))
                    py = int(round(float(raw_point[1])))
                except (TypeError, ValueError):
                    continue
                points.append((max(0, min(width - 1, px)), max(0, min(height - 1, py))))
            if len(points) == 4:
                quad = (points[0], points[1], points[2], points[3])
                if BoxPrimitiveFallback._quad_area(quad) >= 80.0:
                    return quad

        contour_relative = segmentation_record.get("contour_points_relative")
        if isinstance(contour_relative, list) and len(contour_relative) >= 4:
            points = []
            for raw_point in contour_relative[:4]:
                if not isinstance(raw_point, list | tuple) or len(raw_point) < 2:
                    continue
                try:
                    rel_x = max(0.0, min(1.0, float(raw_point[0])))
                    rel_y = max(0.0, min(1.0, float(raw_point[1])))
                except (TypeError, ValueError):
                    continue
                abs_x = int(round(rel_x * width))
                abs_y = int(round(rel_y * height))
                points.append((max(0, min(width - 1, abs_x)), max(0, min(height - 1, abs_y))))
            if len(points) == 4:
                quad = (points[0], points[1], points[2], points[3])
                if BoxPrimitiveFallback._quad_area(quad) >= 80.0:
                    return quad
        return None

    def _build_foreground_mask(self, image: Image.Image) -> Image.Image:
        background_level = self._estimate_background_level(image)
        diff_image = ImageChops.difference(image, Image.new("L", image.size, color=background_level))
        diff_stats = ImageStat.Stat(diff_image)
        diff_mean = float(diff_stats.mean[0] if diff_stats.mean else 0.0)
        diff_std = float((diff_stats.stddev[0] if diff_stats.stddev else 0.0))
        threshold = int(max(18, min(105, round(diff_mean * 1.1 + diff_std * 0.4))))
        mask = diff_image.point(lambda value: 255 if value >= threshold else 0)
        mask = mask.filter(ImageFilter.MedianFilter(size=3))
        return mask.point(lambda value: 255 if value >= 128 else 0)

    @staticmethod
    def _largest_component(mask: Image.Image) -> _MaskComponent | None:
        width, height = mask.size
        if width <= 0 or height <= 0:
            return None

        pixels = mask.load()
        if pixels is None:
            return None

        visited = bytearray(width * height)
        best_count = 0
        best_bbox: tuple[int, int, int, int] | None = None
        best_corners: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None = None

        for y in range(height):
            for x in range(width):
                index = y * width + x
                if visited[index]:
                    continue
                visited[index] = 1
                if int(pixels[x, y]) <= 0:
                    continue

                stack: list[tuple[int, int]] = [(x, y)]
                points: list[tuple[int, int]] = []
                left = right = x
                top = bottom = y

                while stack:
                    cx, cy = stack.pop()
                    points.append((cx, cy))
                    if cx < left:
                        left = cx
                    if cx > right:
                        right = cx
                    if cy < top:
                        top = cy
                    if cy > bottom:
                        bottom = cy

                    for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                        if nx < 0 or ny < 0 or nx >= width or ny >= height:
                            continue
                        n_index = ny * width + nx
                        if visited[n_index]:
                            continue
                        visited[n_index] = 1
                        if int(pixels[nx, ny]) > 0:
                            stack.append((nx, ny))

                count = len(points)
                if count < best_count:
                    continue

                corners = BoxPrimitiveFallback._estimate_component_corners(points)
                if corners is None:
                    continue

                best_count = count
                best_bbox = (left, top, right + 1, bottom + 1)
                best_corners = corners

        if best_count <= 0 or best_bbox is None or best_corners is None:
            return None

        return _MaskComponent(
            pixel_count=best_count,
            bbox=best_bbox,
            corners=best_corners,
        )

    @staticmethod
    def _estimate_component_corners(
        points: list[tuple[int, int]],
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
        if not points:
            return None

        top_left = min(points, key=lambda item: (item[0] + item[1], item[0]))
        top_right = max(points, key=lambda item: (item[0] - item[1], item[0]))
        bottom_right = max(points, key=lambda item: (item[0] + item[1], item[1]))
        bottom_left = max(points, key=lambda item: (item[1] - item[0], item[1]))
        return top_left, top_right, bottom_right, bottom_left

    @staticmethod
    def _corners_to_relative(
        corners: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
        image_size: tuple[int, int],
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        width, height = image_size
        safe_w = max(1.0, float(width))
        safe_h = max(1.0, float(height))
        return (
            (corners[0][0] / safe_w, corners[0][1] / safe_h),
            (corners[1][0] / safe_w, corners[1][1] / safe_h),
            (corners[2][0] / safe_w, corners[2][1] / safe_h),
            (corners[3][0] / safe_w, corners[3][1] / safe_h),
        )

    @staticmethod
    def _analysis_bbox_to_absolute(
        bbox: tuple[int, int, int, int],
        original_size: tuple[int, int],
        analysis_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        width, height = original_size
        analysis_w, analysis_h = analysis_size
        scale_x = width / float(max(1, analysis_w))
        scale_y = height / float(max(1, analysis_h))
        left = int(round(bbox[0] * scale_x))
        top = int(round(bbox[1] * scale_y))
        right = int(round(bbox[2] * scale_x))
        bottom = int(round(bbox[3] * scale_y))
        left = max(0, min(width, left))
        top = max(0, min(height, top))
        right = max(left + 1, min(width, right))
        bottom = max(top + 1, min(height, bottom))
        return left, top, right, bottom

    @staticmethod
    def _analysis_quad_to_absolute(
        corners: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
        original_size: tuple[int, int],
        analysis_size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        width, height = original_size
        analysis_w, analysis_h = analysis_size
        scale_x = width / float(max(1, analysis_w))
        scale_y = height / float(max(1, analysis_h))
        mapped: list[tuple[int, int]] = []
        for point_x, point_y in corners:
            abs_x = int(round(point_x * scale_x))
            abs_y = int(round(point_y * scale_y))
            mapped.append(
                (
                    max(0, min(width - 1, abs_x)),
                    max(0, min(height - 1, abs_y)),
                )
            )
        return (mapped[0], mapped[1], mapped[2], mapped[3])

    @staticmethod
    def _estimate_region_quality(image: Image.Image, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        region = image.crop(bbox)
        brightness_stats = ImageStat.Stat(region)
        brightness = float((brightness_stats.mean[0] if brightness_stats.mean else 0.0) / 255.0)
        edge_region = region.filter(ImageFilter.FIND_EDGES)
        edge_stats = ImageStat.Stat(edge_region)
        sharpness = float((edge_stats.var[0] if edge_stats.var else 0.0) / (255.0 * 255.0))
        return brightness, sharpness

    @staticmethod
    def _estimate_tilt_degrees(
        corners: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
    ) -> float:
        top_left, top_right, _bottom_right, _bottom_left = corners
        dx = float(top_right[0] - top_left[0])
        dy = float(top_right[1] - top_left[1])
        if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
            return 45.0
        angle = abs(math.degrees(math.atan2(dy, dx)))
        while angle >= 180.0:
            angle -= 180.0
        if angle > 90.0:
            angle = 180.0 - angle
        return min(angle, abs(90.0 - angle))

    @staticmethod
    def _estimate_perspective_ratio(
        corners: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
    ) -> float:
        top_width = BoxPrimitiveFallback._distance(corners[0], corners[1])
        bottom_width = BoxPrimitiveFallback._distance(corners[3], corners[2])
        larger = max(top_width, bottom_width, 1e-6)
        smaller = min(top_width, bottom_width)
        return max(0.0, min(1.0, smaller / larger))

    @staticmethod
    def _estimate_center_offset(
        bbox: tuple[int, int, int, int],
        image_size: tuple[int, int],
    ) -> float:
        width, height = image_size
        if width <= 0 or height <= 0:
            return 1.0
        left, top, right, bottom = bbox
        center_x = (left + right) * 0.5
        center_y = (top + bottom) * 0.5
        image_center_x = width * 0.5
        image_center_y = height * 0.5
        dx = (center_x - image_center_x) / max(width * 0.5, 1.0)
        dy = (center_y - image_center_y) / max(height * 0.5, 1.0)
        return min(1.0, math.sqrt(dx * dx + dy * dy))

    @staticmethod
    def _score_sharpness(value: float) -> float:
        return max(0.0, min(1.0, value / 0.025))

    @staticmethod
    def _score_lighting(brightness: float) -> float:
        target = 0.55
        distance = abs(float(brightness) - target)
        return max(0.0, min(1.0, 1.0 - (distance / 0.45)))

    @staticmethod
    def _score_face_ratio(width_height_ratio: float) -> float:
        target = 1.8
        spread = 1.05
        ratio = float(width_height_ratio)
        return max(0.0, min(1.0, 1.0 - (abs(ratio - target) / spread)))

    @staticmethod
    def _score_frontal(tilt_degrees: float, center_offset: float, foreground_ratio: float) -> float:
        tilt_score = max(0.0, min(1.0, 1.0 - (float(tilt_degrees) / 30.0)))
        center_score = max(0.0, min(1.0, 1.0 - float(center_offset)))
        fill_target = 0.2
        fill_spread = 0.25
        fill_score = max(0.0, min(1.0, 1.0 - (abs(float(foreground_ratio) - fill_target) / fill_spread)))
        return tilt_score * 0.4 + center_score * 0.2 + fill_score * 0.4

    @staticmethod
    def _score_side(width_height_ratio: float, perspective_ratio: float, center_offset: float) -> float:
        ratio = float(width_height_ratio)
        ratio_target = 0.60
        ratio_spread = 0.75
        ratio_score = max(0.0, min(1.0, 1.0 - (abs(ratio - ratio_target) / ratio_spread)))
        perspective_score = max(0.0, min(1.0, 1.0 - abs(float(perspective_ratio) - 0.65) / 0.45))
        center_score = max(0.0, min(1.0, 1.0 - float(center_offset)))
        return ratio_score * 0.45 + perspective_score * 0.35 + center_score * 0.20

    @staticmethod
    def _score_top(
        *,
        corners: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
        width_height_ratio: float,
        center_offset: float,
    ) -> float:
        top_width = BoxPrimitiveFallback._distance(corners[0], corners[1])
        bottom_width = BoxPrimitiveFallback._distance(corners[3], corners[2])
        perspective = 0.0
        if bottom_width > 1e-6:
            perspective = max(0.0, min(1.0, (bottom_width - top_width) / bottom_width))
        ratio_score = max(0.0, min(1.0, 1.0 - abs(float(width_height_ratio) - 1.6) / 1.6))
        center_score = max(0.0, min(1.0, 1.0 - float(center_offset)))
        return perspective * 0.45 + ratio_score * 0.35 + center_score * 0.20

    def _build_texture_atlas(
        self,
        project_id: str,
        pipeline_dir: Path,
        observations: list[_ImageShapeObservation],
    ) -> _TextureAtlas | None:
        if not observations:
            return None

        front_observation = self._select_face_observation(observations, face="front")
        if front_observation is None:
            return None
        side_observation = self._select_face_observation(observations, face="side")
        top_observation = self._select_face_observation(observations, face="top")

        front_payload = self._extract_front_texture(front_observation)
        if front_payload is None:
            return None
        front_source, crop_box, quad_box, perspective_corrected, front_rectification = front_payload
        if front_source.width < 24 or front_source.height < 24:
            return None

        front_source = self._normalize_front_texture_orientation(front_source)
        tile_size = 512
        front = ImageOps.pad(
            front_source,
            (tile_size, tile_size),
            method=Image.Resampling.LANCZOS,
            color=(236, 238, 241),
            centering=(0.5, 0.48),
        )
        front = front.filter(ImageFilter.UnsharpMask(radius=1.1, percent=130, threshold=3))

        average_color = tuple(int(channel) for channel in ImageStat.Stat(front).mean[:3])
        back = self._build_solid_face_tile(average_color, tile_size, shade=0.92)
        bottom = self._build_solid_face_tile(average_color, tile_size, shade=0.74)

        side_tile: Image.Image | None = None
        side_source_name: str | None = None
        if side_observation is not None:
            side_payload = self._extract_observation_crop(side_observation)
            if side_payload is not None:
                side_crop, _side_crop_box = side_payload
                side_tile, side_orientation = self._extract_best_side_strip_tile(
                    side_crop,
                    tile_size=tile_size,
                    preferred=self._infer_side_strip_orientation(side_observation),
                )
                side_source_name = side_observation.filename
        if side_tile is None:
            left = self._build_solid_face_tile(average_color, tile_size, shade=0.86)
            right = self._build_solid_face_tile(average_color, tile_size, shade=0.82)
        else:
            left = side_tile
            right = ImageOps.mirror(side_tile)

        top_tile: Image.Image | None = None
        top_source_name: str | None = None
        if top_observation is not None:
            top_payload = self._extract_observation_crop(top_observation)
            if top_payload is not None:
                top_crop, _top_crop_box = top_payload
                top_tile = self._extract_strip_tile(top_crop, orientation="top", tile_size=tile_size)
                top_source_name = top_observation.filename
        if top_tile is None:
            top = self._extract_strip_tile(front, orientation="top", tile_size=tile_size)
        else:
            top = top_tile

        atlas = Image.new("RGB", (tile_size * 3, tile_size * 2), color=(234, 236, 239))
        atlas.paste(front, (0 * tile_size, 0 * tile_size))
        atlas.paste(back, (1 * tile_size, 0 * tile_size))
        atlas.paste(top, (2 * tile_size, 0 * tile_size))
        atlas.paste(left, (0 * tile_size, 1 * tile_size))
        atlas.paste(right, (1 * tile_size, 1 * tile_size))
        atlas.paste(bottom, (2 * tile_size, 1 * tile_size))

        pipeline_dir.mkdir(parents=True, exist_ok=True)
        atlas_path = pipeline_dir / f"{project_id}_box_texture_atlas.png"
        atlas.save(atlas_path, format="PNG")
        return _TextureAtlas(
            atlas_path=atlas_path,
            source_image=front_observation.filename,
            crop_box=crop_box,
            quad_box=quad_box,
            atlas_size=atlas.size,
            perspective_corrected=perspective_corrected,
            selection_score=float(front_observation.selection_score),
            source_brightness=float(front_observation.brightness),
            source_sharpness=float(front_observation.sharpness),
            source_tilt_degrees=float(front_observation.tilt_degrees),
            face_sources={
                "front": front_observation.filename,
                "back": None,
                "left": side_source_name,
                "right": side_source_name,
                "top": top_source_name if top_source_name is not None else front_observation.filename,
                "bottom": None,
            },
            face_rectification={
                "front": front_rectification,
                "back": "solid_color",
                "left": f"strip_from_side_crop_{side_orientation}" if side_tile is not None else "solid_color",
                "right": f"mirrored_side_strip_{side_orientation}" if side_tile is not None else "solid_color",
                "top": "strip_from_top_crop" if top_tile is not None else "strip_from_front_tile",
                "bottom": "solid_color",
            },
        )

    def _select_face_observation(
        self,
        observations: list[_ImageShapeObservation],
        *,
        face: str,
    ) -> _ImageShapeObservation | None:
        if not observations:
            return None

        def _score(item: _ImageShapeObservation) -> float:
            sharpness_score = self._score_sharpness(item.sharpness)
            center_score = max(0.0, min(1.0, 1.0 - item.center_offset))
            segmentation_bonus = 0.06 if item.segmentation_assisted else 0.0
            if face == "front":
                return (
                    item.frontal_score * 0.55
                    + item.selection_score * 0.20
                    + sharpness_score * 0.12
                    + item.face_ratio_score * 0.08
                    + center_score * 0.05
                    + segmentation_bonus
                )
            if face == "side":
                return (
                    item.side_score * 0.58
                    + sharpness_score * 0.14
                    + item.lighting_score * 0.12
                    + center_score * 0.10
                    + item.selection_score * 0.06
                    + segmentation_bonus
                )
            return (
                item.top_score * 0.58
                + sharpness_score * 0.14
                + item.lighting_score * 0.12
                + center_score * 0.10
                + item.selection_score * 0.06
                + segmentation_bonus
            )

        best: _ImageShapeObservation | None = None
        best_score = -1.0
        for item in observations:
            current = _score(item)
            if current > best_score:
                best_score = current
                best = item
        return best

    def _extract_front_texture(
        self,
        observation: _ImageShapeObservation,
    ) -> tuple[
        Image.Image,
        tuple[int, int, int, int],
        tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
        bool,
        str,
    ] | None:
        payload = self._extract_observation_crop(observation)
        if payload is None:
            return None
        crop, crop_box = payload

        try:
            with Image.open(observation.path) as source:
                color_image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError):
            return None

        quad = observation.quad_absolute
        if not self._quad_is_usable_for_rectification(quad, color_image.size):
            quad = (
                (crop_box[0], crop_box[1]),
                (crop_box[2], crop_box[1]),
                (crop_box[2], crop_box[3]),
                (crop_box[0], crop_box[3]),
            )
            return color_image.crop(crop_box), crop_box, quad, False, "crop_bbox"

        rectified = self._rectify_front_face(color_image, quad)
        if rectified is not None and rectified.width >= 24 and rectified.height >= 24:
            return rectified, crop_box, quad, True, "perspective_rectified"
        return color_image.crop(crop_box), crop_box, quad, False, "crop_bbox"

    def _extract_observation_crop(
        self,
        observation: _ImageShapeObservation,
    ) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
        try:
            with Image.open(observation.path) as source:
                color_image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError):
            return None

        crop_box = self._resolve_observation_crop_box(
            observation=observation,
            color_image=color_image,
        )
        if crop_box is None:
            return None
        crop = color_image.crop(crop_box)
        if crop.width < 24 or crop.height < 24:
            return None
        return crop, crop_box

    def _resolve_observation_crop_box(
        self,
        observation: _ImageShapeObservation,
        color_image: Image.Image,
    ) -> tuple[int, int, int, int] | None:
        image_size = color_image.size
        width, height = image_size
        if width <= 0 or height <= 0:
            return None

        left, top, right, bottom = observation.bbox_absolute
        if right - left < 4 or bottom - top < 4:
            bbox = self._relative_bbox_to_absolute(observation.bbox_relative, image_size)
        else:
            bbox = (
                max(0, min(width, left)),
                max(0, min(height, top)),
                max(0, min(width, right)),
                max(0, min(height, bottom)),
            )
        if bbox is None:
            return None

        if self._should_refine_texture_crop(bbox, image_size):
            expected_ratio = self._clamp(observation.width_height_ratio, 1.1, 2.4)
            refined = self._refine_texture_crop_from_saturation(
                image=color_image,
                anchor_crop_box=bbox,
                expected_ratio=expected_ratio,
            )
            if refined is not None:
                bbox = refined
        return bbox

    @staticmethod
    def _infer_side_strip_orientation(observation: _ImageShapeObservation) -> str:
        left_rel, _, right_rel, _ = observation.bbox_relative
        center_x = (left_rel + right_rel) * 0.5
        # If the object center appears on the right half, prefer left strip to capture visible side face.
        return "left" if center_x >= 0.5 else "right"

    @staticmethod
    def _default_texture_anchor_box(image_size: tuple[int, int]) -> tuple[int, int, int, int]:
        width, height = image_size
        if width <= 0 or height <= 0:
            return 0, 0, 1, 1
        return (
            int(round(width * 0.10)),
            int(round(height * 0.23)),
            int(round(width * 0.90)),
            int(round(height * 0.88)),
        )

    def _score_texture_crop_candidate(
        self,
        *,
        image: Image.Image,
        observation: _ImageShapeObservation,
        crop_box: tuple[int, int, int, int],
    ) -> float:
        image_width, image_height = image.size
        if image_width <= 0 or image_height <= 0:
            return -1.0

        left, top, right, bottom = crop_box
        crop_width = max(1, right - left)
        crop_height = max(1, bottom - top)

        area_ratio = (crop_width * crop_height) / float(image_width * image_height)
        ratio = crop_width / float(max(1, crop_height))
        center_x = ((left + right) * 0.5) / float(image_width)
        center_y = ((top + bottom) * 0.5) / float(image_height)

        area_score = max(0.0, min(1.0, 1.0 - (abs(area_ratio - 0.18) / 0.17)))
        ratio_score = max(0.0, min(1.0, 1.0 - (abs(ratio - 1.6) / 0.8)))
        center_distance = math.sqrt((center_x - 0.5) ** 2 + (center_y - 0.55) ** 2)
        center_score = max(0.0, min(1.0, 1.0 - (center_distance / 0.65)))
        saturation_coverage = self._estimate_crop_saturation_coverage(image, crop_box)
        saturation_score = max(0.0, min(1.0, 1.0 - (abs(saturation_coverage - 0.17) / 0.14)))
        sharpness_score = self._score_sharpness(observation.sharpness)

        return (
            area_score * 0.28
            + ratio_score * 0.23
            + center_score * 0.18
            + saturation_score * 0.21
            + sharpness_score * 0.10
        )

    @staticmethod
    def _estimate_crop_saturation_coverage(
        image: Image.Image,
        crop_box: tuple[int, int, int, int],
    ) -> float:
        region = image.crop(crop_box)
        if region.width < 4 or region.height < 4:
            return 0.0
        saturation = region.convert("HSV").split()[1]
        sat_stats = ImageStat.Stat(saturation)
        sat_mean = float(sat_stats.mean[0] if sat_stats.mean else 0.0)
        sat_std = float(sat_stats.stddev[0] if sat_stats.stddev else 0.0)
        sat_threshold = int(max(36, min(140, round(sat_mean + (sat_std * 0.7)))))
        mask = saturation.point(lambda value: 255 if value >= sat_threshold else 0)
        mask_stats = ImageStat.Stat(mask)
        return float((mask_stats.mean[0] if mask_stats.mean else 0.0) / 255.0)

    @staticmethod
    def _normalize_front_texture_orientation(front_source: Image.Image) -> Image.Image:
        if front_source.width < 16 or front_source.height < 16:
            return front_source

        saturation = front_source.convert("HSV").split()[1]
        width, height = saturation.size
        slice_height = max(4, int(round(height * 0.36)))
        top_band = saturation.crop((0, 0, width, slice_height))
        bottom_band = saturation.crop((0, height - slice_height, width, height))
        top_mean = float(ImageStat.Stat(top_band).mean[0])
        bottom_mean = float(ImageStat.Stat(bottom_band).mean[0])
        if bottom_mean > (top_mean * 1.08):
            return front_source.rotate(180, expand=False)
        return front_source

    @staticmethod
    def _count_bbox_border_touches(
        bbox: tuple[int, int, int, int],
        image_size: tuple[int, int],
        *,
        margin: int = 2,
    ) -> int:
        left, top, right, bottom = bbox
        width, height = image_size
        if width <= 0 or height <= 0:
            return 4

        touches = 0
        if left <= margin:
            touches += 1
        if top <= margin:
            touches += 1
        if right >= max(0, width - margin):
            touches += 1
        if bottom >= max(0, height - margin):
            touches += 1
        return touches

    def _should_refine_texture_crop(
        self,
        crop_box: tuple[int, int, int, int],
        image_size: tuple[int, int],
    ) -> bool:
        width, height = image_size
        if width <= 0 or height <= 0:
            return False
        left, top, right, bottom = crop_box
        crop_width = max(1, right - left)
        crop_height = max(1, bottom - top)
        crop_area_ratio = (crop_width * crop_height) / float(width * height)
        aspect_ratio = crop_width / float(max(1, crop_height))
        border_touches = self._count_bbox_border_touches(crop_box, image_size, margin=2)
        if crop_area_ratio > 0.42:
            return True
        if border_touches >= 2:
            return True
        if aspect_ratio < 0.45 or aspect_ratio > 3.1:
            return True
        return False

    def _refine_texture_crop_from_saturation(
        self,
        *,
        image: Image.Image,
        anchor_crop_box: tuple[int, int, int, int],
        expected_ratio: float,
    ) -> tuple[int, int, int, int] | None:
        image_width, image_height = image.size
        if image_width < 24 or image_height < 24:
            return None

        max_analysis_width = 320
        if image_width > max_analysis_width:
            scale = max_analysis_width / float(image_width)
            analysis_width = max_analysis_width
            analysis_height = max(32, int(round(image_height * scale)))
            analysis = image.resize((analysis_width, analysis_height), Image.Resampling.BILINEAR)
        else:
            scale = 1.0
            analysis = image
            analysis_width, analysis_height = image.size

        saturation = analysis.convert("HSV").split()[1]
        sat_stats = ImageStat.Stat(saturation)
        sat_mean = float(sat_stats.mean[0] if sat_stats.mean else 0.0)
        sat_std = float(sat_stats.stddev[0] if sat_stats.stddev else 0.0)
        sat_threshold = int(max(36, min(140, round(sat_mean + (sat_std * 0.7)))))

        mask = saturation.point(lambda value: 255 if value >= sat_threshold else 0)
        mask = mask.filter(ImageFilter.MedianFilter(size=3))
        mask = mask.filter(ImageFilter.MaxFilter(size=3))
        mask = mask.point(lambda value: 255 if value >= 128 else 0)

        anchor_center_rel = self._bbox_center_relative(anchor_crop_box, image.size)
        candidate_bbox = self._select_saturation_component_bbox(
            mask,
            anchor_center_rel,
            expected_ratio=self._clamp(expected_ratio, 0.8, 3.0),
        )
        if candidate_bbox is None:
            return None

        left_s, top_s, right_s, bottom_s = candidate_bbox
        left = int(round(left_s / scale))
        top = int(round(top_s / scale))
        right = int(round(right_s / scale))
        bottom = int(round(bottom_s / scale))

        left = max(0, min(image_width, left))
        top = max(0, min(image_height, top))
        right = max(left + 1, min(image_width, right))
        bottom = max(top + 1, min(image_height, bottom))

        component_width = max(1, right - left)
        component_height = max(1, bottom - top)
        target_ratio = self._clamp(expected_ratio, 1.15, 2.3)
        desired_width = max(
            int(round(component_width * 1.6)),
            int(round(component_height * target_ratio * 0.95)),
            int(round(image_width * 0.14)),
        )
        desired_height = max(
            int(round(component_height * 1.35)),
            int(round(desired_width / max(target_ratio, 1e-6))),
            int(round(image_height * 0.1)),
        )
        desired_width = min(desired_width, int(round(image_width * 0.64)))
        desired_height = min(desired_height, int(round(image_height * 0.45)))

        center_x = (left + right) * 0.5 - (component_width * 0.16)
        center_y = (top + bottom) * 0.5 - (component_height * 0.46)

        refined_left = int(round(center_x - desired_width * 0.5))
        refined_top = int(round(center_y - desired_height * 0.5))
        refined_right = refined_left + desired_width
        refined_bottom = refined_top + desired_height

        if refined_left < 0:
            refined_right -= refined_left
            refined_left = 0
        if refined_top < 0:
            refined_bottom -= refined_top
            refined_top = 0
        if refined_right > image_width:
            overflow = refined_right - image_width
            refined_left = max(0, refined_left - overflow)
            refined_right = image_width
        if refined_bottom > image_height:
            overflow = refined_bottom - image_height
            refined_top = max(0, refined_top - overflow)
            refined_bottom = image_height

        if refined_right - refined_left < 24 or refined_bottom - refined_top < 24:
            return None

        refined_area_ratio = (
            (refined_right - refined_left) * (refined_bottom - refined_top)
        ) / float(max(1, image_width * image_height))
        if refined_area_ratio > 0.56:
            return None

        return refined_left, refined_top, refined_right, refined_bottom

    @staticmethod
    def _bbox_center_relative(
        bbox: tuple[int, int, int, int],
        image_size: tuple[int, int],
    ) -> tuple[float, float]:
        width, height = image_size
        if width <= 0 or height <= 0:
            return 0.5, 0.5
        left, top, right, bottom = bbox
        center_x = ((left + right) * 0.5) / float(width)
        center_y = ((top + bottom) * 0.5) / float(height)
        return (
            max(0.0, min(1.0, center_x)),
            max(0.0, min(1.0, center_y)),
        )

    def _select_saturation_component_bbox(
        self,
        mask: Image.Image,
        anchor_center: tuple[float, float],
        expected_ratio: float | None = None,
    ) -> tuple[int, int, int, int] | None:
        width, height = mask.size
        if width <= 0 or height <= 0:
            return None

        pixels = mask.load()
        if pixels is None:
            return None

        visited = bytearray(width * height)
        best_bbox: tuple[int, int, int, int] | None = None
        best_score = -1.0
        total_area = float(max(1, width * height))
        anchor_x, anchor_y = anchor_center

        for y in range(height):
            for x in range(width):
                index = y * width + x
                if visited[index]:
                    continue
                visited[index] = 1
                if int(pixels[x, y]) <= 0:
                    continue

                stack: list[tuple[int, int]] = [(x, y)]
                pixel_count = 0
                left = right = x
                top = bottom = y

                while stack:
                    cx, cy = stack.pop()
                    pixel_count += 1
                    if cx < left:
                        left = cx
                    if cx > right:
                        right = cx
                    if cy < top:
                        top = cy
                    if cy > bottom:
                        bottom = cy

                    for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                        if nx < 0 or ny < 0 or nx >= width or ny >= height:
                            continue
                        n_index = ny * width + nx
                        if visited[n_index]:
                            continue
                        visited[n_index] = 1
                        if int(pixels[nx, ny]) > 0:
                            stack.append((nx, ny))

                if pixel_count < 32:
                    continue

                bbox = (left, top, right + 1, bottom + 1)
                bbox_width = max(1, bbox[2] - bbox[0])
                bbox_height = max(1, bbox[3] - bbox[1])
                bbox_area = float(bbox_width * bbox_height)
                bbox_area_ratio = bbox_area / total_area
                if bbox_area_ratio < 0.003:
                    continue

                area_score = max(0.0, min(1.0, 1.0 - (abs(bbox_area_ratio - 0.08) / 0.14)))
                fill_ratio = pixel_count / max(1.0, bbox_area)
                fill_score = max(0.0, min(1.0, fill_ratio / 0.8))
                bbox_ratio = bbox_width / float(max(1, bbox_height))
                ratio_target = self._clamp(expected_ratio if expected_ratio is not None else 1.6, 0.8, 3.0)
                ratio_spread = max(0.45, ratio_target * 0.6)
                ratio_score = max(0.0, min(1.0, 1.0 - (abs(bbox_ratio - ratio_target) / ratio_spread)))

                touches = self._count_bbox_border_touches(bbox, (width, height), margin=1)
                border_score = max(0.0, min(1.0, 1.0 - (touches / 3.0)))

                center_x = ((bbox[0] + bbox[2]) * 0.5) / float(width)
                center_y = ((bbox[1] + bbox[3]) * 0.5) / float(height)
                center_distance = math.sqrt((center_x - anchor_x) ** 2 + (center_y - anchor_y) ** 2)
                center_score = max(0.0, min(1.0, 1.0 - (center_distance / 0.7)))

                score = (
                    area_score * 0.35
                    + center_score * 0.30
                    + ratio_score * 0.15
                    + border_score * 0.12
                    + fill_score * 0.08
                )

                if score > best_score:
                    best_score = score
                    best_bbox = bbox

        return best_bbox

    @staticmethod
    def _relative_bbox_to_absolute(
        bbox_relative: tuple[float, float, float, float],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int] | None:
        width, height = image_size
        if width <= 1 or height <= 1:
            return None

        left_rel, top_rel, right_rel, bottom_rel = bbox_relative
        left = int(round(max(0.0, min(1.0, left_rel)) * width))
        top = int(round(max(0.0, min(1.0, top_rel)) * height))
        right = int(round(max(0.0, min(1.0, right_rel)) * width))
        bottom = int(round(max(0.0, min(1.0, bottom_rel)) * height))

        pad_x = max(4, int(round((right - left) * 0.06)))
        pad_y = max(4, int(round((bottom - top) * 0.06)))
        left = max(0, left - pad_x)
        top = max(0, top - pad_y)
        right = min(width, right + pad_x)
        bottom = min(height, bottom + pad_y)

        if right - left < 4 or bottom - top < 4:
            return None
        return left, top, right, bottom

    @staticmethod
    def _relative_quad_to_absolute(
        quad_relative: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]],
        image_size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
        width, height = image_size
        if width <= 1 or height <= 1:
            return None

        points: list[tuple[int, int]] = []
        for rel_x, rel_y in quad_relative:
            abs_x = int(round(max(0.0, min(1.0, rel_x)) * width))
            abs_y = int(round(max(0.0, min(1.0, rel_y)) * height))
            points.append((abs_x, abs_y))
        if len(points) != 4:
            return None

        quad = (points[0], points[1], points[2], points[3])
        if BoxPrimitiveFallback._quad_area(quad) < 80.0:
            return None
        return quad

    def _rectify_front_face(
        self,
        image: Image.Image,
        quad_box: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
    ) -> Image.Image | None:
        if self._quad_area(quad_box) < 80.0:
            return None

        top_width = self._distance(quad_box[0], quad_box[1])
        bottom_width = self._distance(quad_box[3], quad_box[2])
        left_height = self._distance(quad_box[0], quad_box[3])
        right_height = self._distance(quad_box[1], quad_box[2])

        output_width = int(round(max(48.0, min(1200.0, (top_width + bottom_width) * 0.5))))
        output_height = int(round(max(48.0, min(1200.0, (left_height + right_height) * 0.5))))
        if output_width < 24 or output_height < 24:
            return None

        top_left, top_right, bottom_right, bottom_left = quad_box
        # PIL QUAD expects source points in order: UL, LL, LR, UR.
        quad_data = (
            float(top_left[0]),
            float(top_left[1]),
            float(bottom_left[0]),
            float(bottom_left[1]),
            float(bottom_right[0]),
            float(bottom_right[1]),
            float(top_right[0]),
            float(top_right[1]),
        )
        try:
            return image.transform(
                (output_width, output_height),
                Image.Transform.QUAD,
                quad_data,
                resample=Image.Resampling.BICUBIC,
            )
        except Exception:
            return None

    def _quad_is_usable_for_rectification(
        self,
        quad_box: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
        image_size: tuple[int, int],
    ) -> bool:
        width, height = image_size
        if width <= 1 or height <= 1:
            return False

        area = self._quad_area(quad_box)
        area_ratio = area / float(width * height)
        if area_ratio < 0.01 or area_ratio > 0.55:
            return False

        margin = 2
        touches = 0
        for x, y in quad_box:
            if x <= margin or x >= width - margin or y <= margin or y >= height - margin:
                touches += 1
        if touches > 1:
            return False

        top_width = self._distance(quad_box[0], quad_box[1])
        bottom_width = self._distance(quad_box[3], quad_box[2])
        left_height = self._distance(quad_box[0], quad_box[3])
        right_height = self._distance(quad_box[1], quad_box[2])
        mean_width = max(1e-6, (top_width + bottom_width) * 0.5)
        mean_height = max(1e-6, (left_height + right_height) * 0.5)
        aspect_ratio = mean_width / mean_height
        if aspect_ratio < 0.4 or aspect_ratio > 3.0:
            return False

        return True

    @staticmethod
    def _quad_area(
        quad: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
    ) -> float:
        points = [quad[0], quad[1], quad[2], quad[3]]
        area = 0.0
        for index in range(len(points)):
            x1, y1 = points[index]
            x2, y2 = points[(index + 1) % len(points)]
            area += float(x1 * y2 - x2 * y1)
        return abs(area) * 0.5

    @staticmethod
    def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
        dx = float(a[0] - b[0])
        dy = float(a[1] - b[1])
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _build_solid_face_tile(base_color: tuple[int, int, int], tile_size: int, *, shade: float) -> Image.Image:
        r, g, b = base_color
        color = (
            max(0, min(255, int(round(r * shade)))),
            max(0, min(255, int(round(g * shade)))),
            max(0, min(255, int(round(b * shade)))),
        )
        return Image.new("RGB", (tile_size, tile_size), color=color)

    @staticmethod
    def _extract_strip_tile(crop: Image.Image, *, orientation: str, tile_size: int) -> Image.Image:
        width, height = crop.size
        vertical_band = max(4, int(round(width * 0.22)))
        horizontal_band = max(4, int(round(height * 0.22)))

        if orientation == "left":
            region = crop.crop((0, 0, vertical_band, height))
        elif orientation == "right":
            region = crop.crop((max(0, width - vertical_band), 0, width, height))
        elif orientation == "top":
            region = crop.crop((0, 0, width, horizontal_band))
        else:
            region = crop.crop((0, max(0, height - horizontal_band), width, height))

        fitted = ImageOps.fit(region, (tile_size, tile_size), Image.Resampling.BILINEAR, centering=(0.5, 0.5))
        return fitted.filter(ImageFilter.GaussianBlur(radius=0.6))

    @staticmethod
    def _extract_best_side_strip_tile(
        crop: Image.Image,
        *,
        tile_size: int,
        preferred: str = "left",
    ) -> tuple[Image.Image, str]:
        width, height = crop.size
        vertical_band = max(4, int(round(width * 0.22)))
        left_region = crop.crop((0, 0, vertical_band, height))
        right_region = crop.crop((max(0, width - vertical_band), 0, width, height))
        center_band = max(vertical_band, int(round(width * 0.36)))
        center_left = max(0, int(round((width - center_band) * 0.5)))
        center_region = crop.crop((center_left, 0, min(width, center_left + center_band), height))

        def _score(region: Image.Image) -> float:
            hsv = region.convert("HSV")
            saturation = hsv.split()[1]
            sat_mask = saturation.point(lambda value: 255 if value >= 45 else 0)
            sat_stats = ImageStat.Stat(sat_mask)
            sat_ratio = float((sat_stats.mean[0] if sat_stats.mean else 0.0) / 255.0)
            edges = region.convert("L").filter(ImageFilter.FIND_EDGES)
            edge_stats = ImageStat.Stat(edges)
            edge_variance = float((edge_stats.var[0] if edge_stats.var else 0.0) / (255.0 * 255.0))
            return sat_ratio * 0.70 + min(1.0, edge_variance / 0.025) * 0.30

        candidates = {
            "left": (left_region, _score(left_region)),
            "right": (right_region, _score(right_region)),
            "center": (center_region, _score(center_region)),
        }
        candidate_items = sorted(candidates.items(), key=lambda item: item[1][1], reverse=True)
        chosen_orientation = candidate_items[0][0]
        top_score = candidate_items[0][1][1]
        if len(candidate_items) > 1 and abs(top_score - candidate_items[1][1][1]) <= 0.02:
            if preferred in candidates:
                chosen_orientation = preferred
        region = candidates[chosen_orientation][0]
        fitted = ImageOps.fit(region, (tile_size, tile_size), Image.Resampling.BILINEAR, centering=(0.5, 0.5))
        return fitted.filter(ImageFilter.GaussianBlur(radius=0.6)), chosen_orientation

    def _resize_for_analysis(self, image: Image.Image) -> Image.Image:
        if image.width <= self.settings.analysis_max_width:
            return image
        target_width = self.settings.analysis_max_width
        target_height = max(32, int(round(image.height * (target_width / image.width))))
        return image.resize((target_width, target_height), Image.Resampling.BILINEAR)

    @staticmethod
    def _estimate_background_level(image: Image.Image) -> int:
        width, height = image.size
        margin_x = max(1, width // 20)
        margin_y = max(1, height // 20)

        strips = [
            image.crop((0, 0, width, margin_y)),
            image.crop((0, height - margin_y, width, height)),
            image.crop((0, 0, margin_x, height)),
            image.crop((width - margin_x, 0, width, height)),
        ]
        values: list[float] = []
        for strip in strips:
            stats = ImageStat.Stat(strip)
            values.append(float(stats.mean[0] if stats.mean else 0.0))
        if not values:
            return 127
        return int(round(median(values)))

    def _estimate_box_dimensions(
        self,
        output_dir: Path,
        observations: list[_ImageShapeObservation],
    ) -> tuple[tuple[float, float, float], dict[str, Any]]:
        ratios = [item.width_height_ratio for item in observations]
        foreground = [item.foreground_ratio for item in observations]
        median_ratio = float(median(ratios)) if ratios else 1.0
        median_foreground = float(median(foreground)) if foreground else 0.0

        width = self._clamp(median_ratio, 0.8, 1.6)
        height = 1.0

        sparse_ratio, sparse_reference = self._estimate_depth_ratio_from_sparse(output_dir)
        if sparse_ratio is not None:
            depth = self._clamp(width * sparse_ratio, 0.6, 1.5)
        else:
            # Controlled demo default: almost cube for visual cleanliness.
            depth = self._clamp(width * 0.9, 0.7, 1.3)
            sparse_reference = {
                "used": False,
                "reason": "sparse_reference_not_available",
            }

        # If silhouette is very compact, nudge toward cube to keep final object stable and presentable.
        if median_foreground >= 0.35:
            width = self._clamp((width + 1.0) / 2.0, 0.85, 1.4)
            depth = self._clamp((depth + 1.0) / 2.0, 0.75, 1.35)

        sparse_reference["median_image_ratio"] = round(median_ratio, 4)
        sparse_reference["median_foreground_ratio"] = round(median_foreground, 4)
        return (width, height, depth), sparse_reference

    def _estimate_depth_ratio_from_sparse(self, output_dir: Path) -> tuple[float | None, dict[str, Any]]:
        points_path = output_dir / "colmap_sparse_txt" / "points3D.txt"
        if not points_path.exists() or not points_path.is_file():
            return None, {"used": False, "path": str(points_path), "reason": "points3D_txt_not_found"}

        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []

        try:
            for line in points_path.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                parts = raw.split()
                if len(parts) < 4:
                    continue
                try:
                    xs.append(float(parts[1]))
                    ys.append(float(parts[2]))
                    zs.append(float(parts[3]))
                except ValueError:
                    continue
                if len(xs) >= 15000:
                    break
        except OSError:
            return None, {"used": False, "path": str(points_path), "reason": "points3D_txt_unreadable"}

        if len(xs) < 20:
            return None, {"used": False, "path": str(points_path), "reason": "insufficient_sparse_points"}

        extent_x = max(xs) - min(xs)
        extent_y = max(ys) - min(ys)
        extent_z = max(zs) - min(zs)
        extents = sorted([extent_x, extent_y, extent_z], reverse=True)
        if extents[0] <= 1e-9:
            return None, {"used": False, "path": str(points_path), "reason": "degenerate_sparse_extents"}

        depth_ratio = self._clamp(extents[1] / extents[0], 0.65, 1.15)
        return depth_ratio, {
            "used": True,
            "path": str(points_path),
            "sparse_point_count": len(xs),
            "extents": {
                "x": round(extent_x, 6),
                "y": round(extent_y, 6),
                "z": round(extent_z, 6),
            },
            "normalized_depth_ratio": round(depth_ratio, 4),
        }

    @staticmethod
    def _build_box_mesh(dimensions: tuple[float, float, float]) -> MeshModel:
        width, height, depth = dimensions
        hx = width / 2.0
        hy = height / 2.0
        hz = depth / 2.0

        vertices = [
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (-hx, hy, -hz),
            (-hx, -hy, hz),
            (hx, -hy, hz),
            (hx, hy, hz),
            (-hx, hy, hz),
        ]

        faces = [
            (0, 1, 2),
            (0, 2, 3),
            (4, 6, 5),
            (4, 7, 6),
            (0, 4, 5),
            (0, 5, 1),
            (1, 5, 6),
            (1, 6, 2),
            (2, 6, 7),
            (2, 7, 3),
            (3, 7, 4),
            (3, 4, 0),
        ]

        return MeshModel(
            vertices=[(round(x, 6), round(y, 6), round(z, 6)) for x, y, z in vertices],
            faces=faces,
            centroid=(0.0, 0.0, 0.0),
            source_point_count=8,
        )

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))
