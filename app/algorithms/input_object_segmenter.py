from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageFilter,
    ImageOps,
    ImageStat,
    UnidentifiedImageError,
)
try:
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    cv2 = None
    np = None

from .artifacts import write_json


@dataclass(frozen=True)
class InputObjectSegmentationSettings:
    enabled: bool = True
    analysis_max_width: int = 512
    min_component_area_ratio: float = 0.003
    max_component_area_ratio: float = 0.68
    min_component_fill_ratio: float = 0.1
    expected_aspect_ratio: float = 1.6
    aspect_tolerance: float = 2.0
    mask_padding_ratio: float = 0.045
    min_component_score: float = 0.16
    min_segmented_images: int = 2
    min_segmented_ratio: float = 0.2
    block_on_low_success: bool = False

    @classmethod
    def from_settings(cls, settings: Any) -> "InputObjectSegmentationSettings":
        return cls(
            enabled=bool(getattr(settings, "image_object_segmentation_enabled", True)),
            analysis_max_width=max(128, int(getattr(settings, "image_object_segmentation_analysis_max_width", 512))),
            min_component_area_ratio=max(
                0.0,
                min(1.0, float(getattr(settings, "image_object_segmentation_min_component_area_ratio", 0.003))),
            ),
            max_component_area_ratio=max(
                0.0,
                min(1.0, float(getattr(settings, "image_object_segmentation_max_component_area_ratio", 0.68))),
            ),
            min_component_fill_ratio=max(
                0.01,
                min(1.0, float(getattr(settings, "image_object_segmentation_min_component_fill_ratio", 0.1))),
            ),
            expected_aspect_ratio=max(0.2, float(getattr(settings, "image_object_segmentation_expected_aspect_ratio", 1.6))),
            aspect_tolerance=max(0.2, float(getattr(settings, "image_object_segmentation_aspect_tolerance", 2.0))),
            mask_padding_ratio=max(
                0.0,
                min(0.4, float(getattr(settings, "image_object_segmentation_mask_padding_ratio", 0.045))),
            ),
            min_component_score=max(
                0.0,
                min(1.0, float(getattr(settings, "image_object_segmentation_min_component_score", 0.16))),
            ),
            min_segmented_images=max(1, int(getattr(settings, "image_object_segmentation_min_segmented_images", 2))),
            min_segmented_ratio=max(
                0.0,
                min(1.0, float(getattr(settings, "image_object_segmentation_min_segmented_ratio", 0.2))),
            ),
            block_on_low_success=bool(getattr(settings, "image_object_segmentation_block_on_low_success", False)),
        )


@dataclass(frozen=True)
class _ComponentCandidate:
    pixel_count: int
    bbox: tuple[int, int, int, int]
    corners: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
    area_ratio: float
    fill_ratio: float
    aspect_ratio: float
    center_offset: float
    border_touches: int
    score: float


@dataclass(frozen=True)
class InputObjectSegmentationResult:
    allow_processing: bool
    processed_images: list[Path]
    processed_images_dir: Path | None
    masks_dir: Path | None
    report_path: Path | None
    summary: dict[str, Any]


class InputObjectSegmenter:
    name = "input_object_segmentation"

    def __init__(self, settings: InputObjectSegmentationSettings | None = None) -> None:
        self.settings = settings or InputObjectSegmentationSettings()

    @classmethod
    def from_settings(cls, settings: Any) -> "InputObjectSegmenter":
        return cls(settings=InputObjectSegmentationSettings.from_settings(settings))

    def segment_images(
        self,
        selected_images: list[Path],
        report_dir: Path | None = None,
    ) -> InputObjectSegmentationResult:
        ordered_images = list(selected_images)
        total_images = len(ordered_images)
        report_path: Path | None = None

        if not self.settings.enabled:
            summary = {
                "segmenter": self.name,
                "enabled": False,
                "segmentation_enabled": False,
                "segmentation_method": "none",
                "segmentation_success": False,
                "foreground_ratio": 0.0,
                "limitations": ["segmenter_disabled"],
                "allow_processing": True,
                "stage_status": "skipped",
                "candidate_images": total_images,
                "segmented_images": 0,
                "fallback_original_images": total_images,
                "segmentation_ratio": 0.0,
                "blocking_reasons": [],
                "staged_images_dir": str(ordered_images[0].parent) if ordered_images else None,
                "masks_dir": None,
                "images": [
                    {
                        "filename": path.name,
                        "source_path": str(path),
                        "output_path": str(path),
                        "mask_path": None,
                        "status": "fallback_original",
                        "reason": "segmenter_disabled",
                    }
                    for path in ordered_images
                ],
                "thresholds": self._settings_to_dict(),
            }
            if report_dir is not None:
                report_path = write_json(report_dir / "segmentation_report.json", summary)
                write_json(report_dir / "input_object_segmentation_report.json", summary)
            return InputObjectSegmentationResult(
                allow_processing=True,
                processed_images=ordered_images,
                processed_images_dir=ordered_images[0].parent if ordered_images else None,
                masks_dir=None,
                report_path=report_path,
                summary=summary,
            )

        segmented_dir: Path | None = None
        masks_dir: Path | None = None
        crops_dir: Path | None = None
        contours_dir: Path | None = None
        if report_dir is not None:
            segmented_dir = report_dir / "segmented_images"
            masks_dir = report_dir / "masks"
            crops_dir = report_dir / "segmentation_crops"
            contours_dir = report_dir / "segmentation_contours"
            segmented_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "segmentation_masks").mkdir(parents=True, exist_ok=True)
            crops_dir.mkdir(parents=True, exist_ok=True)
            contours_dir.mkdir(parents=True, exist_ok=True)

        processed_images: list[Path] = []
        records: list[dict[str, Any]] = []
        segmented_count = 0
        fallback_count = 0

        for source_path in ordered_images:
            output_path, mask_path, crop_path, contour_path = self._build_output_paths(
                source_path=source_path,
                segmented_dir=segmented_dir,
                masks_dir=masks_dir,
                crops_dir=crops_dir,
                contours_dir=contours_dir,
            )
            record = self._segment_single_image(
                source_path=source_path,
                output_path=output_path,
                mask_path=mask_path,
                crop_path=crop_path,
                contour_path=contour_path,
            )
            records.append(record)
            processed_images.append(output_path)
            if record["status"] == "segmented":
                segmented_count += 1
            else:
                fallback_count += 1

        blocking_reasons: list[str] = []
        segmented_ratio = (segmented_count / total_images) if total_images > 0 else 0.0
        if total_images <= 0:
            blocking_reasons.append("no_input_images")
        if self.settings.block_on_low_success:
            if segmented_count < min(self.settings.min_segmented_images, total_images):
                blocking_reasons.append("insufficient_segmented_images")
            if segmented_ratio < self.settings.min_segmented_ratio:
                blocking_reasons.append("segmented_ratio_below_threshold")

        allow_processing = not blocking_reasons
        summary = {
            "segmenter": self.name,
            "enabled": True,
            "segmentation_enabled": True,
            "allow_processing": allow_processing,
            "stage_status": "completed" if allow_processing else "failed",
            "candidate_images": total_images,
            "segmented_images": segmented_count,
            "fallback_original_images": fallback_count,
            "segmentation_ratio": round(segmented_ratio, 4),
            "blocking_reasons": blocking_reasons,
            "staged_images_dir": str(segmented_dir) if segmented_dir is not None else None,
            "masks_dir": str(masks_dir) if masks_dir is not None else None,
            "crops_dir": str(crops_dir) if crops_dir is not None else None,
            "contours_dir": str(contours_dir) if contours_dir is not None else None,
            "images": records,
            "thresholds": self._settings_to_dict(),
        }
        method_counts: dict[str, int] = {}
        foreground_values: list[float] = []
        success_count = 0
        for item in records:
            method = str(item.get("segmentation_method") or "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1
            try:
                foreground_values.append(float(item.get("foreground_ratio") or 0.0))
            except (TypeError, ValueError):
                pass
            if bool(item.get("segmentation_success")):
                success_count += 1
        summary["segmentation_method"] = max(method_counts, key=method_counts.get) if method_counts else "none"
        summary["segmentation_success"] = success_count > 0
        summary["foreground_ratio"] = round(sum(foreground_values) / len(foreground_values), 6) if foreground_values else 0.0
        summary["limitations"] = [] if success_count > 0 else ["object_not_isolated_reliably"]

        if report_dir is not None:
            report_path = write_json(report_dir / "segmentation_report.json", summary)
            write_json(report_dir / "input_object_segmentation_report.json", summary)
        return InputObjectSegmentationResult(
            allow_processing=allow_processing,
            processed_images=processed_images,
            processed_images_dir=segmented_dir if segmented_dir is not None else (processed_images[0].parent if processed_images else None),
            masks_dir=masks_dir,
            report_path=report_path,
            summary=summary,
        )

    @staticmethod
    def _build_output_paths(
        *,
        source_path: Path,
        segmented_dir: Path | None,
        masks_dir: Path | None,
        crops_dir: Path | None,
        contours_dir: Path | None,
    ) -> tuple[Path, Path | None, Path | None, Path | None]:
        output_path = (segmented_dir / source_path.name) if segmented_dir is not None else source_path
        mask_path = (masks_dir / f"{source_path.stem}_mask.png") if masks_dir is not None else None
        crop_path = (crops_dir / f"{source_path.stem}_crop.jpg") if crops_dir is not None else None
        contour_path = (contours_dir / f"{source_path.stem}_contour.json") if contours_dir is not None else None
        return output_path, mask_path, crop_path, contour_path

    def _segment_single_image(
        self,
        *,
        source_path: Path,
        output_path: Path,
        mask_path: Path | None,
        crop_path: Path | None,
        contour_path: Path | None,
    ) -> dict[str, Any]:
        try:
            with Image.open(source_path) as raw:
                normalized = ImageOps.exif_transpose(raw).convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError):
            self._copy_original(source_path, output_path)
            if crop_path is not None:
                self._copy_original(source_path, crop_path)
            if mask_path is not None:
                self._save_full_mask(mask_path, (1, 1))
                legacy_mask_path = mask_path.parent.parent / "segmentation_masks" / mask_path.name
                self._save_full_mask(legacy_mask_path, (1, 1))
            if contour_path is not None:
                write_json(contour_path, {"filename": source_path.name, "status": "fallback_original", "contour": None})
            return {
                "filename": source_path.name,
                "source_path": str(source_path),
                "output_path": str(output_path),
                "mask_path": str(mask_path) if mask_path else None,
                "crop_path": str(crop_path) if crop_path else None,
                "contour_path": str(contour_path) if contour_path else None,
                "status": "fallback_original",
                "reason": "unreadable_image",
                "bbox": None,
                "bbox_relative": None,
                "contour_points": None,
                "contour_points_relative": None,
                "component_score": None,
                "component_metrics": None,
                "object_metrics": None,
                "background_color": None,
            }

        analysis = self._resize_for_analysis(normalized)
        mask, method_used = self._build_candidate_mask(analysis)
        best_component = self._select_primary_component(mask)

        if best_component is None:
            self._copy_normalized_image(normalized, output_path)
            if crop_path is not None:
                self._copy_normalized_image(normalized, crop_path)
            if mask_path is not None:
                self._save_full_mask(mask_path, normalized.size)
                legacy_mask_path = mask_path.parent.parent / "segmentation_masks" / mask_path.name
                self._save_full_mask(legacy_mask_path, normalized.size)
            if contour_path is not None:
                width, height = normalized.size
                contour_points = [
                    [0, 0],
                    [max(0, width - 1), 0],
                    [max(0, width - 1), max(0, height - 1)],
                    [0, max(0, height - 1)],
                ]
                write_json(
                    contour_path,
                    {
                        "filename": source_path.name,
                        "status": "fallback_original",
                        "contour": contour_points,
                        "reason": "no_component_detected",
                    },
                )
            return {
                "filename": source_path.name,
                "source_path": str(source_path),
                "output_path": str(output_path),
                "mask_path": str(mask_path) if mask_path else None,
                "crop_path": str(crop_path) if crop_path else None,
                "contour_path": str(contour_path) if contour_path else None,
                "status": "fallback_original",
                "reason": "no_component_detected",
                "segmentation_method": method_used,
                "segmentation_success": False,
                "foreground_ratio": 1.0,
                "bbox": None,
                "bbox_relative": None,
                "contour_points": None,
                "contour_points_relative": None,
                "component_score": None,
                "component_metrics": None,
                "object_metrics": None,
                "background_color": None,
            }

        bbox = self._analysis_bbox_to_original(best_component.bbox, normalized.size, analysis.size)
        padded_bbox = self._apply_padding(bbox, normalized.size)
        background_color = self._estimate_background_color(normalized)
        segmented = Image.new("RGB", normalized.size, background_color)
        segmented.paste(normalized.crop(padded_bbox), (padded_bbox[0], padded_bbox[1]))
        self._copy_normalized_image(segmented, output_path)
        crop_image = normalized.crop(padded_bbox)
        if crop_path is not None:
            self._copy_normalized_image(crop_image, crop_path)

        if mask_path is not None:
            self._save_bbox_mask(mask_path, normalized.size, padded_bbox)
            legacy_mask_path = mask_path.parent.parent / "segmentation_masks" / mask_path.name
            self._save_bbox_mask(legacy_mask_path, normalized.size, padded_bbox)

        width, height = normalized.size
        left, top, right, bottom = padded_bbox
        contour_points = self._analysis_points_to_original(best_component.corners, normalized.size, analysis.size)
        contour_points_relative = [
            [
                round(point_x / float(max(1, width)), 6),
                round(point_y / float(max(1, height)), 6),
            ]
            for point_x, point_y in contour_points
        ]
        if contour_path is not None:
            write_json(
                contour_path,
                {
                    "filename": source_path.name,
                    "status": "segmented",
                    "reason": "component_selected",
                    "bbox": [left, top, right, bottom],
                    "contour_points": [list(point) for point in contour_points],
                    "contour_points_relative": contour_points_relative,
                },
            )
        bbox_relative = (
            round(left / float(max(1, width)), 6),
            round(top / float(max(1, height)), 6),
            round(right / float(max(1, width)), 6),
            round(bottom / float(max(1, height)), 6),
        )
        object_aspect_ratio = (right - left) / float(max(1, bottom - top))
        object_metrics = {
            "sharpness": round(self._estimate_region_sharpness(crop_image), 6),
            "visible_area_ratio": round(best_component.area_ratio, 6),
            "rectangularity": round(best_component.fill_ratio, 6),
            "tilt_degrees": round(self._estimate_tilt_degrees(tuple(contour_points)), 4),
            "aspect_ratio": round(object_aspect_ratio, 6),
            "contour_area_ratio": round(self._polygon_area(tuple(contour_points)) / float(max(1, width * height)), 6),
        }

        return {
            "filename": source_path.name,
            "source_path": str(source_path),
            "output_path": str(output_path),
            "mask_path": str(mask_path) if mask_path else None,
            "crop_path": str(crop_path) if crop_path else None,
            "contour_path": str(contour_path) if contour_path else None,
            "status": "segmented",
            "reason": "component_selected",
            "segmentation_method": method_used,
            "segmentation_success": True,
            "foreground_ratio": round(best_component.area_ratio, 6),
            "bbox": [left, top, right, bottom],
            "bbox_relative": list(bbox_relative),
            "contour_points": [list(point) for point in contour_points],
            "contour_points_relative": contour_points_relative,
            "component_score": round(best_component.score, 6),
            "component_metrics": {
                "area_ratio": round(best_component.area_ratio, 6),
                "fill_ratio": round(best_component.fill_ratio, 6),
                "aspect_ratio": round(best_component.aspect_ratio, 6),
                "center_offset": round(best_component.center_offset, 6),
                "border_touches": best_component.border_touches,
                "pixel_count": best_component.pixel_count,
            },
            "object_metrics": object_metrics,
            "background_color": list(background_color),
        }

    def _build_candidate_mask(self, image: Image.Image) -> tuple[Image.Image, str]:
        if cv2 is not None and np is not None:
            mask = self._build_grabcut_mask(image)
            if mask is not None:
                return mask, "grabcut"
        return self._build_threshold_mask(image), "threshold_contours"

    def _build_grabcut_mask(self, image: Image.Image) -> Image.Image | None:
        if cv2 is None or np is None:
            return None
        try:
            rgb = np.asarray(image.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            if h < 8 or w < 8:
                return None
            rect = (max(1, int(w * 0.08)), max(1, int(h * 0.08)), max(4, int(w * 0.84)), max(4, int(h * 0.84)))
            gc_mask = np.zeros((h, w), np.uint8)
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(bgr, gc_mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
            binary = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
            if float(binary.mean()) <= 1.0:
                return None
            return Image.fromarray(binary).convert("L")
        except Exception:
            return None

    def _build_threshold_mask(self, image: Image.Image) -> Image.Image:
        grayscale = image.convert("L")
        saturation = image.convert("HSV").split()[1]
        edges = grayscale.filter(ImageFilter.FIND_EDGES)

        bg_color = self._estimate_background_color(image)
        r, g, b = image.split()
        diff_r = ImageChops.difference(r, Image.new("L", image.size, color=bg_color[0]))
        diff_g = ImageChops.difference(g, Image.new("L", image.size, color=bg_color[1]))
        diff_b = ImageChops.difference(b, Image.new("L", image.size, color=bg_color[2]))
        diff = ImageChops.lighter(ImageChops.lighter(diff_r, diff_g), diff_b)

        diff_stats = ImageStat.Stat(diff)
        diff_threshold = int(
            max(
                18,
                min(
                    128,
                    round(
                        float(diff_stats.mean[0] if diff_stats.mean else 0.0)
                        + float(diff_stats.stddev[0] if diff_stats.stddev else 0.0) * 0.8
                    ),
                ),
            )
        )
        diff_mask = diff.point(lambda value: 255 if value >= diff_threshold else 0)

        sat_stats = ImageStat.Stat(saturation)
        sat_threshold = int(
            max(
                26,
                min(
                    150,
                    round(
                        float(sat_stats.mean[0] if sat_stats.mean else 0.0)
                        + float(sat_stats.stddev[0] if sat_stats.stddev else 0.0) * 0.75
                    ),
                ),
            )
        )
        sat_mask = saturation.point(lambda value: 255 if value >= sat_threshold else 0)

        edge_stats = ImageStat.Stat(edges)
        edge_threshold = int(
            max(
                18,
                min(
                    150,
                    round(
                        float(edge_stats.mean[0] if edge_stats.mean else 0.0)
                        + float(edge_stats.stddev[0] if edge_stats.stddev else 0.0) * 0.8
                    ),
                ),
            )
        )
        edge_mask = edges.point(lambda value: 255 if value >= edge_threshold else 0)

        combined = ImageChops.lighter(diff_mask, sat_mask)
        edge_guided = ImageChops.multiply(diff_mask, edge_mask).point(lambda value: 255 if value >= 20 else 0)
        combined = ImageChops.lighter(combined, edge_guided)
        combined = combined.filter(ImageFilter.MedianFilter(size=3))
        combined = combined.filter(ImageFilter.MaxFilter(size=5))
        combined = combined.filter(ImageFilter.MinFilter(size=3))
        return combined.point(lambda value: 255 if value >= 128 else 0)

    def _select_primary_component(self, mask: Image.Image) -> _ComponentCandidate | None:
        width, height = mask.size
        if width <= 0 or height <= 0:
            return None

        pixels = mask.load()
        if pixels is None:
            return None

        visited = bytearray(width * height)
        best: _ComponentCandidate | None = None

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
                points: list[tuple[int, int]] = []
                left = right = x
                top = bottom = y

                while stack:
                    cx, cy = stack.pop()
                    pixel_count += 1
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

                if pixel_count < 32:
                    continue

                bbox = (left, top, right + 1, bottom + 1)
                bbox_width = max(1, bbox[2] - bbox[0])
                bbox_height = max(1, bbox[3] - bbox[1])
                bbox_area = float(bbox_width * bbox_height)
                total_area = float(max(1, width * height))
                area_ratio = bbox_area / total_area
                if area_ratio < self.settings.min_component_area_ratio:
                    continue
                if area_ratio > self.settings.max_component_area_ratio:
                    continue

                fill_ratio = pixel_count / max(1.0, bbox_area)
                if fill_ratio < (self.settings.min_component_fill_ratio * 0.45):
                    continue

                aspect_ratio = bbox_width / float(max(1, bbox_height))
                center_offset = self._center_offset(bbox, (width, height))
                border_touches = self._count_border_touches(bbox, (width, height))
                corners = self._estimate_component_corners(points)
                if corners is None:
                    continue
                score = self._score_component(
                    area_ratio=area_ratio,
                    fill_ratio=fill_ratio,
                    aspect_ratio=aspect_ratio,
                    center_offset=center_offset,
                    border_touches=border_touches,
                )
                if score < self.settings.min_component_score:
                    continue

                candidate = _ComponentCandidate(
                    pixel_count=pixel_count,
                    bbox=bbox,
                    corners=corners,
                    area_ratio=area_ratio,
                    fill_ratio=fill_ratio,
                    aspect_ratio=aspect_ratio,
                    center_offset=center_offset,
                    border_touches=border_touches,
                    score=score,
                )
                if best is None or candidate.score > best.score:
                    best = candidate

        return best

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

    def _score_component(
        self,
        *,
        area_ratio: float,
        fill_ratio: float,
        aspect_ratio: float,
        center_offset: float,
        border_touches: int,
    ) -> float:
        area_score = max(0.0, min(1.0, 1.0 - (abs(area_ratio - 0.12) / 0.26)))
        fill_score = max(
            0.0,
            min(
                1.0,
                (fill_ratio - self.settings.min_component_fill_ratio * 0.5)
                / max(0.1, (0.72 - self.settings.min_component_fill_ratio * 0.5)),
            ),
        )
        aspect_score = max(
            0.0,
            min(
                1.0,
                1.0
                - (
                    abs(aspect_ratio - self.settings.expected_aspect_ratio)
                    / max(0.2, self.settings.aspect_tolerance)
                ),
            ),
        )
        center_score = max(0.0, min(1.0, 1.0 - (center_offset / 0.9)))
        border_score = max(0.0, min(1.0, 1.0 - (border_touches / 3.0)))

        return (
            area_score * 0.30
            + center_score * 0.28
            + fill_score * 0.20
            + aspect_score * 0.17
            + border_score * 0.05
        )

    @staticmethod
    def _center_offset(
        bbox: tuple[int, int, int, int],
        image_size: tuple[int, int],
    ) -> float:
        width, height = image_size
        if width <= 0 or height <= 0:
            return 1.0
        left, top, right, bottom = bbox
        center_x = (left + right) * 0.5
        center_y = (top + bottom) * 0.5
        dx = (center_x - width * 0.5) / max(width * 0.5, 1.0)
        dy = (center_y - height * 0.55) / max(height * 0.55, 1.0)
        return min(1.5, math.sqrt(dx * dx + dy * dy))

    @staticmethod
    def _count_border_touches(
        bbox: tuple[int, int, int, int],
        image_size: tuple[int, int],
        *,
        margin: int = 1,
    ) -> int:
        left, top, right, bottom = bbox
        width, height = image_size
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

    def _analysis_bbox_to_original(
        self,
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
        return (
            max(0, min(width, left)),
            max(0, min(height, top)),
            max(0, min(width, right)),
            max(0, min(height, bottom)),
        )

    def _analysis_points_to_original(
        self,
        points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
        original_size: tuple[int, int],
        analysis_size: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        width, height = original_size
        analysis_w, analysis_h = analysis_size
        scale_x = width / float(max(1, analysis_w))
        scale_y = height / float(max(1, analysis_h))
        mapped: list[tuple[int, int]] = []
        for px, py in points:
            mapped_x = int(round(px * scale_x))
            mapped_y = int(round(py * scale_y))
            mapped.append(
                (
                    max(0, min(width - 1, mapped_x)),
                    max(0, min(height - 1, mapped_y)),
                )
            )
        return (mapped[0], mapped[1], mapped[2], mapped[3])

    def _apply_padding(
        self,
        bbox: tuple[int, int, int, int],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        width, height = image_size
        left, top, right, bottom = bbox
        pad = int(round(max(width, height) * self.settings.mask_padding_ratio))
        left = max(0, left - pad)
        top = max(0, top - pad)
        right = min(width, right + pad)
        bottom = min(height, bottom + pad)
        if right - left < 24:
            center_x = (left + right) * 0.5
            half = 12
            left = max(0, int(round(center_x - half)))
            right = min(width, int(round(center_x + half)))
        if bottom - top < 24:
            center_y = (top + bottom) * 0.5
            half = 12
            top = max(0, int(round(center_y - half)))
            bottom = min(height, int(round(center_y + half)))
        return left, top, right, bottom

    def _resize_for_analysis(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= self.settings.analysis_max_width:
            return image
        scale = self.settings.analysis_max_width / float(max(1, width))
        target_height = max(64, int(round(height * scale)))
        return image.resize((self.settings.analysis_max_width, target_height), Image.Resampling.BILINEAR)

    @staticmethod
    def _estimate_background_color(image: Image.Image) -> tuple[int, int, int]:
        width, height = image.size
        if width <= 0 or height <= 0:
            return 230, 230, 230
        border = max(2, int(round(min(width, height) * 0.06)))

        top = image.crop((0, 0, width, border))
        bottom = image.crop((0, height - border, width, height))
        left = image.crop((0, 0, border, height))
        right = image.crop((width - border, 0, width, height))
        strips = [top, bottom, left, right]

        channel_values: list[list[float]] = [[], [], []]
        for strip in strips:
            mean = ImageStat.Stat(strip).mean[:3]
            for channel_index in range(3):
                channel_values[channel_index].append(float(mean[channel_index]))
        return tuple(
            int(round(sorted(values)[len(values) // 2])) if values else 230
            for values in channel_values
        )

    @staticmethod
    def _estimate_region_sharpness(region: Image.Image) -> float:
        if region.width <= 2 or region.height <= 2:
            return 0.0
        edges = region.convert("L").filter(ImageFilter.FIND_EDGES)
        stats = ImageStat.Stat(edges)
        variance = float((stats.var[0] if stats.var else 0.0))
        return variance / (255.0 * 255.0)

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
    def _polygon_area(
        corners: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
    ) -> float:
        area = 0.0
        points = [corners[0], corners[1], corners[2], corners[3]]
        for index in range(len(points)):
            x1, y1 = points[index]
            x2, y2 = points[(index + 1) % len(points)]
            area += float(x1 * y2 - x2 * y1)
        return abs(area) * 0.5

    @staticmethod
    def _copy_original(source_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        try:
            output_path.hardlink_to(source_path)
        except OSError:
            shutil.copy2(source_path, output_path)

    @staticmethod
    def _copy_normalized_image(image: Image.Image, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    @staticmethod
    def _save_bbox_mask(mask_path: Path, image_size: tuple[int, int], bbox: tuple[int, int, int, int]) -> None:
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", image_size, color=0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(bbox, fill=255)
        mask.save(mask_path, format="PNG")

    @staticmethod
    def _save_full_mask(mask_path: Path, image_size: tuple[int, int]) -> None:
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", image_size, color=255)
        mask.save(mask_path, format="PNG")

    def _settings_to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.settings.enabled,
            "analysis_max_width": self.settings.analysis_max_width,
            "min_component_area_ratio": self.settings.min_component_area_ratio,
            "max_component_area_ratio": self.settings.max_component_area_ratio,
            "min_component_fill_ratio": self.settings.min_component_fill_ratio,
            "expected_aspect_ratio": self.settings.expected_aspect_ratio,
            "aspect_tolerance": self.settings.aspect_tolerance,
            "mask_padding_ratio": self.settings.mask_padding_ratio,
            "min_component_score": self.settings.min_component_score,
            "min_segmented_images": self.settings.min_segmented_images,
            "min_segmented_ratio": self.settings.min_segmented_ratio,
            "block_on_low_success": self.settings.block_on_low_success,
        }
