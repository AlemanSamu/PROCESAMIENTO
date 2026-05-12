from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

from .artifacts import write_json


DEFAULT_ALLOWED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


@dataclass(frozen=True)
class InputImageValidationSettings:
    enabled: bool = True
    min_images_required: int = 6
    min_width: int = 640
    min_height: int = 480
    min_pixels: int = 640 * 480
    min_sharpness_warning: float = 0.06
    min_sharpness_reject: float = 0.04
    min_brightness: float = 0.15
    max_brightness: float = 0.9
    exposure_warn_margin: float = 0.07
    near_duplicate_warn_distance: int = 6
    near_duplicate_reject_distance: int = 2
    coverage_min_unique_ratio: float = 0.55
    coverage_min_median_hamming: int = 8
    coverage_max_neighbor_similarity_ratio: float = 0.7
    block_on_coverage_failure: bool = False

    @classmethod
    def from_settings(cls, settings: Any) -> "InputImageValidationSettings":
        warn_distance = int(getattr(settings, "image_validation_near_duplicate_warn_hamming", 6))
        reject_distance = int(getattr(settings, "image_validation_near_duplicate_reject_hamming", 2))
        if warn_distance < reject_distance:
            warn_distance, reject_distance = reject_distance, warn_distance

        return cls(
            enabled=bool(getattr(settings, "image_validation_enabled", True)),
            min_images_required=max(1, int(getattr(settings, "image_validation_min_images_required", 6))),
            min_width=max(1, int(getattr(settings, "image_validation_min_width", 640))),
            min_height=max(1, int(getattr(settings, "image_validation_min_height", 480))),
            min_pixels=max(1, int(getattr(settings, "image_validation_min_pixels", 640 * 480))),
            min_sharpness_warning=float(getattr(settings, "image_validation_min_sharpness_warn", 0.06)),
            min_sharpness_reject=float(getattr(settings, "image_validation_min_sharpness_reject", 0.04)),
            min_brightness=float(getattr(settings, "image_validation_min_brightness", 0.15)),
            max_brightness=float(getattr(settings, "image_validation_max_brightness", 0.9)),
            exposure_warn_margin=max(0.0, float(getattr(settings, "image_validation_exposure_warn_margin", 0.07))),
            near_duplicate_warn_distance=max(0, warn_distance),
            near_duplicate_reject_distance=max(0, reject_distance),
            coverage_min_unique_ratio=max(0.0, min(1.0, float(getattr(settings, "image_validation_coverage_min_unique_ratio", 0.55)))),
            coverage_min_median_hamming=max(0, int(getattr(settings, "image_validation_coverage_min_median_hamming", 8))),
            coverage_max_neighbor_similarity_ratio=max(
                0.0,
                min(1.0, float(getattr(settings, "image_validation_coverage_max_neighbor_similarity_ratio", 0.7))),
            ),
            block_on_coverage_failure=bool(getattr(settings, "image_validation_block_on_low_coverage", False)),
        )


@dataclass
class _ImageValidationRecord:
    path: Path
    extension: str
    sha256: str
    width: int = 0
    height: int = 0
    pixel_count: int = 0
    brightness: float = 0.0
    contrast: float = 0.0
    sharpness: float = 0.0
    perceptual_hash: int | None = None
    status: str = "apta"
    rejected_reasons: list[str] = field(default_factory=list)
    warning_reasons: list[str] = field(default_factory=list)
    duplicate_of: str | None = None

    def mark_rejected(self, reason: str) -> None:
        if reason not in self.rejected_reasons:
            self.rejected_reasons.append(reason)
        self.status = "rechazada"

    def mark_warning(self, reason: str) -> None:
        if reason not in self.warning_reasons:
            self.warning_reasons.append(reason)
        if self.status != "rechazada":
            self.status = "advertida"

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.path.name,
            "path": str(self.path),
            "status": self.status,
            "rejected_reasons": list(self.rejected_reasons),
            "warning_reasons": list(self.warning_reasons),
            "duplicate_of": self.duplicate_of,
            "metrics": {
                "width": self.width,
                "height": self.height,
                "pixel_count": self.pixel_count,
                "brightness": self.brightness,
                "contrast": self.contrast,
                "sharpness": self.sharpness,
                "sha256": self.sha256,
            },
        }


@dataclass(frozen=True)
class InputImageValidationResult:
    allow_processing: bool
    accepted_images: list[Path]
    report_path: Path | None
    summary: dict[str, Any]


class InputImageValidator:
    name = "input_image_validation"

    def __init__(
        self,
        validation_settings: InputImageValidationSettings | None = None,
        allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS,
    ) -> None:
        self.validation_settings = validation_settings or InputImageValidationSettings()
        self.allowed_extensions = tuple(sorted({ext.lower() for ext in allowed_extensions}))

    @classmethod
    def from_settings(cls, settings: Any) -> "InputImageValidator":
        configured_extensions = tuple(getattr(settings, "allowed_image_extensions", DEFAULT_ALLOWED_EXTENSIONS))
        return cls(
            validation_settings=InputImageValidationSettings.from_settings(settings),
            allowed_extensions=configured_extensions,
        )

    def validate_batch(
        self,
        images_dir: Path,
        report_dir: Path | None = None,
    ) -> InputImageValidationResult:
        records = self._collect_and_validate(images_dir)
        self._apply_duplicate_rules(records)
        coverage_summary = self._estimate_coverage(records)

        accepted_records = [record for record in records if record.status != "rechazada"]
        accepted_images = [record.path for record in accepted_records]

        total_images = len(records)
        valid_images = sum(1 for record in records if record.status == "apta")
        warning_images = sum(1 for record in records if record.status == "advertida")
        rejected_images = sum(1 for record in records if record.status == "rechazada")

        rejected_reason_counts = self._count_reasons(records, rejected=True)
        warning_reason_counts = self._count_reasons(records, rejected=False)

        batch_warnings: list[str] = []
        if coverage_summary["possible_low_coverage"]:
            batch_warnings.append("possible_low_coverage")

        blocking_reasons: list[str] = []
        minimum_required = self.validation_settings.min_images_required
        if total_images < minimum_required:
            blocking_reasons.append("insufficient_total_images")
        if len(accepted_images) < minimum_required:
            blocking_reasons.append("insufficient_valid_images")
        if self.validation_settings.block_on_coverage_failure and coverage_summary["possible_low_coverage"]:
            blocking_reasons.append("coverage_below_threshold")

        allow_processing = (not self.validation_settings.enabled) or (not blocking_reasons)

        summary = {
            "validator": self.name,
            "enabled": self.validation_settings.enabled,
            "allow_processing": allow_processing,
            "total_images": total_images,
            "valid_images": valid_images,
            "warning_images": warning_images,
            "rejected_images": rejected_images,
            "accepted_images": len(accepted_images),
            "min_images_required": minimum_required,
            "blocking_reasons": blocking_reasons,
            "batch_warnings": batch_warnings,
            "rejected_reason_counts": rejected_reason_counts,
            "warning_reason_counts": warning_reason_counts,
            "coverage": coverage_summary,
            "images": [record.to_dict() for record in records],
            "thresholds": self._settings_to_dict(),
        }

        report_path: Path | None = None
        if report_dir is not None:
            report_path = write_json(
                report_dir / "input_image_validation_report.json",
                summary,
            )

        return InputImageValidationResult(
            allow_processing=allow_processing,
            accepted_images=accepted_images,
            report_path=report_path,
            summary=summary,
        )

    def stage_accepted_images(self, accepted_images: list[Path], target_dir: Path) -> list[Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        staged_paths: list[Path] = []

        for index, source_path in enumerate(accepted_images, start=1):
            target_path = target_dir / f"{index:03d}_{source_path.name}"
            if target_path.exists():
                target_path.unlink(missing_ok=True)

            try:
                target_path.hardlink_to(source_path)
            except OSError:
                shutil.copy2(source_path, target_path)

            staged_paths.append(target_path)

        return staged_paths

    def _collect_and_validate(self, images_dir: Path) -> list[_ImageValidationRecord]:
        if not images_dir.exists():
            return []

        candidates = sorted(
            [path for path in images_dir.iterdir() if path.is_file()],
            key=lambda path: path.name.lower(),
        )
        records: list[_ImageValidationRecord] = []

        for path in candidates:
            extension = path.suffix.lower()
            try:
                checksum = self._hash_file(path)
            except OSError:
                checksum = ""
            record = _ImageValidationRecord(path=path, extension=extension, sha256=checksum)

            if not checksum:
                record.mark_rejected("unreadable_image")
                records.append(record)
                continue

            if extension not in self.allowed_extensions:
                record.mark_rejected("unsupported_extension")
                records.append(record)
                continue

            try:
                with Image.open(path) as image:
                    normalized = ImageOps.exif_transpose(image).convert("RGB")
            except (OSError, UnidentifiedImageError, ValueError):
                record.mark_rejected("unreadable_image")
                records.append(record)
                continue

            grayscale = normalized.convert("L")
            stats = ImageStat.Stat(grayscale)
            edge_map = grayscale.filter(ImageFilter.FIND_EDGES)
            edge_stats = ImageStat.Stat(edge_map)

            record.width = int(normalized.width)
            record.height = int(normalized.height)
            record.pixel_count = record.width * record.height
            record.brightness = round((stats.mean[0] if stats.mean else 0.0) / 255.0, 4)
            record.contrast = round((stats.stddev[0] if stats.stddev else 0.0) / 255.0, 4)
            record.sharpness = round(
                min(
                    1.0,
                    (
                        ((edge_stats.mean[0] if edge_stats.mean else 0.0) / 255.0) * 0.7
                        + ((edge_stats.stddev[0] if edge_stats.stddev else 0.0) / 255.0) * 0.3
                    ),
                ),
                4,
            )
            record.perceptual_hash = self._compute_dhash(grayscale)

            self._apply_quality_thresholds(record)
            records.append(record)

        return records

    def _apply_quality_thresholds(self, record: _ImageValidationRecord) -> None:
        settings = self.validation_settings

        if (
            record.width < settings.min_width
            or record.height < settings.min_height
            or record.pixel_count < settings.min_pixels
        ):
            record.mark_rejected("low_resolution")

        if record.brightness < settings.min_brightness:
            record.mark_rejected("underexposed")
        elif record.brightness > settings.max_brightness:
            record.mark_rejected("overexposed")
        else:
            low_warn_limit = settings.min_brightness + settings.exposure_warn_margin
            high_warn_limit = settings.max_brightness - settings.exposure_warn_margin
            if record.brightness < low_warn_limit:
                record.mark_warning("near_underexposed")
            if record.brightness > high_warn_limit:
                record.mark_warning("near_overexposed")

        if record.sharpness < settings.min_sharpness_reject:
            record.mark_rejected("blurry")
        elif record.sharpness < settings.min_sharpness_warning:
            record.mark_warning("slightly_blurry")

    def _apply_duplicate_rules(self, records: list[_ImageValidationRecord]) -> None:
        seen_checksums: dict[str, _ImageValidationRecord] = {}
        canonical_records: list[_ImageValidationRecord] = []
        settings = self.validation_settings

        for record in records:
            if record.status == "rechazada":
                continue

            previous_exact = seen_checksums.get(record.sha256)
            if previous_exact is not None:
                record.duplicate_of = previous_exact.path.name
                record.mark_rejected("duplicate_exact")
                continue

            best_candidate: _ImageValidationRecord | None = None
            best_distance: int | None = None
            for candidate in canonical_records:
                if candidate.perceptual_hash is None or record.perceptual_hash is None:
                    continue
                distance = self._hamming_distance(candidate.perceptual_hash, record.perceptual_hash)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_candidate = candidate

            if best_distance is not None and best_candidate is not None:
                if best_distance <= settings.near_duplicate_reject_distance:
                    record.duplicate_of = best_candidate.path.name
                    record.mark_rejected("duplicate_near")
                    continue
                if best_distance <= settings.near_duplicate_warn_distance:
                    record.duplicate_of = best_candidate.path.name
                    record.mark_warning("similar_view")

            seen_checksums[record.sha256] = record
            canonical_records.append(record)

    def _estimate_coverage(self, records: list[_ImageValidationRecord]) -> dict[str, Any]:
        accepted = [record for record in records if record.status != "rechazada" and record.perceptual_hash is not None]
        if len(accepted) < 2:
            return {
                "possible_low_coverage": False,
                "unique_hash_ratio": 1.0 if accepted else 0.0,
                "median_nearest_hamming": 64,
                "neighbor_similarity_ratio": 0.0,
            }

        hashes = [record.perceptual_hash for record in accepted if record.perceptual_hash is not None]
        unique_hash_ratio = round(len(set(hashes)) / len(hashes), 4)

        nearest_distances: list[int] = []
        for index, current_hash in enumerate(hashes):
            distances = [
                self._hamming_distance(current_hash, other_hash)
                for other_index, other_hash in enumerate(hashes)
                if other_index != index
            ]
            if distances:
                nearest_distances.append(min(distances))

        median_nearest_hamming = int(median(nearest_distances)) if nearest_distances else 64

        similar_neighbor_pairs = 0
        ordered = sorted(accepted, key=lambda item: item.path.name.lower())
        for left, right in zip(ordered, ordered[1:]):
            if left.perceptual_hash is None or right.perceptual_hash is None:
                continue
            distance = self._hamming_distance(left.perceptual_hash, right.perceptual_hash)
            if distance <= self.validation_settings.near_duplicate_warn_distance:
                similar_neighbor_pairs += 1

        neighbor_total = max(1, len(ordered) - 1)
        neighbor_similarity_ratio = round(similar_neighbor_pairs / neighbor_total, 4)

        possible_low_coverage = (
            (
                unique_hash_ratio < self.validation_settings.coverage_min_unique_ratio
                and median_nearest_hamming < self.validation_settings.coverage_min_median_hamming
            )
            or neighbor_similarity_ratio > self.validation_settings.coverage_max_neighbor_similarity_ratio
        )

        return {
            "possible_low_coverage": possible_low_coverage,
            "unique_hash_ratio": unique_hash_ratio,
            "median_nearest_hamming": median_nearest_hamming,
            "neighbor_similarity_ratio": neighbor_similarity_ratio,
        }

    @staticmethod
    def _count_reasons(records: list[_ImageValidationRecord], *, rejected: bool) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            reasons = record.rejected_reasons if rejected else record.warning_reasons
            for reason in reasons:
                counts[reason] = counts.get(reason, 0) + 1
        return counts

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

    @staticmethod
    def _compute_dhash(grayscale_image: Image.Image, hash_size: int = 8) -> int:
        resized = grayscale_image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(resized.getdata())
        difference_bits = 0

        for row in range(hash_size):
            row_offset = row * (hash_size + 1)
            for col in range(hash_size):
                left_pixel = pixels[row_offset + col]
                right_pixel = pixels[row_offset + col + 1]
                difference_bits <<= 1
                if left_pixel > right_pixel:
                    difference_bits |= 1

        return difference_bits

    @staticmethod
    def _hamming_distance(left: int, right: int) -> int:
        return int((left ^ right).bit_count())

    def _settings_to_dict(self) -> dict[str, Any]:
        settings = self.validation_settings
        return {
            "enabled": settings.enabled,
            "min_images_required": settings.min_images_required,
            "min_width": settings.min_width,
            "min_height": settings.min_height,
            "min_pixels": settings.min_pixels,
            "min_sharpness_warning": settings.min_sharpness_warning,
            "min_sharpness_reject": settings.min_sharpness_reject,
            "min_brightness": settings.min_brightness,
            "max_brightness": settings.max_brightness,
            "exposure_warn_margin": settings.exposure_warn_margin,
            "near_duplicate_warn_distance": settings.near_duplicate_warn_distance,
            "near_duplicate_reject_distance": settings.near_duplicate_reject_distance,
            "coverage_min_unique_ratio": settings.coverage_min_unique_ratio,
            "coverage_min_median_hamming": settings.coverage_min_median_hamming,
            "coverage_max_neighbor_similarity_ratio": settings.coverage_max_neighbor_similarity_ratio,
            "block_on_coverage_failure": settings.block_on_coverage_failure,
            "allowed_extensions": list(self.allowed_extensions),
        }
