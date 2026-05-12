from __future__ import annotations

from collections import deque
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from .artifacts import write_json


@dataclass(frozen=True)
class InputImageSelectionSettings:
    enabled: bool = True
    min_images_required: int = 6
    max_images: int = 60
    target_keep_ratio: float = 0.75
    min_quality_score: float = 0.35
    quality_weight: float = 0.65
    diversity_weight: float = 0.35
    near_duplicate_hamming: int = 3
    diversity_min_hamming: int = 8

    @classmethod
    def from_settings(cls, settings: Any) -> "InputImageSelectionSettings":
        quality_weight = float(getattr(settings, "image_selection_quality_weight", 0.65))
        diversity_weight = float(getattr(settings, "image_selection_diversity_weight", 0.35))
        if quality_weight <= 0 and diversity_weight <= 0:
            quality_weight, diversity_weight = 0.65, 0.35

        weight_sum = max(quality_weight + diversity_weight, 1e-6)
        quality_weight = quality_weight / weight_sum
        diversity_weight = diversity_weight / weight_sum

        near_duplicate_hamming = max(0, int(getattr(settings, "image_selection_near_duplicate_hamming", 3)))
        diversity_min_hamming = max(0, int(getattr(settings, "image_selection_diversity_min_hamming", 8)))
        if diversity_min_hamming <= near_duplicate_hamming:
            diversity_min_hamming = near_duplicate_hamming + 1

        return cls(
            enabled=bool(getattr(settings, "image_selection_enabled", True)),
            min_images_required=max(1, int(getattr(settings, "image_selection_min_images_required", 6))),
            max_images=max(1, int(getattr(settings, "image_selection_max_images", 60))),
            target_keep_ratio=max(0.1, min(1.0, float(getattr(settings, "image_selection_target_keep_ratio", 0.75)))),
            min_quality_score=max(0.0, min(1.0, float(getattr(settings, "image_selection_min_quality_score", 0.35)))),
            quality_weight=quality_weight,
            diversity_weight=diversity_weight,
            near_duplicate_hamming=near_duplicate_hamming,
            diversity_min_hamming=diversity_min_hamming,
        )


@dataclass
class _ImageSelectionCandidate:
    path: Path
    status_from_validation: str
    warning_reasons: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0
    pixel_count: int = 0
    brightness: float = 0.0
    sharpness: float = 0.0
    quality_score: float = 0.0
    perceptual_hash: int | None = None
    selected_rank: int | None = None
    selection_state: str = "candidate"
    selection_reason: str | None = None

    @property
    def filename(self) -> str:
        return self.path.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "path": str(self.path),
            "status_from_validation": self.status_from_validation,
            "warning_reasons": list(self.warning_reasons),
            "quality_score": round(self.quality_score, 4),
            "perceptual_hash_hex": f"{self.perceptual_hash:016x}" if self.perceptual_hash is not None else None,
            "metrics": {
                "width": self.width,
                "height": self.height,
                "pixel_count": self.pixel_count,
                "brightness": round(self.brightness, 4),
                "sharpness": round(self.sharpness, 4),
            },
            "selection": {
                "state": self.selection_state,
                "reason": self.selection_reason,
                "rank": self.selected_rank,
            },
        }


@dataclass(frozen=True)
class InputImageSelectionResult:
    allow_processing: bool
    selected_images: list[Path]
    report_path: Path | None
    summary: dict[str, Any]


class InputImageSelector:
    name = "input_image_selector"

    def __init__(
        self,
        selection_settings: InputImageSelectionSettings | None = None,
    ) -> None:
        self.selection_settings = selection_settings or InputImageSelectionSettings()

    @classmethod
    def from_settings(cls, settings: Any) -> "InputImageSelector":
        return cls(selection_settings=InputImageSelectionSettings.from_settings(settings))

    def select_images(
        self,
        validation_summary: dict[str, Any],
        candidate_images: list[Path],
        report_dir: Path | None = None,
    ) -> InputImageSelectionResult:
        candidates = self._build_candidates(validation_summary, candidate_images)
        total_candidates = len(candidates)

        if not self.selection_settings.enabled:
            for rank, candidate in enumerate(self._sort_candidates_by_filename(candidates), start=1):
                candidate.selection_state = "selected"
                candidate.selection_reason = "selector_disabled"
                candidate.selected_rank = rank
            selected_images = [candidate.path for candidate in sorted(candidates, key=lambda item: item.selected_rank or 0)]
            summary = self._build_summary(
                candidates,
                selected_images,
                blocking_reasons=[],
                target_count=len(selected_images),
            )
            report_path = write_json(report_dir / "input_image_selection_report.json", summary) if report_dir else None
            return InputImageSelectionResult(
                allow_processing=True,
                selected_images=selected_images,
                report_path=report_path,
                summary=summary,
            )

        deduped_candidates, dedup_discarded = self._deduplicate_candidates(candidates)
        target_count = self._compute_target_count(total_candidates, len(deduped_candidates))

        selected_candidates, remaining_candidates = self._select_balanced_subset(deduped_candidates, target_count)

        min_required = self.selection_settings.min_images_required
        if len(selected_candidates) < min_required:
            fill_candidates = deque(
                self._sort_candidates_by_quality(
                    [candidate for candidate in remaining_candidates if candidate.selection_state != "discarded"]
                )
            )
            while fill_candidates and len(selected_candidates) < min_required:
                candidate = fill_candidates.popleft()
                remaining_candidates.remove(candidate)
                self._mark_selected(
                    candidate,
                    rank=len(selected_candidates) + 1,
                    reason="selected_to_meet_minimum",
                )
                selected_candidates.append(candidate)

        self._mark_remaining_candidates(selected_candidates, remaining_candidates)

        all_discarded = [
            candidate
            for candidate in (dedup_discarded + remaining_candidates)
            if candidate.selection_state == "discarded"
        ]
        selected_images = [candidate.path for candidate in sorted(selected_candidates, key=lambda item: item.selected_rank or 0)]

        blocking_reasons: list[str] = []
        if not selected_images:
            blocking_reasons.append("no_images_selected")
        if len(selected_images) < min_required:
            blocking_reasons.append("insufficient_selected_images")

        summary = self._build_summary(
            candidates,
            selected_images,
            blocking_reasons=blocking_reasons,
            target_count=target_count,
            discarded_candidates=all_discarded,
        )

        report_path = write_json(report_dir / "input_image_selection_report.json", summary) if report_dir else None
        return InputImageSelectionResult(
            allow_processing=not blocking_reasons,
            selected_images=selected_images,
            report_path=report_path,
            summary=summary,
        )

    def stage_selected_images(self, selected_images: list[Path], target_dir: Path) -> list[Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        staged_paths: list[Path] = []

        for index, source_path in enumerate(selected_images, start=1):
            target_path = target_dir / f"{index:03d}_{source_path.name}"
            if target_path.exists():
                target_path.unlink(missing_ok=True)

            try:
                target_path.hardlink_to(source_path)
            except OSError:
                shutil.copy2(source_path, target_path)

            staged_paths.append(target_path)

        return staged_paths

    def _build_candidates(
        self,
        validation_summary: dict[str, Any],
        candidate_images: list[Path],
    ) -> list[_ImageSelectionCandidate]:
        metadata_by_path: dict[str, dict[str, Any]] = {}
        for item in validation_summary.get("images", []):
            if not isinstance(item, dict):
                continue
            path_value = str(item.get("path") or "").strip()
            if not path_value:
                continue
            metadata_by_path[path_value] = item

        candidates: list[_ImageSelectionCandidate] = []
        for candidate_path in sorted(candidate_images, key=lambda item: item.name.lower()):
            metadata = metadata_by_path.get(str(candidate_path), {})
            metrics = metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {}
            status = str(metadata.get("status") or "apta").strip().lower() or "apta"
            warning_reasons = [str(item) for item in metadata.get("warning_reasons", []) if str(item).strip()]

            candidate = _ImageSelectionCandidate(
                path=candidate_path,
                status_from_validation=status,
                warning_reasons=warning_reasons,
                width=int(metrics.get("width") or 0),
                height=int(metrics.get("height") or 0),
                pixel_count=int(metrics.get("pixel_count") or 0),
                brightness=float(metrics.get("brightness") or 0.0),
                sharpness=float(metrics.get("sharpness") or 0.0),
            )
            candidate.perceptual_hash = self._compute_dhash_from_path(candidate.path)
            candidates.append(candidate)

        max_pixels = max((candidate.pixel_count for candidate in candidates), default=1)
        for candidate in candidates:
            candidate.quality_score = self._compute_quality_score(candidate, max_pixels)

        return candidates

    def _deduplicate_candidates(
        self,
        candidates: list[_ImageSelectionCandidate],
    ) -> tuple[list[_ImageSelectionCandidate], list[_ImageSelectionCandidate]]:
        deduped: list[_ImageSelectionCandidate] = []
        discarded: list[_ImageSelectionCandidate] = []

        ordered = self._sort_candidates_by_quality(candidates)
        for candidate in ordered:
            nearest = self._nearest_candidate(candidate, deduped)
            if nearest is not None and nearest[1] <= self.selection_settings.near_duplicate_hamming:
                self._mark_discarded(candidate, f"near_duplicate_of:{nearest[0].filename}")
                discarded.append(candidate)
                continue
            deduped.append(candidate)

        return deduped, discarded

    def _select_balanced_subset(
        self,
        deduped_candidates: list[_ImageSelectionCandidate],
        target_count: int,
    ) -> tuple[list[_ImageSelectionCandidate], list[_ImageSelectionCandidate]]:
        if not deduped_candidates or target_count <= 0:
            return [], list(deduped_candidates)

        selected: list[_ImageSelectionCandidate] = []
        remaining = self._sort_candidates_by_quality(deduped_candidates)

        seed = remaining.pop(0)
        self._mark_selected(seed, rank=1, reason="best_quality_seed")
        selected.append(seed)

        while remaining and len(selected) < target_count:
            best_candidate: _ImageSelectionCandidate | None = None
            best_value = float("-inf")
            best_min_hamming = 64

            for candidate in remaining:
                min_hamming = self._minimum_hamming_to_selected(candidate, selected)
                diversity_score = min_hamming / 64.0
                combined_score = (
                    self.selection_settings.quality_weight * candidate.quality_score
                    + self.selection_settings.diversity_weight * diversity_score
                )

                if min_hamming < self.selection_settings.diversity_min_hamming:
                    combined_score -= 0.08
                if candidate.quality_score < self.selection_settings.min_quality_score:
                    combined_score -= 0.12

                if combined_score > best_value:
                    best_value = combined_score
                    best_candidate = candidate
                    best_min_hamming = min_hamming

            if best_candidate is None:
                break

            if (
                best_candidate.quality_score < self.selection_settings.min_quality_score
                and len(selected) >= self.selection_settings.min_images_required
            ):
                break

            remaining.remove(best_candidate)
            self._mark_selected(
                best_candidate,
                rank=len(selected) + 1,
                reason=f"quality_diversity_balance:min_hamming={best_min_hamming}",
            )
            selected.append(best_candidate)

        return selected, remaining

    def _mark_remaining_candidates(
        self,
        selected_candidates: list[_ImageSelectionCandidate],
        remaining_candidates: list[_ImageSelectionCandidate],
    ) -> None:
        for candidate in remaining_candidates:
            if candidate.selection_state == "discarded":
                continue

            min_hamming = self._minimum_hamming_to_selected(candidate, selected_candidates)
            if candidate.quality_score < self.selection_settings.min_quality_score:
                reason = "quality_below_threshold"
            elif min_hamming < self.selection_settings.diversity_min_hamming:
                reason = "low_diversity_lower_priority"
            else:
                reason = "selection_limit_reached"

            self._mark_discarded(candidate, reason)

    def _compute_target_count(self, total_candidates: int, available_candidates: int) -> int:
        if available_candidates <= 0 or total_candidates <= 0:
            return 0

        minimum = self.selection_settings.min_images_required
        maximum = min(self.selection_settings.max_images, available_candidates)
        ratio_target = max(1, int(round(total_candidates * self.selection_settings.target_keep_ratio)))
        target = max(minimum, ratio_target)
        target = min(target, maximum)
        return max(0, target)

    def _build_summary(
        self,
        candidates: list[_ImageSelectionCandidate],
        selected_images: list[Path],
        blocking_reasons: list[str],
        target_count: int,
        discarded_candidates: list[_ImageSelectionCandidate] | None = None,
    ) -> dict[str, Any]:
        total_candidates = len(candidates)
        selected_count = len(selected_images)
        discarded_count = total_candidates - selected_count
        reduction_ratio = ((total_candidates - selected_count) / total_candidates) if total_candidates else 0.0

        discarded_reason_counts: dict[str, int] = {}
        for candidate in (discarded_candidates or []):
            reason = str(candidate.selection_reason or "unspecified")
            discarded_reason_counts[reason] = discarded_reason_counts.get(reason, 0) + 1

        selected_records: list[_ImageSelectionCandidate] = []
        discarded_records: list[_ImageSelectionCandidate] = []
        for candidate in candidates:
            if candidate.selection_state == "selected":
                selected_records.append(candidate)
            elif candidate.selection_state == "discarded":
                discarded_records.append(candidate)

        selected_records.sort(key=lambda item: item.selected_rank or 0)
        discarded_records = self._sort_candidates_by_filename(discarded_records)

        return {
            "selector": self.name,
            "enabled": self.selection_settings.enabled,
            "allow_processing": not blocking_reasons,
            "candidate_images": total_candidates,
            "selected_images": selected_count,
            "discarded_images": discarded_count,
            "target_selected_images": target_count,
            "min_images_required": self.selection_settings.min_images_required,
            "blocking_reasons": list(blocking_reasons),
            "discarded_reason_counts": discarded_reason_counts,
            "comparison": {
                "before_selection_count": total_candidates,
                "after_selection_count": selected_count,
                "removed_count": total_candidates - selected_count,
                "reduction_ratio": round(reduction_ratio, 4),
                "estimated_processing_load_reduction_pct": round(reduction_ratio * 100.0, 2),
            },
            "selected": [candidate.to_dict() for candidate in selected_records],
            "discarded": [candidate.to_dict() for candidate in discarded_records],
            "candidates": [candidate.to_dict() for candidate in self._sort_candidates_by_filename(candidates)],
            "thresholds": self._settings_to_dict(),
        }

    def _compute_quality_score(self, candidate: _ImageSelectionCandidate, max_pixels: int) -> float:
        sharpness_score = self._clamp(candidate.sharpness, 0.0, 1.0)
        exposure_score = 1.0 - min(1.0, abs(candidate.brightness - 0.5) / 0.5)
        resolution_score = self._clamp(candidate.pixel_count / max(max_pixels, 1), 0.0, 1.0)

        warning_penalty = min(0.25, 0.03 * len(candidate.warning_reasons))
        status_penalty = 0.06 if candidate.status_from_validation == "advertida" else 0.0

        raw_score = (
            0.55 * sharpness_score
            + 0.25 * exposure_score
            + 0.20 * resolution_score
            - warning_penalty
            - status_penalty
        )
        return self._clamp(raw_score, 0.0, 1.0)

    def _minimum_hamming_to_selected(
        self,
        candidate: _ImageSelectionCandidate,
        selected_candidates: list[_ImageSelectionCandidate],
    ) -> int:
        if not selected_candidates:
            return 64
        if candidate.perceptual_hash is None:
            return 64

        distances = [
            self._hamming_distance(candidate.perceptual_hash, selected.perceptual_hash)
            for selected in selected_candidates
            if selected.perceptual_hash is not None
        ]
        return min(distances) if distances else 64

    def _nearest_candidate(
        self,
        candidate: _ImageSelectionCandidate,
        others: list[_ImageSelectionCandidate],
    ) -> tuple[_ImageSelectionCandidate, int] | None:
        if candidate.perceptual_hash is None:
            return None

        nearest: tuple[_ImageSelectionCandidate, int] | None = None
        for other in others:
            if other.perceptual_hash is None:
                continue
            distance = self._hamming_distance(candidate.perceptual_hash, other.perceptual_hash)
            if nearest is None or distance < nearest[1]:
                nearest = (other, distance)
        return nearest

    @staticmethod
    def _mark_selected(candidate: _ImageSelectionCandidate, rank: int, reason: str) -> None:
        candidate.selection_state = "selected"
        candidate.selected_rank = rank
        candidate.selection_reason = reason

    @staticmethod
    def _mark_discarded(candidate: _ImageSelectionCandidate, reason: str) -> None:
        candidate.selection_state = "discarded"
        candidate.selection_reason = reason
        candidate.selected_rank = None

    def _settings_to_dict(self) -> dict[str, Any]:
        settings = self.selection_settings
        return {
            "enabled": settings.enabled,
            "min_images_required": settings.min_images_required,
            "max_images": settings.max_images,
            "target_keep_ratio": settings.target_keep_ratio,
            "min_quality_score": settings.min_quality_score,
            "quality_weight": settings.quality_weight,
            "diversity_weight": settings.diversity_weight,
            "near_duplicate_hamming": settings.near_duplicate_hamming,
            "diversity_min_hamming": settings.diversity_min_hamming,
        }

    @staticmethod
    def _compute_dhash_from_path(path: Path, hash_size: int = 8) -> int | None:
        try:
            with Image.open(path) as image:
                grayscale = ImageOps.exif_transpose(image).convert("L")
        except (OSError, UnidentifiedImageError, ValueError):
            return None

        resized = grayscale.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
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
    def _hamming_distance(left: int | None, right: int | None) -> int:
        if left is None or right is None:
            return 64
        return int((left ^ right).bit_count())

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _sort_candidates_by_filename(candidates: list[_ImageSelectionCandidate]) -> list[_ImageSelectionCandidate]:
        return sorted(candidates, key=lambda item: item.filename.lower())

    @staticmethod
    def _sort_candidates_by_quality(candidates: list[_ImageSelectionCandidate]) -> list[_ImageSelectionCandidate]:
        return sorted(candidates, key=lambda item: (-item.quality_score, item.filename.lower()))
