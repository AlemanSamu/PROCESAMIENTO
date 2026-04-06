from __future__ import annotations

import hashlib
import math
from pathlib import Path

try:  # OpenCV es opcional; el backend sigue funcionando sin esta dependencia.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - depende del entorno
    cv2 = None

try:  # pragma: no cover - depende del entorno
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - depende del entorno
    np = None

from PIL import Image, ImageOps

from app.core.errors import ProcessingError

from .artifacts import (
    FeatureMatch,
    FeaturePoint,
    ImageFeatures,
    PipelineStageResult,
    PreprocessedImage,
    write_json,
)


class FeatureMatcher:
    """Extrae caracteristicas reales cuando es posible y cae a sintesis si no hay soporte."""

    name = "feature_extraction_and_matching"
    _MAX_FEATURES = 32
    _PATCH_RADIUS = 2
    _MIN_REAL_FEATURES = 4

    def run(
        self,
        images: list[PreprocessedImage],
        work_dir: Path,
    ) -> tuple[list[ImageFeatures], list[FeatureMatch], PipelineStageResult]:
        if not images:
            raise ProcessingError("No hay imagenes preprocesadas para extraer caracteristicas.")

        feature_sets: list[ImageFeatures] = []
        descriptor_sets: list[list[tuple[float, ...]]] = []
        feature_payload: list[dict[str, object]] = []
        match_payload: list[dict[str, object]] = []
        matches: list[FeatureMatch] = []
        extraction_modes: list[str] = []
        detection_backends: list[str] = []
        matching_backends: list[str] = []

        for image in images:
            keypoints, descriptors, mode, backend = self._extract_features(image)
            feature_set = ImageFeatures(
                image=image,
                keypoints=keypoints,
                descriptor_seed=self._descriptor_seed(image, descriptors, backend),
                score=self._feature_score(image, keypoints),
            )
            feature_sets.append(feature_set)
            descriptor_sets.append(descriptors)
            extraction_modes.append(mode)
            detection_backends.append(backend)
            feature_payload.append(
                {
                    **feature_set.to_dict(),
                    "mode": mode,
                    "detector": backend,
                }
            )

        for left_index, (left, right) in enumerate(zip(feature_sets, feature_sets[1:])):
            correspondences, confidence, backend, match_mode = self._match_feature_sets(
                left_features=left,
                left_descriptors=descriptor_sets[left_index],
                right_features=right,
                right_descriptors=descriptor_sets[left_index + 1],
            )
            match = FeatureMatch(
                left_image=left.image.source.normalized_name,
                right_image=right.image.source.normalized_name,
                matched_pairs=len(correspondences) if correspondences else self._synthetic_match_count(left, right, confidence),
                confidence=confidence,
                correspondences=correspondences,
            )
            matches.append(match)
            match_payload.append(
                {
                    **match.to_dict(),
                    "mode": match_mode,
                    "backend": backend,
                }
            )
            matching_backends.append(backend)

        stage_mode = "real" if any(mode == "real" for mode in extraction_modes) else "synthetic"
        if any(payload.get("mode") == "real" for payload in match_payload):
            stage_mode = "real"

        artifacts = {
            "stage": self.name,
            "mode": stage_mode,
            "feature_count": len(feature_sets),
            "match_count": len(matches),
            "detectors": detection_backends,
            "matching_backends": matching_backends,
            "features": feature_payload,
            "matches": match_payload,
        }
        artifact_path = write_json(work_dir / "features_and_matches.json", artifacts)
        report = PipelineStageResult(
            name=self.name,
            status="completed",
            summary=(
                "Se extrajeron caracteristicas reales cuando fue posible y se generaron "
                "emparejamientos entre imagenes consecutivas."
            ),
            mode=stage_mode,
            artifact_path=artifact_path,
            metrics={
                "feature_count": len(feature_sets),
                "match_count": len(matches),
                "mode": stage_mode,
                "real_feature_count": sum(1 for item in extraction_modes if item == "real"),
                "synthetic_feature_count": sum(1 for item in extraction_modes if item != "real"),
            },
        )
        return feature_sets, matches, report

    def _extract_features(
        self,
        image: PreprocessedImage,
    ) -> tuple[list[FeaturePoint], list[tuple[float, ...]], str, str]:
        if cv2 is not None and np is not None:
            try:
                return self._extract_with_orb(image)
            except Exception:
                pass

        try:
            return self._extract_with_pil(image)
        except Exception:
            return self._extract_synthetic(image)

    def _extract_with_orb(
        self,
        image: PreprocessedImage,
    ) -> tuple[list[FeaturePoint], list[tuple[float, ...]], str, str]:
        if cv2 is None or np is None:  # pragma: no cover - depende del entorno
            raise ProcessingError("OpenCV no esta disponible.")

        with Image.open(image.preprocessed_path) as raw:
            normalized = ImageOps.exif_transpose(raw).convert("L")
            if max(normalized.size) > 320:
                normalized.thumbnail((320, 320))
            gray_array = np.array(normalized)

        orb = cv2.ORB_create(nfeatures=self._MAX_FEATURES * 4)
        keypoints, descriptors = orb.detectAndCompute(gray_array, None)
        if not keypoints or descriptors is None:
            raise ProcessingError("ORB no encontro caracteristicas suficientes.")

        selected = list(zip(keypoints, descriptors))[: self._MAX_FEATURES]
        points: list[FeaturePoint] = []
        descriptor_rows: list[tuple[float, ...]] = []
        width = max(normalized.size[0] - 1, 1)
        height = max(normalized.size[1] - 1, 1)

        for keypoint, descriptor in selected:
            x = round(float(keypoint.pt[0]) / width, 6)
            y = round(float(keypoint.pt[1]) / height, 6)
            score = round(min(0.999, abs(float(keypoint.response)) / 50.0), 4)
            points.append(FeaturePoint(x=x, y=y, score=score))
            descriptor_rows.append(tuple(float(value) / 255.0 for value in descriptor.tolist()))

        if len(points) < self._MIN_REAL_FEATURES:
            raise ProcessingError("ORB no produjo suficientes keypoints.")

        return points, descriptor_rows, "real", "orb"

    def _extract_with_pil(
        self,
        image: PreprocessedImage,
    ) -> tuple[list[FeaturePoint], list[tuple[float, ...]], str, str]:
        with Image.open(image.preprocessed_path) as raw:
            normalized = ImageOps.exif_transpose(raw).convert("L")
            if max(normalized.size) > 320:
                normalized.thumbnail((320, 320))

            width, height = normalized.size
            pixels = list(normalized.getdata())
            grid = [pixels[row * width : (row + 1) * width] for row in range(height)]

        candidates = self._detect_keypoints(grid, width, height)
        if len(candidates) < self._MIN_REAL_FEATURES:
            raise ProcessingError("La extraccion real no encontro suficientes caracteristicas.")

        points: list[FeaturePoint] = []
        descriptors: list[tuple[float, ...]] = []
        for score, x, y in candidates[: self._MAX_FEATURES]:
            descriptor = self._build_patch_descriptor(grid, width, height, x, y)
            points.append(
                FeaturePoint(
                    x=round(x / max(width - 1, 1), 6),
                    y=round(y / max(height - 1, 1), 6),
                    score=round(min(0.999, score / 255.0), 4),
                )
            )
            descriptors.append(descriptor)

        return points, descriptors, "real", "pil-gradient"

    def _extract_synthetic(
        self,
        image: PreprocessedImage,
    ) -> tuple[list[FeaturePoint], list[tuple[float, ...]], str, str]:
        keypoints = self._build_synthetic_keypoints(image)
        descriptors = self._build_synthetic_descriptors(image, len(keypoints))
        return keypoints, descriptors, "synthetic", "hash-synthetic"

    def _detect_keypoints(self, grid: list[list[int]], width: int, height: int) -> list[tuple[float, int, int]]:
        if width < 4 or height < 4:
            return []

        cell_size = max(8, min(width, height) // 8 or 8)
        candidates: list[tuple[float, int, int]] = []

        for top in range(1, max(height - 1, 1), cell_size):
            bottom = min(top + cell_size, height - 1)
            for left in range(1, max(width - 1, 1), cell_size):
                right = min(left + cell_size, width - 1)
                best: tuple[float, int, int] | None = None
                for y in range(top, bottom):
                    for x in range(left, right):
                        score = self._gradient_score(grid, x, y)
                        if best is None or score > best[0]:
                            best = (score, x, y)
                if best and best[0] >= 18:
                    candidates.append(best)

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected: list[tuple[float, int, int]] = []
        for score, x, y in candidates:
            if any(abs(x - sx) < 4 and abs(y - sy) < 4 for _, sx, sy in selected):
                continue
            selected.append((score, x, y))
            if len(selected) >= self._MAX_FEATURES:
                break

        return selected

    @staticmethod
    def _gradient_score(grid: list[list[int]], x: int, y: int) -> float:
        left = grid[y][x - 1]
        right = grid[y][x + 1]
        up = grid[y - 1][x]
        down = grid[y + 1][x]
        diag_a = grid[y - 1][x - 1]
        diag_b = grid[y + 1][x + 1]
        diag_c = grid[y - 1][x + 1]
        diag_d = grid[y + 1][x - 1]
        return float(
            abs(right - left)
            + abs(down - up)
            + 0.35 * abs(diag_b - diag_a)
            + 0.35 * abs(diag_d - diag_c)
        )

    def _build_patch_descriptor(
        self,
        grid: list[list[int]],
        width: int,
        height: int,
        x: int,
        y: int,
    ) -> tuple[float, ...]:
        values: list[float] = []
        for dy in range(-self._PATCH_RADIUS, self._PATCH_RADIUS + 1):
            row_index = min(max(y + dy, 0), height - 1)
            for dx in range(-self._PATCH_RADIUS, self._PATCH_RADIUS + 1):
                col_index = min(max(x + dx, 0), width - 1)
                values.append(grid[row_index][col_index] / 255.0)

        mean_value = sum(values) / max(len(values), 1)
        return tuple(round(value - mean_value, 6) for value in values)

    def _build_synthetic_keypoints(self, image: PreprocessedImage) -> list[FeaturePoint]:
        digest = image.source.sha256
        points: list[FeaturePoint] = []
        for index in range(8):
            start = index * 4
            chunk = digest[start : start + 8] or "0"
            raw = int(chunk, 16)
            angle = (raw % 3600) / 3600 * 2 * math.pi
            radius = 0.12 + ((raw >> 8) % 25) / 100
            x = round(0.5 + math.cos(angle) * radius, 6)
            y = round(0.5 + math.sin(angle) * radius, 6)
            score = round(0.45 + ((raw >> 16) % 40) / 100, 4)
            points.append(FeaturePoint(x=x, y=y, score=score))
        return points

    def _build_synthetic_descriptors(
        self,
        image: PreprocessedImage,
        keypoint_count: int,
    ) -> list[tuple[float, ...]]:
        seed = image.source.sha256
        descriptors: list[tuple[float, ...]] = []
        for index in range(keypoint_count):
            cursor = index * 6
            values: list[float] = []
            for offset in range(16):
                chunk = seed[(cursor + offset * 2) % len(seed) : (cursor + offset * 2) % len(seed) + 2]
                raw = int(chunk or "0", 16)
                values.append(round(raw / 255.0, 6))
            mean_value = sum(values) / len(values)
            descriptors.append(tuple(round(value - mean_value, 6) for value in values))
        return descriptors

    def _match_feature_sets(
        self,
        left_features: ImageFeatures,
        left_descriptors: list[tuple[float, ...]],
        right_features: ImageFeatures,
        right_descriptors: list[tuple[float, ...]],
    ) -> tuple[list[tuple[FeaturePoint, FeaturePoint]], float, str, str]:
        correspondences, average_distance, backend = self._match_descriptors(
            left_descriptors,
            right_descriptors,
            left_features.keypoints,
            right_features.keypoints,
        )
        if correspondences:
            confidence = self._real_match_confidence(
                left_features=left_features,
                right_features=right_features,
                correspondences=correspondences,
                average_distance=average_distance,
            )
            return correspondences, confidence, backend, "real"

        confidence = self._synthetic_match_confidence(left_features, right_features)
        return [], confidence, "synthetic-hash", "synthetic"

    def _match_descriptors(
        self,
        left_descriptors: list[tuple[float, ...]],
        right_descriptors: list[tuple[float, ...]],
        left_keypoints: list[FeaturePoint],
        right_keypoints: list[FeaturePoint],
    ) -> tuple[list[tuple[FeaturePoint, FeaturePoint]], float, str]:
        if not left_descriptors or not right_descriptors:
            return [], 1.0, "synthetic-hash"

        if cv2 is not None and np is not None and len(left_descriptors[0]) == 32:
            try:
                return self._match_with_orb_like_backend(left_descriptors, right_descriptors, left_keypoints, right_keypoints)
            except Exception:
                pass

        return self._match_with_descriptor_backbone(
            left_descriptors,
            right_descriptors,
            left_keypoints,
            right_keypoints,
        )

    def _match_with_orb_like_backend(
        self,
        left_descriptors: list[tuple[float, ...]],
        right_descriptors: list[tuple[float, ...]],
        left_keypoints: list[FeaturePoint],
        right_keypoints: list[FeaturePoint],
    ) -> tuple[list[tuple[FeaturePoint, FeaturePoint]], float, str]:
        if cv2 is None or np is None:  # pragma: no cover - depende del entorno
            raise ProcessingError("OpenCV no esta disponible.")

        left = np.array([[int(round(value * 255)) for value in descriptor] for descriptor in left_descriptors], dtype=np.uint8)
        right = np.array([[int(round(value * 255)) for value in descriptor] for descriptor in right_descriptors], dtype=np.uint8)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = matcher.knnMatch(left, right, k=2)
        correspondences: list[tuple[FeaturePoint, FeaturePoint]] = []
        distances: list[float] = []
        used_right: set[int] = set()

        for entry in raw_matches:
            if not entry:
                continue
            best = entry[0]
            second = entry[1] if len(entry) > 1 else None
            ratio = best.distance / max(second.distance, 1e-6) if second else 0.0
            if ratio > 0.86 and best.distance > 24:
                continue
            if best.trainIdx in used_right:
                continue
            used_right.add(best.trainIdx)
            correspondences.append((left_keypoints[best.queryIdx], right_keypoints[best.trainIdx]))
            distances.append(best.distance / 255.0)

        if not correspondences:
            return [], 1.0, "orb-bfmatcher"

        return correspondences, sum(distances) / len(distances), "orb-bfmatcher"

    def _match_with_descriptor_backbone(
        self,
        left_descriptors: list[tuple[float, ...]],
        right_descriptors: list[tuple[float, ...]],
        left_keypoints: list[FeaturePoint],
        right_keypoints: list[FeaturePoint],
    ) -> tuple[list[tuple[FeaturePoint, FeaturePoint]], float, str]:
        if not left_descriptors or not right_descriptors:
            return [], 1.0, "synthetic-hash"

        best_right_for_left: list[tuple[int, float, float]] = []
        for descriptor in left_descriptors:
            distances = [
                (self._descriptor_distance(descriptor, candidate), index)
                for index, candidate in enumerate(right_descriptors)
            ]
            distances.sort(key=lambda item: item[0])
            if not distances:
                continue
            best_distance, best_index = distances[0]
            second_distance = distances[1][0] if len(distances) > 1 else best_distance + 1.0
            best_right_for_left.append((best_index, best_distance, second_distance))

        best_left_for_right: dict[int, int] = {}
        for left_index, descriptor in enumerate(right_descriptors):
            distances = [
                (self._descriptor_distance(descriptor, candidate), index)
                for index, candidate in enumerate(left_descriptors)
            ]
            distances.sort(key=lambda item: item[0])
            if not distances:
                continue
            _, best_index = distances[0]
            best_left_for_right[left_index] = best_index

        correspondences: list[tuple[FeaturePoint, FeaturePoint]] = []
        distances: list[float] = []
        used_right: set[int] = set()

        for left_index, (best_right_index, best_distance, second_distance) in enumerate(best_right_for_left):
            if best_left_for_right.get(best_right_index) != left_index:
                continue
            ratio = best_distance / max(second_distance, 1e-6)
            if ratio > 0.86 and best_distance > 0.22:
                continue
            if best_right_index in used_right:
                continue
            used_right.add(best_right_index)
            correspondences.append((left_keypoints[left_index], right_keypoints[best_right_index]))
            distances.append(best_distance)

        if not correspondences:
            return [], 1.0, "pil-descriptor"

        return correspondences, sum(distances) / len(distances), "pil-descriptor"

    @staticmethod
    def _descriptor_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        length = min(len(left), len(right))
        if length == 0:
            return 1.0
        total = sum(abs(left[index] - right[index]) for index in range(length))
        return total / length

    @staticmethod
    def _real_match_confidence(
        *,
        left_features: ImageFeatures,
        right_features: ImageFeatures,
        correspondences: list[tuple[FeaturePoint, FeaturePoint]],
        average_distance: float,
    ) -> float:
        overlap = len(correspondences) / max(min(len(left_features.keypoints), len(right_features.keypoints)), 1)
        quality = max(0.0, 1.0 - min(1.0, average_distance))
        score = (left_features.score + right_features.score) / 2
        return round(min(0.99, 0.2 + overlap * 0.45 + quality * 0.25 + score * 0.1), 4)

    @staticmethod
    def _synthetic_match_confidence(left: ImageFeatures, right: ImageFeatures) -> float:
        left_seed = left.descriptor_seed
        right_seed = right.descriptor_seed
        shared_prefix = 0
        for left_char, right_char in zip(left_seed, right_seed):
            if left_char != right_char:
                break
            shared_prefix += 1
        raw = (shared_prefix / max(len(left_seed), 1)) * 0.65
        score = (left.score + right.score) / 2
        return round(min(0.98, 0.25 + raw + (score * 0.45)), 4)

    @staticmethod
    def _feature_score(image: PreprocessedImage, keypoints: list[FeaturePoint]) -> float:
        if not keypoints:
            base = (image.brightness + image.contrast + image.sharpness) / 3
            density = min(image.source.pixel_count / 120_000, 1)
            return round(min(0.95, base * 0.85 + density * 0.15), 4)

        mean_keypoint_score = sum(point.score for point in keypoints) / len(keypoints)
        image_quality = (image.brightness + image.contrast + image.sharpness) / 3
        density = min(image.source.pixel_count / 120_000, 1)
        return round(min(0.99, mean_keypoint_score * 0.45 + image_quality * 0.4 + density * 0.15), 4)

    @staticmethod
    def _descriptor_seed(
        image: PreprocessedImage,
        descriptors: list[tuple[float, ...]],
        backend: str,
    ) -> str:
        if not descriptors:
            return image.source.sha256[:16]

        descriptor_bytes = "|".join(
            ",".join(f"{value:.4f}" for value in descriptor[:16])
            for descriptor in descriptors[:8]
        )
        digest = hashlib.sha256(
            f"{image.source.sha256}:{backend}:{descriptor_bytes}".encode("utf-8")
        ).hexdigest()
        return digest[:16]

    @staticmethod
    def _synthetic_match_count(left: ImageFeatures, right: ImageFeatures, confidence: float) -> int:
        return max(
            8,
            min(
                len(left.keypoints),
                len(right.keypoints),
                int(round(8 + confidence * 10)),
            ),
        )
