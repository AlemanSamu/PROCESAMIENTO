from __future__ import annotations

from math import atan2, cos, pi, sin
from pathlib import Path

from app.core.errors import ProcessingError

from .artifacts import CameraPose, FeatureMatch, FeaturePoint, ImageFeatures, PipelineStageResult, write_json


class PoseEstimator:
    """Construye poses sinteticas por defecto y estimaciones aproximadas cuando hay correspondencias reales."""

    name = "pose_estimation"

    def run(
        self,
        features: list[ImageFeatures],
        matches: list[FeatureMatch],
        work_dir: Path,
    ) -> tuple[list[CameraPose], PipelineStageResult]:
        if not features:
            raise ProcessingError("No hay caracteristicas suficientes para estimar poses.")

        confidence_by_image = self._build_confidence_map(matches)
        match_lookup = self._build_match_lookup(matches)
        poses: list[CameraPose] = []
        payload: list[dict[str, object]] = []
        total = len(features)
        real_pose_count = 0
        synthetic_pose_count = 0

        for index, feature_set in enumerate(features):
            image_name = feature_set.image.source.normalized_name
            real_match = self._pick_real_match(image_name, match_lookup)
            if real_match is not None:
                pose = self._estimate_real_pose(feature_set, real_match)
                pose_mode = "real"
                real_pose_count += 1
            else:
                pose = self._estimate_synthetic_pose(index, total, feature_set, confidence_by_image)
                pose_mode = "synthetic"
                synthetic_pose_count += 1

            poses.append(pose)
            payload.append({**pose.to_dict(), "mode": pose_mode})

        mode = "real" if real_pose_count > 0 else "synthetic"
        artifact_path = write_json(
            work_dir / "poses.json",
            {
                "stage": self.name,
                "mode": mode,
                "pose_count": len(poses),
                "real_pose_count": real_pose_count,
                "synthetic_pose_count": synthetic_pose_count,
                "poses": payload,
            },
        )
        report = PipelineStageResult(
            name=self.name,
            status="completed",
            summary="Se estimaron poses reales cuando hubo correspondencias suficientes y se mantuvo un fallback sintetico.",
            mode=mode,
            artifact_path=artifact_path,
            metrics={
                "pose_count": len(poses),
                "real_pose_count": real_pose_count,
                "synthetic_pose_count": synthetic_pose_count,
                "mode": mode,
            },
        )
        return poses, report

    @staticmethod
    def _build_confidence_map(matches: list[FeatureMatch]) -> dict[str, float]:
        confidence: dict[str, float] = {}
        for match in matches:
            confidence[match.left_image] = max(confidence.get(match.left_image, 0.0), match.confidence)
            confidence[match.right_image] = max(confidence.get(match.right_image, 0.0), match.confidence)
        return confidence

    @staticmethod
    def _build_match_lookup(matches: list[FeatureMatch]) -> dict[str, list[FeatureMatch]]:
        lookup: dict[str, list[FeatureMatch]] = {}
        for match in matches:
            lookup.setdefault(match.left_image, []).append(match)
            lookup.setdefault(match.right_image, []).append(match)
        return lookup

    @staticmethod
    def _pick_real_match(image_name: str, lookup: dict[str, list[FeatureMatch]]) -> FeatureMatch | None:
        candidates = [
            match
            for match in lookup.get(image_name, [])
            if match.correspondences and len(match.correspondences) >= 4
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.confidence)

    def _estimate_real_pose(self, feature_set: ImageFeatures, match: FeatureMatch) -> CameraPose:
        source_points, target_points = self._pair_points(feature_set.image.source.normalized_name, match)
        if len(source_points) < 4 or len(target_points) < 4:
            return self._estimate_synthetic_pose(
                feature_set.image.source.index - 1,
                feature_set.image.source.index,
                feature_set,
                {feature_set.image.source.normalized_name: match.confidence},
            )

        source_centroid = self._centroid(source_points)
        target_centroid = self._centroid(target_points)
        source_spread = self._average_radius(source_points, source_centroid)
        target_spread = self._average_radius(target_points, target_centroid)
        scale = target_spread / max(source_spread, 1e-6)
        yaw = self._mean_angle_difference(source_points, target_points, source_centroid, target_centroid)
        translation = self._real_translation(
            source_centroid=source_centroid,
            target_centroid=target_centroid,
            scale=scale,
            score=feature_set.score,
            confidence=match.confidence,
        )
        rotation = self._yaw_to_quaternion(yaw)
        matrix = self._compose_matrix(translation, rotation)
        return CameraPose(
            image_name=feature_set.image.source.normalized_name,
            translation=translation,
            rotation=rotation,
            matrix=matrix,
            confidence=round(match.confidence, 4),
        )

    def _estimate_synthetic_pose(
        self,
        index: int,
        total: int,
        feature_set: ImageFeatures,
        confidence_by_image: dict[str, float],
    ) -> CameraPose:
        angle = (index / max(total, 1)) * 2 * pi
        confidence = confidence_by_image.get(feature_set.image.source.normalized_name, feature_set.score)
        radius = 1.35 + (index * 0.14) + confidence * 0.35
        height = 0.9 + feature_set.score * 0.8
        translation = (
            round(cos(angle) * radius, 6),
            round(sin(angle) * radius, 6),
            round(height, 6),
        )
        rotation = self._yaw_to_quaternion(angle + confidence * 0.2)
        matrix = self._compose_matrix(translation, rotation)
        return CameraPose(
            image_name=feature_set.image.source.normalized_name,
            translation=translation,
            rotation=rotation,
            matrix=matrix,
            confidence=round(confidence, 4),
        )

    @staticmethod
    def _pair_points(image_name: str, match: FeatureMatch) -> tuple[list[FeaturePoint], list[FeaturePoint]]:
        if image_name == match.left_image:
            source = [left for left, _ in match.correspondences]
            target = [right for _, right in match.correspondences]
        elif image_name == match.right_image:
            source = [right for _, right in match.correspondences]
            target = [left for left, _ in match.correspondences]
        else:
            source = []
            target = []
        return source, target

    @staticmethod
    def _centroid(points: list[FeaturePoint]) -> tuple[float, float]:
        total = max(len(points), 1)
        x = sum(point.x for point in points) / total
        y = sum(point.y for point in points) / total
        return (x, y)

    @staticmethod
    def _average_radius(points: list[FeaturePoint], centroid: tuple[float, float]) -> float:
        if not points:
            return 1.0
        cx, cy = centroid
        distances = [((point.x - cx) ** 2 + (point.y - cy) ** 2) ** 0.5 for point in points]
        return sum(distances) / max(len(distances), 1)

    @staticmethod
    def _mean_angle_difference(
        source_points: list[FeaturePoint],
        target_points: list[FeaturePoint],
        source_centroid: tuple[float, float],
        target_centroid: tuple[float, float],
    ) -> float:
        sin_sum = 0.0
        cos_sum = 0.0
        for source_point, target_point in zip(source_points, target_points):
            source_angle = atan2(source_point.y - source_centroid[1], source_point.x - source_centroid[0])
            target_angle = atan2(target_point.y - target_centroid[1], target_point.x - target_centroid[0])
            delta = target_angle - source_angle
            sin_sum += sin(delta)
            cos_sum += cos(delta)
        if sin_sum == 0 and cos_sum == 0:
            return 0.0
        return atan2(sin_sum, cos_sum)

    @staticmethod
    def _real_translation(
        *,
        source_centroid: tuple[float, float],
        target_centroid: tuple[float, float],
        scale: float,
        score: float,
        confidence: float,
    ) -> tuple[float, float, float]:
        dx = (target_centroid[0] - source_centroid[0]) * scale * 2.5
        dy = (target_centroid[1] - source_centroid[1]) * scale * 2.5
        dz = 0.75 + score * 0.65 + confidence * 0.4
        return (round(dx, 6), round(dy, 6), round(dz, 6))

    @staticmethod
    def _yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
        half = yaw / 2
        return (0.0, 0.0, round(sin(half), 6), round(cos(half), 6))

    @staticmethod
    def _compose_matrix(
        translation: tuple[float, float, float],
        rotation: tuple[float, float, float, float],
    ) -> list[list[float]]:
        _, _, z, w = rotation
        sin_yaw = 2 * z * w
        cos_yaw = 1 - 2 * (z * z)
        x, y, z_pos = translation
        return [
            [round(cos_yaw, 6), round(-sin_yaw, 6), 0.0, x],
            [round(sin_yaw, 6), round(cos_yaw, 6), 0.0, y],
            [0.0, 0.0, 1.0, z_pos],
            [0.0, 0.0, 0.0, 1.0],
        ]
