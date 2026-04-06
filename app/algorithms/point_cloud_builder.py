from __future__ import annotations

from math import atan2, cos, pi, sin
from pathlib import Path

from app.core.errors import ProcessingError

from .artifacts import CameraPose, FeatureMatch, PipelineStageResult, Point3D, PointCloud, write_json


class PointCloudBuilder:
    """Construye puntos 3D aproximados desde poses reales o, si hace falta, una nube sintetica estable."""

    name = "point_cloud_building"

    def run(
        self,
        poses: list[CameraPose],
        work_dir: Path,
        matches: list[FeatureMatch] | None = None,
    ) -> tuple[PointCloud, PipelineStageResult]:
        if not poses:
            raise ProcessingError("No hay poses para construir la nube de puntos.")

        match_lookup = self._build_match_lookup(matches or [])
        points: list[Point3D] = []
        real_point_count = 0
        synthetic_point_count = 0

        for pose_index, pose in enumerate(poses):
            real_match = self._pick_real_match(pose.image_name, match_lookup)
            if real_match is not None:
                generated_points = self._build_real_points(pose, real_match)
                if generated_points:
                    points.extend(generated_points)
                    real_point_count += len(generated_points)
                    continue

            generated_points = self._build_synthetic_points(pose, pose_index)
            points.extend(generated_points)
            synthetic_point_count += len(generated_points)

        cloud = PointCloud(points=points, bounds=self._bounds(points))
        mode = "real" if real_point_count > 0 else "synthetic"
        artifact_path = write_json(
            work_dir / "point_cloud.json",
            {
                "stage": self.name,
                "mode": mode,
                "point_count": len(points),
                "real_point_count": real_point_count,
                "synthetic_point_count": synthetic_point_count,
                "bounds": cloud.bounds,
                "points": [point.to_dict() for point in points],
            },
        )
        report = PipelineStageResult(
            name=self.name,
            status="completed",
            summary=(
                "Se construyo una nube de puntos aproximada a partir de las poses y de las "
                "correspondencias reales cuando estuvieron disponibles."
            ),
            mode=mode,
            artifact_path=artifact_path,
            metrics={
                "point_count": len(points),
                "real_point_count": real_point_count,
                "synthetic_point_count": synthetic_point_count,
                "mode": mode,
            },
        )
        return cloud, report

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

    def _build_real_points(self, pose: CameraPose, match: FeatureMatch) -> list[Point3D]:
        source_points, target_points = self._pair_points(pose.image_name, match)
        if len(source_points) < 4 or len(target_points) < 4:
            return []

        yaw = self._pose_yaw(pose.rotation)
        points: list[Point3D] = []
        for index, (source_point, target_point) in enumerate(zip(source_points, target_points)):
            if index >= 48:
                break

            midpoint_x = (source_point.x + target_point.x) / 2
            midpoint_y = (source_point.y + target_point.y) / 2
            disparity = abs(source_point.x - target_point.x) + abs(source_point.y - target_point.y)
            depth = 0.65 + pose.confidence * 0.8 + (1.0 - min(disparity, 1.0)) * 0.55
            local_x = (midpoint_x - 0.5) * depth * 2.0
            local_y = (midpoint_y - 0.5) * depth * 2.0
            world_x, world_y = self._rotate(local_x, local_y, yaw)
            intensity = round(0.4 + pose.confidence * 0.45 + (source_point.score + target_point.score) * 0.1, 4)
            points.append(
                Point3D(
                    x=round(pose.translation[0] + world_x, 6),
                    y=round(pose.translation[1] + world_y, 6),
                    z=round(pose.translation[2] + depth * 0.35, 6),
                    intensity=intensity,
                    source_image=pose.image_name,
                )
            )

        return points

    def _build_synthetic_points(self, pose: CameraPose, pose_index: int) -> list[Point3D]:
        points: list[Point3D] = []
        ring_size = 12 + int(round(pose.confidence * 8))
        radius = 0.08 + pose.confidence * 0.05 + pose_index * 0.005
        for sample_index in range(ring_size):
            angle = (sample_index / ring_size) * 2 * pi
            x = round(pose.translation[0] + cos(angle) * radius, 6)
            y = round(pose.translation[1] + sin(angle) * radius, 6)
            z = round(
                pose.translation[2] + sin(angle * 2) * radius * 0.35,
                6,
            )
            intensity = round(0.35 + pose.confidence * 0.55, 4)
            points.append(
                Point3D(
                    x=x,
                    y=y,
                    z=z,
                    intensity=intensity,
                    source_image=pose.image_name,
                )
            )
        return points

    @staticmethod
    def _pair_points(image_name: str, match: FeatureMatch) -> tuple[list, list]:
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
    def _pose_yaw(rotation: tuple[float, float, float, float]) -> float:
        _, _, z, w = rotation
        return atan2(2 * z * w, 1 - 2 * (z * z))

    @staticmethod
    def _rotate(x: float, y: float, yaw: float) -> tuple[float, float]:
        return (
            round(cos(yaw) * x - sin(yaw) * y, 6),
            round(sin(yaw) * x + cos(yaw) * y, 6),
        )

    @staticmethod
    def _bounds(points: list[Point3D]) -> dict[str, float]:
        if not points:
            return {
                "min_x": 0.0,
                "max_x": 0.0,
                "min_y": 0.0,
                "max_y": 0.0,
                "min_z": 0.0,
                "max_z": 0.0,
            }

        xs = [point.x for point in points]
        ys = [point.y for point in points]
        zs = [point.z for point in points]
        return {
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
            "min_z": min(zs),
            "max_z": max(zs),
        }
