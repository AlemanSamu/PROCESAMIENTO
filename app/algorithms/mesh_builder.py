from __future__ import annotations

from math import atan2
from pathlib import Path

from app.core.errors import ProcessingError

from .artifacts import MeshModel, PipelineStageResult, PointCloud, write_json


class MeshBuilder:
    """Convierte la nube de puntos en una malla simple y estable."""

    name = "mesh_generation"

    def run(self, point_cloud: PointCloud, work_dir: Path) -> tuple[MeshModel, PipelineStageResult]:
        if not point_cloud.points:
            raise ProcessingError("No hay puntos para generar la malla.")

        centroid = self._centroid(point_cloud.points)
        ordered_vertices = self._ordered_vertices(point_cloud.points, centroid)
        faces = self._build_faces(ordered_vertices)
        if not faces:
            raise ProcessingError("No fue posible generar caras de la malla.")

        mesh = MeshModel(
            vertices=ordered_vertices,
            faces=faces,
            centroid=centroid,
            source_point_count=len(point_cloud.points),
        )
        artifact_path = write_json(
            work_dir / "mesh.json",
            {
                "stage": self.name,
                "mesh": mesh.to_dict(),
            },
        )
        report = PipelineStageResult(
            name=self.name,
            status="completed",
            summary="Se genero una malla basica a partir de la nube de puntos.",
            artifact_path=artifact_path,
            metrics={
                "vertex_count": len(ordered_vertices),
                "face_count": len(faces),
            },
        )
        return mesh, report

    @staticmethod
    def _centroid(points) -> tuple[float, float, float]:
        total = len(points)
        x = sum(point.x for point in points) / total
        y = sum(point.y for point in points) / total
        z = sum(point.z for point in points) / total
        return (round(x, 6), round(y, 6), round(z, 6))

    @staticmethod
    def _ordered_vertices(points, centroid: tuple[float, float, float]) -> list[tuple[float, float, float]]:
        ordered = sorted(
            [(point.x, point.y, point.z) for point in points],
            key=lambda vertex: (
                atan2(vertex[1] - centroid[1], vertex[0] - centroid[0]),
                vertex[2],
            ),
        )

        if len(ordered) < 3:
            x, y, z = centroid
            ordered = [
                (round(x - 0.1, 6), round(y - 0.05, 6), round(z, 6)),
                (round(x + 0.1, 6), round(y - 0.05, 6), round(z, 6)),
                (round(x, 6), round(y + 0.1, 6), round(z + 0.08, 6)),
            ]

        return ordered

    @staticmethod
    def _build_faces(vertices: list[tuple[float, float, float]]) -> list[tuple[int, int, int]]:
        if len(vertices) < 3:
            return []

        faces: list[tuple[int, int, int]] = []
        for index in range(1, len(vertices) - 1):
            faces.append((0, index, index + 1))

        if not faces:
            faces.append((0, 1, 2))

        return faces
