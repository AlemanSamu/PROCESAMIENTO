from __future__ import annotations

import json
import struct
from pathlib import Path

from app.core.errors import ProcessingError
from app.models.schemas import OutputFormat

from .artifacts import ExportResult, MeshModel, write_json


class ModelExporter:
    """Exporta la malla a OBJ o GLB manteniendo el contrato actual del backend."""

    name = "model_export"

    def export(
        self,
        project_id: str,
        mesh: MeshModel,
        output_dir: Path,
        output_format: OutputFormat,
        work_dir: Path,
    ) -> ExportResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_format == OutputFormat.OBJ:
            model_path = output_dir / f"{project_id}_model.obj"
            self._write_obj(model_path, mesh, project_id)
        elif output_format == OutputFormat.GLB:
            model_path = output_dir / f"{project_id}_model.glb"
            self._write_glb(model_path, mesh, project_id)
        else:
            raise ProcessingError(f"Formato de exportacion no soportado: {output_format.value}.")

        bytes_written = model_path.stat().st_size
        write_json(
            work_dir / "export.json",
            {
                "stage": self.name,
                "project_id": project_id,
                "output_format": output_format.value,
                "model_path": str(model_path),
                "bytes_written": bytes_written,
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
            },
        )
        return ExportResult(
            model_path=model_path,
            output_format=output_format,
            bytes_written=bytes_written,
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces),
        )

    def _write_obj(self, model_path: Path, mesh: MeshModel, project_id: str) -> None:
        lines = [
            "# OBJ generado por ReconstructionPipeline",
            f"# project_id={project_id}",
            f"# source_point_count={mesh.source_point_count}",
            "o ReconstructionObject",
        ]
        for vertex in mesh.vertices:
            lines.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}")
        for face in mesh.faces:
            lines.append(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}")

        model_path.write_text("\n".join(lines), encoding="utf-8")

    def _write_glb(self, model_path: Path, mesh: MeshModel, project_id: str) -> None:
        if len(mesh.vertices) < 3:
            raise ProcessingError("La malla no contiene suficientes vertices para GLB.")

        positions = struct.pack(
            "<" + "f" * (len(mesh.vertices) * 3),
            *[component for vertex in mesh.vertices for component in vertex],
        )

        flat_indices: list[int] = []
        for face in mesh.faces:
            flat_indices.extend(face)

        if any(index > 65535 for index in flat_indices):
            raise ProcessingError("La malla excede el rango compatible con este exportador GLB.")

        indices = struct.pack("<" + "H" * len(flat_indices), *flat_indices)
        bin_payload = positions + indices
        if len(bin_payload) % 4:
            bin_payload += b"\x00" * (4 - (len(bin_payload) % 4))

        json_payload = {
            "asset": {
                "version": "2.0",
                "generator": "ReconstructionPipeline",
            },
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [
                {
                    "name": "ReconstructionMesh",
                    "primitives": [
                        {
                            "attributes": {"POSITION": 0},
                            "indices": 1,
                            "mode": 4,
                        }
                    ],
                }
            ],
            "buffers": [{"byteLength": len(bin_payload)}],
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteOffset": 0,
                    "byteLength": len(positions),
                    "target": 34962,
                },
                {
                    "buffer": 0,
                    "byteOffset": len(positions),
                    "byteLength": len(indices),
                    "target": 34963,
                },
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": len(mesh.vertices),
                    "type": "VEC3",
                    "min": self._min_bounds(mesh.vertices),
                    "max": self._max_bounds(mesh.vertices),
                },
                {
                    "bufferView": 1,
                    "componentType": 5123,
                    "count": len(flat_indices),
                    "type": "SCALAR",
                },
            ],
            "extras": {
                "projectId": project_id,
                "sourcePointCount": mesh.source_point_count,
                "vertexCount": len(mesh.vertices),
                "faceCount": len(mesh.faces),
                "note": "Generated by ReconstructionPipeline",
            },
        }

        json_bytes = json.dumps(
            json_payload,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        if len(json_bytes) % 4:
            json_bytes += b" " * (4 - (len(json_bytes) % 4))

        total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_payload)
        header = struct.pack("<4sII", b"glTF", 2, total_length)
        json_chunk = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
        bin_chunk = struct.pack("<I4s", len(bin_payload), b"BIN\x00") + bin_payload
        model_path.write_bytes(header + json_chunk + bin_chunk)

    @staticmethod
    def _min_bounds(vertices: list[tuple[float, float, float]]) -> list[float]:
        xs = [vertex[0] for vertex in vertices]
        ys = [vertex[1] for vertex in vertices]
        zs = [vertex[2] for vertex in vertices]
        return [min(xs), min(ys), min(zs)]

    @staticmethod
    def _max_bounds(vertices: list[tuple[float, float, float]]) -> list[float]:
        xs = [vertex[0] for vertex in vertices]
        ys = [vertex[1] for vertex in vertices]
        zs = [vertex[2] for vertex in vertices]
        return [max(xs), max(ys), max(zs)]
