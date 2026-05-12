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

    def export_textured_box(
        self,
        *,
        project_id: str,
        dimensions: tuple[float, float, float],
        texture_atlas_path: Path,
        output_dir: Path,
        output_format: OutputFormat,
        work_dir: Path,
    ) -> ExportResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_format != OutputFormat.GLB:
            raise ProcessingError(
                "El exportador texturizado tipo box solo soporta GLB en este modo controlado."
            )
        if not texture_atlas_path.exists() or not texture_atlas_path.is_file():
            raise ProcessingError(
                f"No se encontro el atlas de textura requerido para exportar GLB texturizado: {texture_atlas_path}"
            )

        vertices, faces, uv = self._build_textured_box_geometry(dimensions)
        texture_bytes = texture_atlas_path.read_bytes()
        if not texture_bytes:
            raise ProcessingError(
                f"El atlas de textura esta vacio y no puede usarse en GLB: {texture_atlas_path}"
            )

        model_path = output_dir / f"{project_id}_model.glb"
        self._write_textured_glb(
            model_path=model_path,
            project_id=project_id,
            vertices=vertices,
            faces=faces,
            uv=uv,
            texture_png_bytes=texture_bytes,
        )

        bytes_written = model_path.stat().st_size
        write_json(
            work_dir / "export.json",
            {
                "stage": self.name,
                "project_id": project_id,
                "output_format": output_format.value,
                "model_path": str(model_path),
                "bytes_written": bytes_written,
                "vertex_count": int(len(vertices)),
                "face_count": int(len(faces)),
                "textured": True,
                "texture_atlas_path": str(texture_atlas_path),
                "note": "GLB texturizado generado desde atlas de imagenes de entrada.",
            },
        )
        return ExportResult(
            model_path=model_path,
            output_format=output_format,
            bytes_written=bytes_written,
            vertex_count=int(len(vertices)),
            face_count=int(len(faces)),
        )

    def _write_textured_glb(
        self,
        *,
        model_path: Path,
        project_id: str,
        vertices: list[tuple[float, float, float]],
        faces: list[tuple[int, int, int]],
        uv: list[tuple[float, float]],
        texture_png_bytes: bytes,
    ) -> None:
        if len(vertices) < 3:
            raise ProcessingError("La malla texturizada no contiene suficientes vertices para GLB.")
        if len(vertices) != len(uv):
            raise ProcessingError("Los UV no coinciden con la cantidad de vertices del mesh texturizado.")

        flat_indices: list[int] = []
        for face in faces:
            flat_indices.extend(face)
        if any(index > 65535 for index in flat_indices):
            raise ProcessingError("La malla texturizada excede el rango de indices uint16.")

        positions = struct.pack(
            "<" + "f" * (len(vertices) * 3),
            *[component for vertex in vertices for component in vertex],
        )
        texcoords = struct.pack(
            "<" + "f" * (len(uv) * 2),
            *[component for uv_pair in uv for component in uv_pair],
        )
        indices = struct.pack("<" + "H" * len(flat_indices), *flat_indices)

        chunks: list[bytes] = []
        buffer_views: list[dict[str, int]] = []
        offset = 0

        def _append_chunk(payload: bytes, target: int | None = None) -> None:
            nonlocal offset
            chunks.append(payload)
            view: dict[str, int] = {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(payload),
            }
            if target is not None:
                view["target"] = target
            buffer_views.append(view)
            offset += len(payload)
            if offset % 4:
                pad = 4 - (offset % 4)
                chunks.append(b"\x00" * pad)
                offset += pad

        _append_chunk(positions, target=34962)
        _append_chunk(texcoords, target=34962)
        _append_chunk(indices, target=34963)
        _append_chunk(texture_png_bytes, target=None)

        bin_payload = b"".join(chunks)
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
                    "name": "ReconstructionTexturedBox",
                    "primitives": [
                        {
                            "attributes": {
                                "POSITION": 0,
                                "TEXCOORD_0": 1,
                            },
                            "indices": 2,
                            "material": 0,
                            "mode": 4,
                        }
                    ],
                }
            ],
            "materials": [
                {
                    "name": "CapturedBoxMaterial",
                    "pbrMetallicRoughness": {
                        "baseColorTexture": {"index": 0},
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.95,
                    },
                }
            ],
            "textures": [{"sampler": 0, "source": 0}],
            "samplers": [
                {
                    "magFilter": 9729,
                    "minFilter": 9987,
                    "wrapS": 10497,
                    "wrapT": 10497,
                }
            ],
            "images": [{"bufferView": 3, "mimeType": "image/png"}],
            "buffers": [{"byteLength": len(bin_payload)}],
            "bufferViews": buffer_views,
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": len(vertices),
                    "type": "VEC3",
                    "min": self._min_bounds(vertices),
                    "max": self._max_bounds(vertices),
                },
                {
                    "bufferView": 1,
                    "componentType": 5126,
                    "count": len(uv),
                    "type": "VEC2",
                },
                {
                    "bufferView": 2,
                    "componentType": 5123,
                    "count": len(flat_indices),
                    "type": "SCALAR",
                },
            ],
            "extras": {
                "projectId": project_id,
                "vertexCount": len(vertices),
                "faceCount": len(faces),
                "textured": True,
                "note": "Generated by ReconstructionPipeline with captured atlas texture",
            },
        }

        json_bytes = json.dumps(
            json_payload,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        if len(json_bytes) % 4:
            json_bytes += b" " * (4 - (len(json_bytes) % 4))
        if len(bin_payload) % 4:
            bin_payload += b"\x00" * (4 - (len(bin_payload) % 4))

        total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_payload)
        header = struct.pack("<4sII", b"glTF", 2, total_length)
        json_chunk = struct.pack("<I4s", len(json_bytes), b"JSON") + json_bytes
        bin_chunk = struct.pack("<I4s", len(bin_payload), b"BIN\x00") + bin_payload
        model_path.write_bytes(header + json_chunk + bin_chunk)

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

    @classmethod
    def _build_textured_box_geometry(
        cls,
        dimensions: tuple[float, float, float],
    ) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], list[tuple[float, float]]]:
        width, height, depth = dimensions
        hx = width / 2.0
        hy = height / 2.0
        hz = depth / 2.0

        # Layout del atlas (3x2):
        # fila 0: front | back | top
        # fila 1: left  | right| bottom
        face_specs = [
            (
                [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)],
                cls._tile_uv_bounds(col=0, row=0, cols=3, rows=2),
            ),
            (
                [(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz)],
                cls._tile_uv_bounds(col=1, row=0, cols=3, rows=2),
            ),
            (
                [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)],
                cls._tile_uv_bounds(col=0, row=1, cols=3, rows=2),
            ),
            (
                [(hx, -hy, hz), (hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz)],
                cls._tile_uv_bounds(col=1, row=1, cols=3, rows=2),
            ),
            (
                [(-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz), (-hx, hy, -hz)],
                cls._tile_uv_bounds(col=2, row=0, cols=3, rows=2),
            ),
            (
                [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)],
                cls._tile_uv_bounds(col=2, row=1, cols=3, rows=2),
            ),
        ]

        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        uv: list[tuple[float, float]] = []

        for vertex_block, (u0, v0, u1, v1) in face_specs:
            start = len(vertices)
            vertices.extend(vertex_block)
            uv.extend(
                [
                    (u0, v0),
                    (u1, v0),
                    (u1, v1),
                    (u0, v1),
                ]
            )
            faces.append((start, start + 1, start + 2))
            faces.append((start, start + 2, start + 3))

        return vertices, faces, uv

    @staticmethod
    def _tile_uv_bounds(*, col: int, row: int, cols: int, rows: int) -> tuple[float, float, float, float]:
        u0 = col / float(cols)
        u1 = (col + 1) / float(cols)
        # UV usa origen abajo-izquierda; el atlas se compone con origen arriba-izquierda.
        top = row / float(rows)
        bottom = (row + 1) / float(rows)
        v1 = 1.0 - top
        v0 = 1.0 - bottom
        return u0, v0, u1, v1


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
