from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _count_points_from_txt(points_txt: Path) -> int:
    if not points_txt.exists() or not points_txt.is_file():
        return 0
    try:
        count = 0
        for line in points_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            count += 1
        return count
    except Exception:
        return 0


def _recommendation(
    *,
    classification: str,
    points: int,
    cameras: int,
    dense_available: bool,
    geometry_source: str,
) -> str:
    if geometry_source == "primitive_box":
        return "resultado defendible solo como fallback"
    if classification == "success_real":
        return "dataset aceptable"
    if classification == "success_sparse_only":
        if not dense_available:
            return "activar dense"
        return "usar perfil quality"
    if points < 500 or cameras < 6:
        return "repetir fotos"
    return "usar perfil quality"


def _sparse_density_level(points: int) -> str:
    if points < 1500:
        return "low"
    if points <= 5000:
        return "medium"
    return "high"


def _defendibility_label(classification: str, is_real_sfm: bool, dense_mesh_usable: bool) -> str:
    if classification == "success_real" and is_real_sfm and dense_mesh_usable:
        return "reconstruccion real"
    if classification == "success_approx_surface":
        return "superficie aproximada desde sparse"
    if classification == "success_sparse_only":
        return "sparse parcial"
    return "fallback academico"


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspeccion rapida de salida de reconstruccion por project_id.")
    parser.add_argument("--project-id", required=True, type=str, help="ID del proyecto (ej: 7251d6262a11)")
    parser.add_argument(
        "--projects-root",
        default=Path("data/projects"),
        type=Path,
        help="Ruta raiz de proyectos (default: data/projects).",
    )
    args = parser.parse_args()

    project_id = args.project_id.strip()
    project_dir = args.projects_root / project_id
    output_dir = project_dir / "output"
    pipeline_dir = output_dir / "pipeline"
    sparse0 = output_dir / "workspace" / "sparse" / "0"
    points_txt = output_dir / "colmap_sparse_txt" / "points3D.txt"

    meta = _load_json(project_dir / "meta.json") or {}
    quality = _load_json(pipeline_dir / "quality_report.json") or {}
    colmap = _load_json(pipeline_dir / "colmap_report.json") or {}
    fallback = _load_json(pipeline_dir / "fallback_report.json") or {}

    model_glb = output_dir / f"{project_id}_model.glb"
    model_obj = output_dir / f"{project_id}_model.obj"
    model_path = model_glb if model_glb.exists() else model_obj if model_obj.exists() else None

    classification = str(quality.get("quality_classification") or meta.get("processing_metadata", {}).get("quality_classification") or "unknown")
    geometry_source = str(quality.get("geometry_source") or "unknown")
    visualization_type = str(quality.get("visualization_type") or "point_cloud")
    surface_reconstruction = quality.get("surface_reconstruction") if isinstance(quality.get("surface_reconstruction"), dict) else {}
    visualization_metrics = quality.get("visualization_metrics") if isinstance(quality.get("visualization_metrics"), dict) else {}
    texture_report = quality.get("texture_report") if isinstance(quality.get("texture_report"), dict) else {}
    cameras = int(quality.get("cameras_reconstructed") or colmap.get("cameras_reconstructed") or 0)
    images_registered = int(quality.get("images_registered") or colmap.get("images_registered") or 0)
    points = int(quality.get("points3D_count") or colmap.get("points3D_count") or _count_points_from_txt(points_txt) or 0)
    dense_available = bool(quality.get("dense_available"))
    if not dense_available:
        dense_available = bool((output_dir / "workspace" / "dense" / "fused.ply").exists())
    fallback_used = bool((meta.get("processing_metadata", {}).get("fallback_used")) or colmap.get("fallback_used") or bool(fallback))
    is_primitive_box = geometry_source == "primitive_box"
    is_real_sfm = bool(
        quality.get("is_real_sfm")
        if "is_real_sfm" in quality
        else geometry_source in {"colmap_dense", "colmap_sparse_point_cloud", "colmap_sparse"}
    )
    mesh_faces = int((quality.get("metrics") or {}).get("mesh_face_count") or meta.get("processing_metadata", {}).get("mesh_face_count") or 0)
    dense_mesh_usable = bool(dense_available and mesh_faces > 0 and geometry_source == "colmap_dense")
    sparse_level = str(quality.get("sparse_density_level") or _sparse_density_level(points))
    defendible_as = _defendibility_label(classification, is_real_sfm, dense_mesh_usable)
    recommendation = _recommendation(
        classification=classification,
        points=points,
        cameras=cameras,
        dense_available=dense_available,
        geometry_source=geometry_source,
    )

    print(f"project_id: {project_id}")
    print(f"project_dir: {project_dir}")
    print(f"status: {meta.get('status', 'unknown')}")
    print(f"classification: {classification}")
    print(f"colmap_real_or_fallback: {'fallback' if fallback_used else 'colmap_real'}")
    print(f"sfm_real_executed: {'si' if is_real_sfm else 'no'}")
    print(f"dense_mesh_usable: {'si' if dense_mesh_usable else 'no'}")
    print(f"visualization_generated: {visualization_type}")
    print(f"surface_method_used: {surface_reconstruction.get('method_used', 'n/a')}")
    print(f"surface_vertices_count: {surface_reconstruction.get('vertices_count', 'n/a')}")
    print(f"surface_faces_count: {surface_reconstruction.get('faces_count', 'n/a')}")
    visual_faces_count = int(visualization_metrics.get("visual_faces_count") or 0)
    visual_faces_are_reconstruction = bool(visualization_metrics.get("visual_faces_are_reconstruction"))
    if visual_faces_count > 0 and not visual_faces_are_reconstruction:
        print("warning_visual_faces: Las caras reportadas pertenecen a la visualizacion, no a una malla reconstruida.")
    if mesh_faces < 500:
        print("warning_visual: Malla no usable: faces_count menor al minimo requerido (500).")
    if visualization_type in {"point_cloud", "point_spheres"}:
        print("warning_visual_mode: Resultado visual principal: nube sparse.")
        print("warning_solidity: No es reconstruccion solida.")
    print(f"sparse_density_level: {sparse_level}")
    print(f"defendible_as: {defendible_as}")
    print(f"texture_method: {texture_report.get('texture_method', quality.get('texture_source', 'none'))}")
    print(f"texture_source: {quality.get('texture_source', 'none')}")
    print(f"texture_selected_images: {texture_report.get('selected_images', [])}")
    print(f"texture_confidence: {texture_report.get('texture_confidence', 'n/a')}")
    print(f"textured_faces_count: {texture_report.get('textured_faces_count', 'n/a')}")
    print(f"untextured_faces_count: {texture_report.get('untextured_faces_count', 'n/a')}")
    print(f"final_model_texture_quality: {quality.get('final_model_texture_quality', 'n/a')}")
    print(f"final_model_visual_score: {quality.get('final_model_visual_score', 'n/a')}")
    print(f"final_model_is_presentable: {quality.get('final_model_is_presentable', 'n/a')}")
    print(f"texture_limitations: {texture_report.get('texture_limitations', [])}")
    print(f"geometry_source: {geometry_source}")
    print(f"cameras_reconstructed: {cameras}")
    print(f"images_registered: {images_registered}")
    print(f"points3D_count: {points}")
    print(f"dense_available: {dense_available}")
    print(f"sparse0_exists: {sparse0.exists()}")
    print(f"points3D_txt_exists: {points_txt.exists()}")
    print(f"final_model_path: {model_path if model_path else 'missing'}")
    print(f"is_primitive_box: {is_primitive_box}")
    print(f"logs_dir: {output_dir / 'logs' / 'colmap'}")
    print(f"recommended_next_action: {recommendation}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
