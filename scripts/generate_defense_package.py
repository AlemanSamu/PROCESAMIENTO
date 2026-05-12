from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera un paquete de defensa academica con reportes y artefactos tecnicos."
    )
    parser.add_argument(
        "--project-id",
        default=None,
        help="ID del proyecto. Si se omite, se usa el proyecto mas reciente en data/projects.",
    )
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=Path("data/projects"),
        help="Raiz de proyectos del backend.",
    )
    parser.add_argument(
        "--experiments-reports",
        type=Path,
        default=Path("data/experiments/reports"),
        help="Carpeta con profile_comparison.csv/json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("defense_package"),
        help="Carpeta base donde se generara el paquete.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Carpeta opcional del dataset real para anexar reporte de validacion si existe.",
    )
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON invalido (no objeto): {path}")
    return payload


def _safe_iso_to_dt(raw: Any) -> datetime:
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _find_latest_project_dir(projects_root: Path) -> Path:
    if not projects_root.exists():
        raise RuntimeError(f"No existe la carpeta de proyectos: {projects_root}")

    candidates: list[tuple[datetime, Path]] = []
    for entry in projects_root.iterdir():
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = _load_json(meta_path)
            updated = _safe_iso_to_dt(meta.get("updated_at"))
        except Exception:
            updated = datetime.fromtimestamp(meta_path.stat().st_mtime, tz=timezone.utc)
        candidates.append((updated, entry))

    if not candidates:
        raise RuntimeError(f"No se encontraron proyectos con meta.json en {projects_root}")

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _resolve_project_dir(projects_root: Path, project_id: str | None) -> Path:
    if project_id:
        project_dir = projects_root / str(project_id).strip()
        if not project_dir.exists() or not project_dir.is_dir():
            raise RuntimeError(f"No existe el proyecto solicitado: {project_dir}")
        return project_dir
    return _find_latest_project_dir(projects_root)


def _resolve_candidate(raw_path: Any, base_dir: Path) -> Path | None:
    if not raw_path:
        return None
    raw = str(raw_path).strip()
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    if candidate.exists():
        return candidate
    return None


def _copy_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _copy_optional_file(
    *,
    source_path: Path | None,
    destination_dir: Path,
    label: str,
    copied: dict[str, str],
) -> None:
    if source_path is None or not source_path.exists() or not source_path.is_file():
        return
    target = destination_dir / source_path.name
    _copy_file(source_path, target)
    copied[label] = str(target)


def _copy_optional_tree(
    *,
    source_dir: Path | None,
    destination_dir: Path,
    label: str,
    copied: dict[str, str],
) -> None:
    if source_dir is None or not source_dir.exists() or not source_dir.is_dir():
        return
    if destination_dir.exists():
        shutil.rmtree(destination_dir, ignore_errors=True)
    shutil.copytree(source_dir, destination_dir)
    copied[label] = str(destination_dir)


def _build_summary_markdown(
    *,
    project_id: str,
    meta: dict[str, Any],
    processing_metadata: dict[str, Any],
    quality_report: dict[str, Any] | None,
    fallback_report: dict[str, Any] | None,
    copied_paths: dict[str, str],
) -> str:
    metrics = quality_report.get("metrics") if isinstance((quality_report or {}).get("metrics"), dict) else {}
    classification = str(
        (quality_report or {}).get("quality_classification")
        or processing_metadata.get("quality_classification")
        or "unknown"
    )
    status = str(meta.get("status") or "unknown")
    engine = str(processing_metadata.get("engine") or processing_metadata.get("engine_requested") or "unknown")
    fallback_used = bool(
        processing_metadata.get("fallback_used")
        or (processing_metadata.get("fallback") or {}).get("used")
        or (processing_metadata.get("sparse_fallback") or {}).get("used")
    )
    recommended = str(
        (quality_report or {}).get("recommended_next_action")
        or processing_metadata.get("status_message")
        or ""
    ).strip()
    limitations = (quality_report or {}).get("limitations")
    limitations_lines = []
    if isinstance(limitations, list):
        limitations_lines = [f"- {str(item)}" for item in limitations if str(item).strip()]

    files_md = "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(copied_paths.items()))
    if not files_md:
        files_md = "- (sin archivos copiados)"

    fallback_line = "Si" if fallback_used else "No"
    fallback_reason = str((fallback_report or {}).get("reason_message") or "").strip()
    fallback_reason_md = fallback_reason if fallback_reason else "N/A"

    return (
        f"# DEFENSE SUMMARY - {project_id}\n\n"
        f"- Fecha UTC: {datetime.now(timezone.utc).isoformat()}\n"
        f"- Proyecto: `{project_id}`\n"
        f"- Estado final: `{status}`\n"
        f"- Motor: `{engine}`\n"
        f"- Clasificacion de calidad: `{classification}`\n"
        f"- Fallback usado: `{fallback_line}`\n"
        f"- Camaras reconstruidas: `{metrics.get('cameras_reconstructed', 'N/A')}`\n"
        f"- Puntos 3D: `{metrics.get('points_3d_count', 'N/A')}`\n"
        f"- Tiempo total (s): `{metrics.get('total_processing_seconds', 'N/A')}`\n\n"
        f"## Recomendacion tecnica\n\n"
        f"{recommended or 'Sin recomendacion reportada.'}\n\n"
        f"## Limitaciones\n\n"
        f"{chr(10).join(limitations_lines) if limitations_lines else '- Sin limitaciones explicitas en quality_report.'}\n\n"
        f"## Fallback (si aplica)\n\n"
        f"- Usado: `{fallback_line}`\n"
        f"- Motivo: {fallback_reason_md}\n\n"
        f"## Archivos incluidos\n\n"
        f"{files_md}\n"
    )


def generate_defense_package(
    *,
    project_id: str | None,
    projects_root: Path,
    experiments_reports: Path,
    output_root: Path,
    input_dir: Path | None,
) -> dict[str, Any]:
    projects_root = projects_root.resolve()
    experiments_reports = experiments_reports.resolve()
    output_root = output_root.resolve()
    project_dir = _resolve_project_dir(projects_root, project_id)
    project_id_resolved = project_dir.name

    meta_path = project_dir / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"No existe meta.json para el proyecto: {project_id_resolved}")
    meta = _load_json(meta_path)
    processing_metadata = (
        meta.get("processing_metadata")
        if isinstance(meta.get("processing_metadata"), dict)
        else {}
    )
    artifacts = (
        processing_metadata.get("artifacts")
        if isinstance(processing_metadata.get("artifacts"), dict)
        else {}
    )

    output_dir = project_dir / "output"
    pipeline_dir = output_dir / "pipeline"
    package_dir = output_root / project_id_resolved
    package_dir.mkdir(parents=True, exist_ok=True)

    copied_paths: dict[str, str] = {}
    missing_expected: list[str] = []

    quality_report_path = _resolve_candidate(
        artifacts.get("quality_report") or pipeline_dir / "quality_report.json",
        base_dir=Path.cwd(),
    )
    colmap_report_path = _resolve_candidate(
        artifacts.get("colmap_report") or pipeline_dir / "colmap_report.json",
        base_dir=Path.cwd(),
    )
    fallback_report_path = _resolve_candidate(
        artifacts.get("fallback_report") or pipeline_dir / "fallback_report.json",
        base_dir=Path.cwd(),
    )
    preprocessing_manifest_path = _resolve_candidate(
        artifacts.get("preprocessing_manifest") or pipeline_dir / "preprocessing_manifest.json",
        base_dir=Path.cwd(),
    )

    reports_dir = package_dir / "reports"
    _copy_optional_file(
        source_path=quality_report_path,
        destination_dir=reports_dir,
        label="quality_report",
        copied=copied_paths,
    )
    if "quality_report" not in copied_paths:
        missing_expected.append("quality_report")
    _copy_optional_file(
        source_path=colmap_report_path,
        destination_dir=reports_dir,
        label="colmap_report",
        copied=copied_paths,
    )
    if "colmap_report" not in copied_paths:
        missing_expected.append("colmap_report")
    _copy_optional_file(
        source_path=fallback_report_path,
        destination_dir=reports_dir,
        label="fallback_report",
        copied=copied_paths,
    )
    _copy_optional_file(
        source_path=preprocessing_manifest_path,
        destination_dir=reports_dir,
        label="preprocessing_manifest",
        copied=copied_paths,
    )
    if "preprocessing_manifest" not in copied_paths:
        missing_expected.append("preprocessing_manifest")

    if experiments_reports.exists():
        _copy_optional_file(
            source_path=(
                experiments_reports / "profile_comparison.json"
                if (experiments_reports / "profile_comparison.json").exists()
                else None
            ),
            destination_dir=package_dir / "experiments",
            label="profile_comparison_json",
            copied=copied_paths,
        )
        if "profile_comparison_json" not in copied_paths:
            missing_expected.append("profile_comparison_json")
        _copy_optional_file(
            source_path=(
                experiments_reports / "profile_comparison.csv"
                if (experiments_reports / "profile_comparison.csv").exists()
                else None
            ),
            destination_dir=package_dir / "experiments",
            label="profile_comparison_csv",
            copied=copied_paths,
        )
        if "profile_comparison_csv" not in copied_paths:
            missing_expected.append("profile_comparison_csv")
    else:
        missing_expected.extend(["profile_comparison_json", "profile_comparison_csv"])

    logs_dir = _resolve_candidate(
        (processing_metadata.get("logs") or {}).get("directory") if isinstance(processing_metadata.get("logs"), dict) else None,
        base_dir=Path.cwd(),
    )
    if logs_dir is None:
        candidate_logs = output_dir / "logs" / "colmap"
        logs_dir = candidate_logs if candidate_logs.exists() else None
    _copy_optional_tree(
        source_dir=logs_dir,
        destination_dir=package_dir / "logs" / "colmap",
        label="colmap_logs",
        copied=copied_paths,
    )
    if "colmap_logs" not in copied_paths:
        missing_expected.append("colmap_logs")

    model_path = _resolve_candidate(
        processing_metadata.get("final_model_path") or artifacts.get("model_path"),
        base_dir=Path.cwd(),
    )
    if model_path is None and meta.get("model_filename"):
        candidate = output_dir / str(meta["model_filename"])
        model_path = candidate if candidate.exists() else None
    _copy_optional_file(
        source_path=model_path,
        destination_dir=package_dir / "model",
        label="final_model",
        copied=copied_paths,
    )
    if "final_model" not in copied_paths:
        missing_expected.append("final_model")

    if input_dir is not None:
        resolved_input = input_dir.resolve()
        validation_report = resolved_input / "dataset_validation_report.json"
        _copy_optional_file(
            source_path=validation_report if validation_report.exists() else None,
            destination_dir=package_dir / "dataset",
            label="dataset_validation_report",
            copied=copied_paths,
        )

    quality_report = _load_json(quality_report_path) if quality_report_path and quality_report_path.exists() else None
    fallback_report = _load_json(fallback_report_path) if fallback_report_path and fallback_report_path.exists() else None

    summary_md = _build_summary_markdown(
        project_id=project_id_resolved,
        meta=meta,
        processing_metadata=processing_metadata,
        quality_report=quality_report,
        fallback_report=fallback_report,
        copied_paths=copied_paths,
    )
    summary_path = package_dir / "DEFENSE_SUMMARY.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    copied_paths["defense_summary"] = str(summary_path)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id_resolved,
        "project_dir": str(project_dir),
        "package_dir": str(package_dir),
        "copied_files": copied_paths,
        "missing_expected_evidence": sorted(set(missing_expected)),
    }
    manifest_path = package_dir / "package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    copied_paths["package_manifest"] = str(manifest_path)
    manifest["copied_files"] = copied_paths
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> int:
    args = _build_parser().parse_args()
    manifest = generate_defense_package(
        project_id=args.project_id,
        projects_root=args.projects_root,
        experiments_reports=args.experiments_reports,
        output_root=args.output_root,
        input_dir=args.input,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
