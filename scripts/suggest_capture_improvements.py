from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera recomendaciones de captura a partir de dataset_validation_report.json"
    )
    parser.add_argument("--report", required=True, type=Path, help="Ruta del reporte JSON.")
    return parser


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"Reporte no encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _get_float(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except Exception:
        return default


def _get_int(data: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(data.get(key, default))
    except Exception:
        return default


def build_suggestions(report: dict[str, Any]) -> str:
    summary = dict(report.get("summary") or {})
    metrics = dict(report.get("metrics") or {})
    duplicates = dict(report.get("duplicates") or {})

    valid_images = _get_int(summary, "image_count_valid")
    avg_sharpness = _get_float(metrics, "average_sharpness")
    avg_features = _get_float(metrics, "average_feature_points")
    usable_ratio = _get_float(metrics, "usable_images_ratio")
    duplicate_ratio = _get_float(duplicates, "possible_duplicates_ratio")
    visual_variety = _get_float(metrics, "visual_variety_score")
    angular_coverage = _get_float(metrics, "angular_coverage_score")
    readiness = _get_float(metrics, "mesh_readiness_score")

    missing_for_45 = max(0, 45 - valid_images)
    missing_for_60 = max(0, 60 - valid_images)

    lines: list[str] = []
    lines.append("SUGERENCIAS DE CAPTURA PARA MEJORAR MALLA")
    lines.append(f"- Imagenes validas actuales: {valid_images}")
    lines.append(f"- Mesh readiness score: {readiness:.3f} (0 a 1)")
    lines.append("")
    lines.append("BRECHAS PRINCIPALES")
    lines.append(f"- Fotos faltantes para 45: {missing_for_45}")
    lines.append(f"- Fotos faltantes para 60: {missing_for_60}")

    if avg_features < 90:
        lines.append(
            "- Falta textura/feature points: usa objeto o fondo con mas patron visual."
        )
    if duplicate_ratio > 0.12:
        lines.append(
            "- Hay demasiados duplicados: evita rafagas en el mismo angulo y aumenta desplazamiento entre fotos."
        )
    if avg_sharpness < 60:
        lines.append(
            "- Nitidez baja: estabiliza camara, mejora luz uniforme y evita desenfoque por movimiento."
        )
    if usable_ratio < 0.6:
        lines.append(
            "- Pocas imagenes utiles: repite captura descartando borrosas, oscuras o sobreexpuestas."
        )
    if angular_coverage < 0.5:
        lines.append(
            "- Cobertura angular baja: agrega tomas desde arriba, altura media y abajo alrededor de 360 grados."
        )
    if visual_variety < 0.45:
        lines.append(
            "- Variedad visual baja: cambia ligeramente distancia/angulo manteniendo overlap 70-80%."
        )
    if avg_features < 60 and duplicate_ratio > 0.2:
        lines.append(
            "- Conviene cambiar objeto o preparar superficie con textura no brillante para mejorar reconstruccion."
        )

    lines.append("")
    lines.append("RECOMENDACION FINAL")
    if readiness >= 0.65 and valid_images >= 45:
        lines.append("- Dataset con buena opcion para intentar dense real.")
    elif readiness >= 0.45 and valid_images >= 30:
        lines.append("- Dataset apto para superficie aproximada desde sparse.")
    else:
        lines.append("- Dataset probablemente se quedara en sparse; recaptura antes de corrida final.")

    return "\n".join(lines)


def main() -> int:
    args = _build_parser().parse_args()
    report = _load_report(args.report.resolve())
    print(build_suggestions(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
