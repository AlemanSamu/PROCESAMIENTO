from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.services.reconstruction_calibration import (
    CALIBRATOR_PATH_DEFAULT,
    load_calibrator,
    predict_success_from_metrics,
)
from scripts.validate_real_dataset import validate_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predice probabilidad de exito antes de correr COLMAP.")
    parser.add_argument("--dataset", required=True, type=Path, help="Carpeta de imagenes.")
    parser.add_argument(
        "--calibrator",
        type=Path,
        default=CALIBRATOR_PATH_DEFAULT,
        help="Ruta al calibrador json.",
    )
    return parser


def _suggest_actions(report: dict[str, Any]) -> list[str]:
    metrics = dict(report.get("metrics") or {})
    probs = dict(report.get("predicted_success_probabilities") or {})
    suggestions: list[str] = []
    if int((report.get("summary") or {}).get("image_count_valid") or 0) < 45:
        suggestions.append("capturar mas fotos (objetivo: 45 a 60)")
    if float(metrics.get("average_feature_points") or 0.0) < 90.0:
        suggestions.append("usar objeto con mas textura")
    if float(metrics.get("average_sharpness") or 0.0) < 60.0:
        suggestions.append("mejorar iluminacion y estabilidad para evitar blur")
    if float(metrics.get("angular_coverage_score") or 0.0) < 0.5:
        suggestions.append("cambiar angulos y agregar vistas arriba/abajo")
    if float((report.get("duplicates") or {}).get("possible_duplicates_ratio") or 0.0) > 0.12:
        suggestions.append("evitar duplicados consecutivos y mantener overlap 70-80%")
    if float(probs.get("dense_real") or 0.0) < 0.25:
        suggestions.append("usar perfil quality para maximizar oportunidad de malla")
    if not suggestions:
        suggestions.append("ejecutar COLMAP con perfil quality")
    return suggestions


def main() -> int:
    args = _build_parser().parse_args()
    dataset = args.dataset.resolve()
    report = validate_dataset(
        input_dir=dataset,
        min_images=20,
        max_images=60,
        duplicate_hamming_threshold=6,
        max_duplicate_ratio=0.12,
    )
    prediction = predict_success_from_metrics(
        {
            **dict(report.get("metrics") or {}),
            "image_count_valid": int((report.get("summary") or {}).get("image_count_valid") or 0),
            "image_count_total": int((report.get("summary") or {}).get("image_count_total") or 0),
        },
        calibrator=load_calibrator(args.calibrator.resolve()),
    )
    report.update(prediction)
    report["next_actions"] = _suggest_actions(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
