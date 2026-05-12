from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena/calibra predictor de exito de reconstruccion.")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("data/experiments/reconstruction_history.ndjson"),
        help="Ruta al historial ndjson.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/models/reconstruction_calibrator.json"),
        help="Ruta de salida del calibrador.",
    )
    return parser


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _as_label(row: dict[str, Any]) -> str:
    level = str(row.get("final_success_level") or "").strip().lower()
    if level in {"dense_real", "approx_surface", "sparse_only"}:
        return level
    quality = str(row.get("quality_classification") or "").strip().lower()
    if quality == "success_real":
        return "dense_real"
    if quality == "success_approx_surface":
        return "approx_surface"
    return "sparse_only"


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _train_thresholds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_label[_as_label(row)].append(float(row.get("mesh_readiness_score") or 0.0))

    sparse_avg = _mean(by_label.get("sparse_only", []))
    surface_avg = _mean(by_label.get("approx_surface", []))
    dense_avg = _mean(by_label.get("dense_real", []))

    dense_min = max(0.45, min(0.9, (surface_avg + dense_avg) / 2.0 if dense_avg > 0 else 0.67))
    surface_min = max(0.25, min(0.8, (sparse_avg + surface_avg) / 2.0 if surface_avg > 0 else 0.46))

    return {
        "method": "heuristic_thresholds",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(rows),
        "thresholds": {
            "approx_surface_min_score": round(surface_min, 6),
            "dense_real_min_score": round(dense_min, 6),
        },
    }


def _train_sklearn(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore
    except Exception:
        return None

    if len(rows) < 35:
        return None

    x: list[list[float]] = []
    y_dense: list[int] = []
    y_surface: list[int] = []
    for row in rows:
        x.append(
            [
                float(row.get("mesh_readiness_score") or 0.0),
                float(row.get("angular_coverage_score") or 0.0),
                float(row.get("visual_variety_score") or 0.0),
                float(row.get("average_feature_points") or 0.0),
                float(row.get("usable_images_ratio") or 0.0),
                float(row.get("image_count") or 0.0),
            ]
        )
        label = _as_label(row)
        y_dense.append(1 if label == "dense_real" else 0)
        y_surface.append(1 if label == "approx_surface" else 0)

    if len(set(y_dense)) < 2 or len(set(y_surface)) < 2:
        return None

    dense_model = LogisticRegression(max_iter=500)
    dense_model.fit(x, y_dense)
    surface_model = LogisticRegression(max_iter=500)
    surface_model.fit(x, y_surface)

    dense_coef = dense_model.coef_[0]
    surface_coef = surface_model.coef_[0]
    return {
        "method": "sklearn_logreg",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(rows),
        "coefficients": {
            "bias_dense": float(dense_model.intercept_[0]),
            "w_readiness_dense": float(dense_coef[0]),
            "w_angular_dense": float(dense_coef[1]),
            "w_variety_dense": float(dense_coef[2]),
            "w_features_dense": float(dense_coef[3]),
            "w_usable_dense": float(dense_coef[4]),
            "w_images_dense": float(dense_coef[5]),
            "bias_surface": float(surface_model.intercept_[0]),
            "w_readiness_surface": float(surface_coef[0]),
            "w_angular_surface": float(surface_coef[1]),
            "w_variety_surface": float(surface_coef[2]),
            "w_features_surface": float(surface_coef[3]),
            "w_usable_surface": float(surface_coef[4]),
            "w_images_surface": float(surface_coef[5]),
        },
    }


def main() -> int:
    args = _build_parser().parse_args()
    rows = _load_history(args.history.resolve())
    model = _train_sklearn(rows) or _train_thresholds(rows)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(model, indent=2, ensure_ascii=False))
    print(f"[saved] {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
