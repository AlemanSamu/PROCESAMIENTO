from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CALIBRATOR_PATH_DEFAULT = Path("data/experiments/models/reconstruction_calibrator.json")
HISTORY_PATH_DEFAULT = Path("data/experiments/reconstruction_history.ndjson")


def load_calibrator(path: Path | None = None) -> dict[str, Any]:
    target = path or CALIBRATOR_PATH_DEFAULT
    if not target.exists() or not target.is_file():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}


def predict_success_from_metrics(
    metrics: dict[str, Any],
    calibrator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    calibrator_payload = calibrator or {}
    method = str(calibrator_payload.get("method") or "heuristic_thresholds").strip().lower()
    thresholds = dict(calibrator_payload.get("thresholds") or {})

    readiness = float(metrics.get("mesh_readiness_score") or 0.0)
    angular = float(metrics.get("angular_coverage_score") or 0.0)
    variety = float(metrics.get("visual_variety_score") or 0.0)
    features = float(metrics.get("average_feature_points") or 0.0)
    usable = float(metrics.get("usable_images_ratio") or 0.0)
    image_count = float(metrics.get("image_count_valid") or metrics.get("image_count_total") or 0.0)

    dense_threshold = float(thresholds.get("dense_real_min_score") or 0.67)
    surface_threshold = float(thresholds.get("approx_surface_min_score") or 0.46)

    if method == "sklearn_logreg" and "coefficients" in calibrator_payload:
        coeffs = dict(calibrator_payload.get("coefficients") or {})
        dense_logit = (
            float(coeffs.get("bias_dense", -1.2))
            + readiness * float(coeffs.get("w_readiness_dense", 2.7))
            + angular * float(coeffs.get("w_angular_dense", 1.1))
            + variety * float(coeffs.get("w_variety_dense", 0.8))
            + usable * float(coeffs.get("w_usable_dense", 1.2))
            + min(features / 200.0, 1.2) * float(coeffs.get("w_features_dense", 0.9))
            + min(image_count / 60.0, 1.2) * float(coeffs.get("w_images_dense", 0.9))
        )
        surface_logit = (
            float(coeffs.get("bias_surface", -0.8))
            + readiness * float(coeffs.get("w_readiness_surface", 2.2))
            + angular * float(coeffs.get("w_angular_surface", 0.9))
            + variety * float(coeffs.get("w_variety_surface", 0.7))
            + usable * float(coeffs.get("w_usable_surface", 0.9))
            + min(features / 180.0, 1.2) * float(coeffs.get("w_features_surface", 0.8))
            + min(image_count / 50.0, 1.2) * float(coeffs.get("w_images_surface", 0.8))
        )
        dense_prob = 1.0 / (1.0 + (2.718281828 ** (-dense_logit)))
        surface_prob_raw = 1.0 / (1.0 + (2.718281828 ** (-surface_logit)))
        surface_prob = max(0.0, min(1.0, surface_prob_raw * (1.0 - dense_prob * 0.6)))
        sparse_prob = max(0.0, min(1.0, 1.0 - dense_prob - surface_prob))
    else:
        dense_prob = max(
            0.0,
            min(
                1.0,
                (readiness - dense_threshold + 0.22) * 1.3
                + angular * 0.15
                + usable * 0.15,
            ),
        )
        surface_prob = max(
            0.0,
            min(
                1.0,
                (readiness - surface_threshold + 0.28) * 1.2
                + variety * 0.12
                + min(features / 220.0, 1.0) * 0.12,
            ),
        )
        if dense_prob > 0.45:
            surface_prob *= 0.75
        sparse_prob = max(0.0, min(1.0, 1.0 - max(dense_prob, surface_prob) * 0.75))

    total = max(1e-9, dense_prob + surface_prob + sparse_prob)
    probs = {
        "sparse": round(sparse_prob / total, 6),
        "approx_surface": round(surface_prob / total, 6),
        "dense_real": round(dense_prob / total, 6),
    }
    predicted = max(probs.items(), key=lambda item: item[1])[0]
    recommended_profile = "quality" if probs["dense_real"] >= 0.3 or probs["approx_surface"] >= 0.45 else "balanced"
    should_run_colmap = bool(
        probs["dense_real"] >= 0.2 or probs["approx_surface"] >= 0.35 or probs["sparse"] >= 0.5
    )
    return {
        "predicted_success_level": predicted,
        "predicted_success_probabilities": probs,
        "recommended_profile": recommended_profile,
        "should_run_colmap": should_run_colmap,
    }


def append_history_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")


def to_final_success_level(quality_classification: str) -> str:
    value = str(quality_classification or "").strip().lower()
    if value == "success_real":
        return "dense_real"
    if value == "success_approx_surface":
        return "approx_surface"
    if value == "success_sparse_only":
        return "sparse_only"
    if value == "fallback_completed":
        return "fallback_only"
    return "failed"
