from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services.reconstruction_calibration import (
    append_history_record,
    predict_success_from_metrics,
)
from scripts.train_reconstruction_calibrator import _load_history, _train_thresholds
from scripts.validate_real_dataset import validate_dataset


def _write_test_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("failed to encode test image")
    path.write_bytes(encoded.tobytes())


def test_append_history_record(tmp_path: Path) -> None:
    history = tmp_path / "history.ndjson"
    append_history_record(history, {"project_id": "p1", "quality_classification": "success_sparse_only"})
    append_history_record(history, {"project_id": "p2", "quality_classification": "success_real"})
    rows = _load_history(history)
    assert len(rows) == 2
    assert rows[0]["project_id"] == "p1"
    assert rows[1]["project_id"] == "p2"


def test_train_thresholds_with_few_data() -> None:
    rows = [
        {"mesh_readiness_score": 0.35, "quality_classification": "success_sparse_only"},
        {"mesh_readiness_score": 0.55, "quality_classification": "success_approx_surface"},
        {"mesh_readiness_score": 0.78, "quality_classification": "success_real"},
    ]
    model = _train_thresholds(rows)
    assert model["method"] == "heuristic_thresholds"
    assert model["thresholds"]["dense_real_min_score"] >= model["thresholds"]["approx_surface_min_score"]


def test_predictor_with_simulated_data() -> None:
    metrics = {
        "mesh_readiness_score": 0.72,
        "angular_coverage_score": 0.68,
        "visual_variety_score": 0.52,
        "average_feature_points": 135.0,
        "usable_images_ratio": 0.82,
        "image_count_valid": 52,
    }
    prediction = predict_success_from_metrics(metrics, calibrator={"method": "heuristic_thresholds"})
    probs = prediction["predicted_success_probabilities"]
    assert set(probs.keys()) == {"sparse", "approx_surface", "dense_real"}
    assert abs(sum(float(v) for v in probs.values()) - 1.0) < 1e-5


def test_validate_dataset_includes_prediction_fields(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    for idx in range(6):
        _write_test_image(dataset / f"img_{idx:02d}.jpg", seed=idx + 10)

    report = validate_dataset(
        input_dir=dataset,
        min_images=4,
        max_images=20,
        duplicate_hamming_threshold=6,
        max_duplicate_ratio=0.2,
    )
    assert "predicted_success_level" in report
    assert "predicted_success_probabilities" in report
    assert "recommended_profile" in report
    assert "should_run_colmap" in report
