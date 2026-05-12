from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.services.reconstruction_calibration import (
    CALIBRATOR_PATH_DEFAULT,
    load_calibrator,
    predict_success_from_metrics,
)

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valida un dataset real de imagenes para reconstruccion 3D."
    )
    parser.add_argument("--input", required=True, type=Path, help="Carpeta de imagenes.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta opcional de salida JSON. Por defecto: <input>/dataset_validation_report.json",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=20,
        help="Cantidad minima recomendada para clasificar como apto.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=40,
        help="Cantidad maxima recomendada para clasificar como apto.",
    )
    parser.add_argument(
        "--duplicate-hamming-threshold",
        type=int,
        default=6,
        help="Distancia Hamming maxima para marcar posibles duplicados.",
    )
    parser.add_argument(
        "--max-duplicate-ratio",
        type=float,
        default=0.12,
        help="Ratio maximo recomendado de pares duplicados sobre total de pares.",
    )
    return parser


def _collect_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError(f"La carpeta de entrada no existe: {input_dir}")

    images = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    )
    return images


def _read_image(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None


def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _dhash64(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    hash_value = 0
    for bit in diff.flatten():
        hash_value = (hash_value << 1) | int(bool(bit))
    return hash_value


def _phash64(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low_freq = dct[:8, :8]
    median_value = float(np.median(low_freq[1:, 1:]))
    bits = low_freq > median_value
    hash_value = 0
    for bit in bits.flatten():
        hash_value = (hash_value << 1) | int(bool(bit))
    return hash_value


def _normalize_for_hash(gray: np.ndarray) -> np.ndarray:
    # Robust pre-normalization for low-quality photos:
    # denoise + local contrast equalization before hashing.
    denoised = cv2.fastNlMeansDenoising(gray, None, 6, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(denoised)


def _build_orb_features(gray: np.ndarray) -> tuple[int, np.ndarray | None]:
    orb = cv2.ORB_create(nfeatures=300)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return len(keypoints or []), descriptors


def _orb_match_ratio(
    desc_a: np.ndarray | None,
    kpts_a: int,
    desc_b: np.ndarray | None,
    kpts_b: int,
) -> float:
    if desc_a is None or desc_b is None or kpts_a <= 0 or kpts_b <= 0:
        return 0.0
    try:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(desc_a, desc_b)
    except Exception:
        return 0.0

    if not matches:
        return 0.0
    good_matches = [match for match in matches if match.distance <= 64]
    denominator = float(max(1, min(kpts_a, kpts_b)))
    return float(len(good_matches)) / denominator


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _score_linear(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    return _clamp01((float(value) - float(min_value)) / (float(max_value) - float(min_value)))


def _estimate_angular_coverage(per_image: list[dict[str, Any]]) -> float:
    if not per_image:
        return 0.0
    values = [int(str(item.get("phash64_hex", "0")), 16) for item in per_image]
    unique_bins = {int(value % 12) for value in values}
    return _clamp01(len(unique_bins) / 12.0)


def _estimate_visual_variety(
    *,
    duplicate_ratio: float,
    sharpness_values: list[float],
    brightness_values: list[float],
    feature_points_values: list[int],
) -> float:
    duplicate_component = 1.0 - _clamp01(duplicate_ratio / 0.35)
    sharpness_cv = 0.0
    brightness_cv = 0.0
    feature_cv = 0.0
    if sharpness_values:
        sharp_mean = _safe_mean(sharpness_values)
        sharp_std = float(np.std(np.array(sharpness_values, dtype=np.float32)))
        sharpness_cv = sharp_std / max(1.0, sharp_mean)
    if brightness_values:
        br_mean = _safe_mean(brightness_values)
        br_std = float(np.std(np.array(brightness_values, dtype=np.float32)))
        brightness_cv = br_std / max(0.05, br_mean)
    if feature_points_values:
        feat_mean = _safe_mean([float(v) for v in feature_points_values])
        feat_std = float(np.std(np.array(feature_points_values, dtype=np.float32)))
        feature_cv = feat_std / max(1.0, feat_mean)

    variability_component = _clamp01((sharpness_cv + brightness_cv + feature_cv) / 1.2)
    return _clamp01((duplicate_component * 0.6) + (variability_component * 0.4))


def _mesh_readiness_score(
    *,
    image_count: int,
    avg_sharpness: float,
    avg_feature_points: float,
    duplicate_ratio: float,
    usable_images_ratio: float,
    visual_variety_score: float,
    angular_coverage_score: float,
) -> float:
    image_count_score = _score_linear(float(image_count), 20.0, 60.0)
    sharpness_score = _score_linear(avg_sharpness, 35.0, 140.0)
    feature_score = _score_linear(avg_feature_points, 60.0, 220.0)
    duplicate_score = 1.0 - _clamp01(duplicate_ratio / 0.35)
    usable_score = _clamp01(usable_images_ratio)

    score = (
        image_count_score * 0.18
        + sharpness_score * 0.18
        + feature_score * 0.22
        + duplicate_score * 0.16
        + usable_score * 0.14
        + _clamp01(visual_variety_score) * 0.07
        + _clamp01(angular_coverage_score) * 0.05
    )
    return round(_clamp01(score), 6)


def _dataset_mesh_recommendation(
    *,
    mesh_readiness_score: float,
    image_count: int,
    usable_images_ratio: float,
    avg_feature_points: float,
    angular_coverage_score: float,
) -> tuple[dict[str, bool], str]:
    apto_para_sparse = (
        image_count >= 20 and usable_images_ratio >= 0.45 and avg_feature_points >= 55.0
    )
    apto_para_superficie = (
        image_count >= 30
        and usable_images_ratio >= 0.6
        and avg_feature_points >= 90.0
        and angular_coverage_score >= 0.45
        and mesh_readiness_score >= 0.45
    )
    apto_para_dense = (
        image_count >= 45
        and usable_images_ratio >= 0.75
        and avg_feature_points >= 120.0
        and angular_coverage_score >= 0.6
        and mesh_readiness_score >= 0.65
    )

    if apto_para_dense:
        recommendation = "Este dataset puede intentar dense real."
    elif apto_para_superficie:
        recommendation = "Este dataset puede dar superficie aproximada."
    else:
        recommendation = "Este dataset probablemente solo dara sparse."

    return (
        {
            "apto_para_sparse": bool(apto_para_sparse),
            "apto_para_superficie": bool(apto_para_superficie),
            "apto_para_dense": bool(apto_para_dense),
        },
        recommendation,
    )


def _evaluate_recommendation(
    *,
    image_count: int,
    min_images: int,
    max_images: int,
    avg_megapixels: float,
    avg_sharpness: float,
    avg_brightness: float,
    duplicate_ratio: float,
    max_duplicate_ratio: float,
    unreadable_count: int,
    usable_images_ratio: float,
) -> tuple[str, str, list[str]]:
    warnings: list[str] = []

    if image_count == 0:
        return (
            "no_apto",
            "No se pudieron leer imagenes validas en el dataset. Revisa formato de archivos y vuelve a capturar con buena iluminacion.",
            ["dataset_vacio_o_ilegible"],
        )

    if unreadable_count > 0:
        warnings.append(f"imagenes_no_legibles:{unreadable_count}")
    if image_count < min_images:
        warnings.append(f"pocas_imagenes:{image_count} (< {min_images})")
    if image_count > max_images:
        warnings.append(f"muchas_imagenes:{image_count} (> {max_images})")
    if avg_megapixels < 0.8:
        warnings.append(f"resolucion_promedio_baja:{avg_megapixels:.2f}MP")
    if avg_sharpness < 60.0:
        warnings.append(f"nitidez_promedio_baja:{avg_sharpness:.2f}")
    if avg_brightness < 0.20 or avg_brightness > 0.85:
        warnings.append(f"brillo_promedio_fuera_rango:{avg_brightness:.3f}")
    if duplicate_ratio > max_duplicate_ratio:
        warnings.append(
            f"duplicados_altos:{duplicate_ratio:.3f} (> {max_duplicate_ratio:.3f})"
        )
    if usable_images_ratio < 0.5:
        warnings.append("baja_cantidad_imagenes_utiles")

    severe = False
    if image_count < max(8, min_images // 2):
        severe = True
    if avg_megapixels < 0.4:
        severe = True
    if avg_sharpness < 25.0:
        severe = True
    if avg_brightness < 0.12 or avg_brightness > 0.92:
        severe = True
    if duplicate_ratio > 0.45:
        severe = True
    if unreadable_count >= max(3, image_count // 4):
        severe = True

    problem_reasons: list[str] = []
    if image_count < min_images:
        problem_reasons.append("pocas imagenes")
    if avg_sharpness < 60.0:
        problem_reasons.append("nitidez baja")
    if avg_brightness < 0.20 or avg_brightness > 0.85:
        problem_reasons.append("iluminacion no uniforme")
    if duplicate_ratio > max_duplicate_ratio:
        problem_reasons.append("exceso de imagenes muy similares/duplicadas")
    if usable_images_ratio < 0.5:
        problem_reasons.append("pocas imagenes realmente utiles")

    reasons_text = ", ".join(problem_reasons) if problem_reasons else "sin problemas criticos detectados"

    if severe:
        return (
            "no_apto",
            (
                "El dataset presenta problemas severos y se recomienda recapturar imagenes. "
                f"Problemas detectados: {reasons_text}."
            ),
            warnings,
        )

    has_warnings = bool(warnings)
    if has_warnings:
        return (
            "mejorar",
            (
                "El dataset es util, pero conviene mejorarlo antes de la corrida final. "
                f"Ajustes sugeridos: {reasons_text}."
            ),
            warnings,
        )

    return (
        "apto",
        "El dataset cumple criterios recomendados para prueba real con COLMAP (nitidez, iluminacion y variacion de vistas en rango aceptable).",
        warnings,
    )


def validate_dataset(
    *,
    input_dir: Path,
    min_images: int,
    max_images: int,
    duplicate_hamming_threshold: int,
    max_duplicate_ratio: float,
) -> dict[str, Any]:
    image_paths = _collect_images(input_dir)
    image_count_total = len(image_paths)

    per_image: list[dict[str, Any]] = []
    widths: list[float] = []
    heights: list[float] = []
    megapixels: list[float] = []
    brightness_values: list[float] = []
    sharpness_values: list[float] = []
    feature_points_values: list[int] = []
    hash_items: list[dict[str, Any]] = []
    unreadable_files: list[str] = []
    usable_images_count = 0

    for path in image_paths:
        image = _read_image(path)
        if image is None:
            unreadable_files.append(path.name)
            continue

        height, width = image.shape[:2]
        gray = _to_gray(image)
        brightness = float(gray.mean() / 255.0)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        normalized = _normalize_for_hash(gray)
        dhash_value = _dhash64(normalized)
        phash_value = _phash64(normalized)
        keypoint_count, descriptors = _build_orb_features(normalized)

        widths.append(float(width))
        heights.append(float(height))
        megapixels.append(float((width * height) / 1_000_000.0))
        brightness_values.append(brightness)
        sharpness_values.append(sharpness)
        feature_points_values.append(int(keypoint_count))
        hash_items.append(
            {
                "name": path.name,
                "dhash64": dhash_value,
                "phash64": phash_value,
                "keypoint_count": int(keypoint_count),
                "descriptors": descriptors,
            }
        )

        confidence = (
            "high"
            if sharpness >= 60.0 and keypoint_count >= 100
            else "medium"
            if sharpness >= 30.0 and keypoint_count >= 50
            else "low"
        )
        if confidence != "low":
            usable_images_count += 1

        per_image.append(
            {
                "file": path.name,
                "width": int(width),
                "height": int(height),
                "brightness": round(brightness, 6),
                "sharpness": round(sharpness, 6),
                "dhash64_hex": f"{dhash_value:016x}",
                "phash64_hex": f"{phash_value:016x}",
                "feature_points": int(keypoint_count),
                "identification_confidence": confidence,
            }
        )

    duplicate_pairs: list[dict[str, Any]] = []
    for i in range(len(hash_items)):
        item_a = hash_items[i]
        for j in range(i + 1, len(hash_items)):
            item_b = hash_items[j]
            dhash_distance = _hamming_distance(int(item_a["dhash64"]), int(item_b["dhash64"]))
            phash_distance = _hamming_distance(int(item_a["phash64"]), int(item_b["phash64"]))
            orb_ratio = _orb_match_ratio(
                item_a.get("descriptors"),
                int(item_a.get("keypoint_count") or 0),
                item_b.get("descriptors"),
                int(item_b.get("keypoint_count") or 0),
            )

            # Hybrid duplicate check for low-quality photos:
            # 1) both perceptual hashes are very close, or
            # 2) one hash is close and ORB says both images are very similar.
            close_hashes = (
                dhash_distance <= duplicate_hamming_threshold
                and phash_distance <= max(duplicate_hamming_threshold + 2, 8)
            )
            hash_plus_orb = (
                (dhash_distance <= duplicate_hamming_threshold or phash_distance <= duplicate_hamming_threshold + 4)
                and orb_ratio >= 0.72
            )

            if (close_hashes and orb_ratio >= 0.45) or hash_plus_orb:
                duplicate_pairs.append(
                    {
                        "a": str(item_a["name"]),
                        "b": str(item_b["name"]),
                        "dhash_distance": int(dhash_distance),
                        "phash_distance": int(phash_distance),
                        "orb_match_ratio": round(float(orb_ratio), 6),
                    }
                )

    valid_count = len(per_image)
    total_pairs = (valid_count * (valid_count - 1)) // 2
    duplicate_ratio = (
        float(len(duplicate_pairs)) / float(total_pairs)
        if total_pairs > 0
        else 0.0
    )

    avg_width = _safe_mean(widths)
    avg_height = _safe_mean(heights)
    avg_megapixels = _safe_mean(megapixels)
    avg_brightness = _safe_mean(brightness_values)
    avg_sharpness = _safe_mean(sharpness_values)
    usable_images_ratio = float(usable_images_count) / float(valid_count) if valid_count > 0 else 0.0
    avg_feature_points = _safe_mean([float(v) for v in feature_points_values])
    high_count = sum(1 for item in per_image if item.get("identification_confidence") == "high")
    medium_count = sum(1 for item in per_image if item.get("identification_confidence") == "medium")
    low_count = sum(1 for item in per_image if item.get("identification_confidence") == "low")
    high_ratio = float(high_count) / float(valid_count) if valid_count > 0 else 0.0
    medium_ratio = float(medium_count) / float(valid_count) if valid_count > 0 else 0.0
    low_ratio = float(low_count) / float(valid_count) if valid_count > 0 else 0.0
    angular_coverage_score = _estimate_angular_coverage(per_image)
    visual_variety_score = _estimate_visual_variety(
        duplicate_ratio=duplicate_ratio,
        sharpness_values=sharpness_values,
        brightness_values=brightness_values,
        feature_points_values=feature_points_values,
    )
    mesh_readiness_score = _mesh_readiness_score(
        image_count=valid_count,
        avg_sharpness=avg_sharpness,
        avg_feature_points=avg_feature_points,
        duplicate_ratio=duplicate_ratio,
        usable_images_ratio=usable_images_ratio,
        visual_variety_score=visual_variety_score,
        angular_coverage_score=angular_coverage_score,
    )
    mesh_recommendation_flags, mesh_recommendation_message = _dataset_mesh_recommendation(
        mesh_readiness_score=mesh_readiness_score,
        image_count=valid_count,
        usable_images_ratio=usable_images_ratio,
        avg_feature_points=avg_feature_points,
        angular_coverage_score=angular_coverage_score,
    )

    recommendation, recommendation_message, warnings = _evaluate_recommendation(
        image_count=valid_count,
        min_images=min_images,
        max_images=max_images,
        avg_megapixels=avg_megapixels,
        avg_sharpness=avg_sharpness,
        avg_brightness=avg_brightness,
        duplicate_ratio=duplicate_ratio,
        max_duplicate_ratio=max_duplicate_ratio,
        unreadable_count=len(unreadable_files),
        usable_images_ratio=usable_images_ratio,
    )

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "summary": {
            "image_count_total": int(image_count_total),
            "image_count_valid": int(valid_count),
            "image_count_unreadable": int(len(unreadable_files)),
        },
        "metrics": {
            "average_width": round(avg_width, 3),
            "average_height": round(avg_height, 3),
            "average_megapixels": round(avg_megapixels, 6),
            "average_sharpness": round(avg_sharpness, 6),
            "average_brightness": round(avg_brightness, 6),
            "average_feature_points": round(avg_feature_points, 6),
            "usable_images_ratio": round(usable_images_ratio, 6),
            "high_confidence_ratio": round(high_ratio, 6),
            "medium_confidence_ratio": round(medium_ratio, 6),
            "low_confidence_ratio": round(low_ratio, 6),
            "visual_variety_score": round(visual_variety_score, 6),
            "angular_coverage_score": round(angular_coverage_score, 6),
            "mesh_readiness_score": mesh_readiness_score,
        },
        "duplicates": {
            "hamming_threshold": int(duplicate_hamming_threshold),
            "possible_duplicates_count": int(len(duplicate_pairs)),
            "possible_duplicates_ratio": round(duplicate_ratio, 6),
            "possible_duplicates": duplicate_pairs[:200],
            "total_pairs_checked": int(total_pairs),
            "detection_method": "hybrid_dhash_phash_orb",
        },
        "warnings": warnings,
        "recommendation": recommendation,
        "recommendation_message": recommendation_message,
        "mesh_recommendation": mesh_recommendation_flags,
        "mesh_recommendation_message": mesh_recommendation_message,
        "unreadable_files": unreadable_files,
        "images": per_image,
    }
    prediction_input = {
        "mesh_readiness_score": mesh_readiness_score,
        "angular_coverage_score": angular_coverage_score,
        "visual_variety_score": visual_variety_score,
        "average_feature_points": avg_feature_points,
        "usable_images_ratio": usable_images_ratio,
        "image_count_valid": valid_count,
        "image_count_total": image_count_total,
    }
    prediction = predict_success_from_metrics(
        prediction_input,
        calibrator=load_calibrator(CALIBRATOR_PATH_DEFAULT),
    )
    report.update(prediction)
    return report


def main() -> int:
    args = _build_parser().parse_args()
    input_dir = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_dir / "dataset_validation_report.json"
    )

    report = validate_dataset(
        input_dir=input_dir,
        min_images=max(1, int(args.min_images)),
        max_images=max(1, int(args.max_images)),
        duplicate_hamming_threshold=max(0, int(args.duplicate_hamming_threshold)),
        max_duplicate_ratio=max(0.0, float(args.max_duplicate_ratio)),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[saved] {output_path}")

    recommendation = str(report.get("recommendation") or "").strip().lower()
    return 1 if recommendation == "no_apto" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
