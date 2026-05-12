from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analiza historial de reconstruccion.")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("data/experiments/reconstruction_history.ndjson"),
        help="Ruta al historial ndjson.",
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
            row = json.loads(raw)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> int:
    args = _build_parser().parse_args()
    rows = _load_history(args.history.resolve())
    if not rows:
        print("No hay corridas en el historial.")
        return 0

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_profile: dict[str, int] = defaultdict(int)
    points_by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        cls = str(row.get("quality_classification") or "unknown")
        by_class[cls].append(row)
        by_profile[str(row.get("profile") or "unknown")] += 1
        points_by_dataset[str(row.get("dataset_path") or "unknown")].append(float(row.get("points3D_count") or 0.0))

    print(f"corridas_totales: {len(rows)}")
    print("clasificacion_top:")
    for cls, items in sorted(by_class.items(), key=lambda item: len(item[1]), reverse=True):
        print(f"- {cls}: {len(items)}")

    print("mejores_datasets_por_success_real:")
    dense_rows = [row for row in rows if str(row.get("quality_classification")) == "success_real"]
    dense_by_dataset: dict[str, int] = defaultdict(int)
    for row in dense_rows:
        dense_by_dataset[str(row.get("dataset_path") or "unknown")] += 1
    for dataset, count in sorted(dense_by_dataset.items(), key=lambda item: item[1], reverse=True)[:10]:
        print(f"- {dataset}: {count}")

    print("promedio_points3D_por_dataset:")
    for dataset, values in sorted(points_by_dataset.items(), key=lambda item: _mean(item[1]), reverse=True)[:10]:
        print(f"- {dataset}: {_mean(values):.2f}")

    print("perfil_mas_usado:")
    for profile, count in sorted(by_profile.items(), key=lambda item: item[1], reverse=True):
        print(f"- {profile}: {count}")

    dense_metrics = [row for row in rows if str(row.get("final_success_level") or "") == "dense_real"]
    if dense_metrics:
        print("condiciones_que_predicen_success_real:")
        print(f"- mesh_readiness_score_promedio: {_mean([float(r.get('mesh_readiness_score') or 0.0) for r in dense_metrics]):.3f}")
        print(f"- angular_coverage_score_promedio: {_mean([float(r.get('angular_coverage_score') or 0.0) for r in dense_metrics]):.3f}")
        print(f"- visual_variety_score_promedio: {_mean([float(r.get('visual_variety_score') or 0.0) for r in dense_metrics]):.3f}")
        print(f"- average_feature_points_promedio: {_mean([float(r.get('average_feature_points') or 0.0) for r in dense_metrics]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
