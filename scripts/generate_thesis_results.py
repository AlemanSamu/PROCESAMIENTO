from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.thesis_results_service import (
    generate_thesis_results_package,
    load_csv_rows,
    load_json_file,
)
from config import get_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera tablas y capitulo academico reutilizable a partir de reportes de experimentos."
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Ruta al resumen JSON de experimentos. Default: <metrics_root>/reports/processing_experiment_summary.json",
    )
    parser.add_argument(
        "--runs-csv",
        type=Path,
        help="Ruta al CSV de corridas. Default: <metrics_root>/reports/processing_runs_table.csv",
    )
    parser.add_argument(
        "--reasons-csv",
        type=Path,
        help="Ruta al CSV de frecuencias de razones. Default: <metrics_root>/reports/processing_reason_frequencies_table.csv",
    )
    parser.add_argument(
        "--stage-csv",
        type=Path,
        help="Ruta al CSV de tiempos por etapa. Default: <metrics_root>/reports/processing_stage_timings_table.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directorio de salida para tablas y markdown. Default: <metrics_root>/reports/thesis",
    )
    parser.add_argument(
        "--baseline-variant",
        type=str,
        default="baseline",
        help="Nombre de variante baseline para comparacion before vs after.",
    )
    parser.add_argument(
        "--enhanced-variant",
        type=str,
        default="enhanced",
        help="Nombre de variante mejorada para comparacion before vs after.",
    )
    return parser


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"No se encontro {label}: {path}")


def main() -> int:
    settings = get_settings()
    parser = _build_parser()
    args = parser.parse_args()

    metrics_root = Path(getattr(settings, "metrics_evidence_root", Path("data/experiments")))
    reports_root = metrics_root / "reports"

    summary_json = args.summary_json or (reports_root / "processing_experiment_summary.json")
    runs_csv = args.runs_csv or (reports_root / "processing_runs_table.csv")
    reasons_csv = args.reasons_csv or (reports_root / "processing_reason_frequencies_table.csv")
    stage_csv = args.stage_csv or (reports_root / "processing_stage_timings_table.csv")
    output_dir = args.output_dir or (reports_root / "thesis")

    _require_file(summary_json, "summary JSON")
    _require_file(runs_csv, "runs CSV")
    _require_file(reasons_csv, "reasons CSV")
    _require_file(stage_csv, "stage CSV")

    summary_payload = load_json_file(summary_json)
    runs_rows = load_csv_rows(runs_csv)
    reason_rows = load_csv_rows(reasons_csv)
    stage_rows = load_csv_rows(stage_csv)

    outputs = generate_thesis_results_package(
        summary_payload=summary_payload,
        runs_rows=runs_rows,
        reason_rows=reason_rows,
        stage_rows=stage_rows,
        output_dir=output_dir,
        baseline_variant=str(args.baseline_variant).strip() or "baseline",
        enhanced_variant=str(args.enhanced_variant).strip() or "enhanced",
        source_paths={
            "summary_json": summary_json,
            "runs_csv": runs_csv,
            "reasons_csv": reasons_csv,
            "stage_csv": stage_csv,
        },
    )

    print("Entregables generados:")
    for key, path in outputs.items():
        print(f"- {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
