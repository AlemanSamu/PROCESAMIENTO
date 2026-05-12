from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.technical_evidence_service import load_run_records, write_experiment_reports
from config import get_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera reportes comparativos (JSON + CSV) a partir de corridas NDJSON."
    )
    parser.add_argument(
        "--runs-file",
        type=Path,
        help="Ruta del archivo NDJSON de corridas. Default: <metrics_evidence_root>/processing_runs.ndjson",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directorio de salida para reportes. Default: <metrics_evidence_root>/reports",
    )
    parser.add_argument(
        "--before-variant",
        type=str,
        help="Etiqueta de variante baseline para delta before vs after (ej: baseline).",
    )
    parser.add_argument(
        "--after-variant",
        type=str,
        help="Etiqueta de variante mejorada para delta before vs after (ej: enhanced).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Filtra corridas por escenario (good, mixed, bad).",
    )
    return parser


def main() -> int:
    settings = get_settings()
    parser = _build_parser()
    args = parser.parse_args()

    metrics_root = Path(getattr(settings, "metrics_evidence_root", Path("data/experiments")))
    runs_file = args.runs_file or (metrics_root / "processing_runs.ndjson")
    output_dir = args.output_dir or (metrics_root / "reports")

    runs = load_run_records(runs_file)
    if args.scenario:
        scenario_filter = str(args.scenario).strip().lower()
        runs = [
            record
            for record in runs
            if str((record.get("run_info") or {}).get("scenario_label") or "").strip().lower() == scenario_filter
        ]

    if not runs:
        print(f"No se encontraron corridas para procesar en: {runs_file}", file=sys.stderr)
        return 1

    outputs = write_experiment_reports(
        runs=runs,
        output_dir=output_dir,
        before_variant=args.before_variant,
        after_variant=args.after_variant,
    )
    print(f"Corridas procesadas: {len(runs)}")
    print(f"Resumen JSON: {outputs['summary_json']}")
    print(f"Tabla corridas CSV: {outputs['runs_csv']}")
    print(f"Tabla tiempos por etapa CSV: {outputs['stage_timings_csv']}")
    print(f"Tabla frecuencias de razones CSV: {outputs['reason_frequencies_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
