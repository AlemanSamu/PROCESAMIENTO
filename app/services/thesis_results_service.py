from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


def load_json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_table(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    return path


def build_variant_metrics_rows(runs_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in runs_rows:
        grouped[_value_or(row.get("variant"), "unknown")].append(row)

    results: list[dict[str, Any]] = []
    for variant in sorted(grouped):
        rows = grouped[variant]
        run_count = len(rows)
        completed = sum(1 for row in rows if _normalized(row.get("status")) == "completed")
        failed = sum(1 for row in rows if _normalized(row.get("status")) == "failed")
        blocked = sum(1 for row in rows if _to_bool(row.get("blocked_by_input_deficiency")))

        total_images_loaded = sum(_to_int(row.get("total_images_loaded")) for row in rows)
        accepted_images = sum(_to_int(row.get("accepted_images")) for row in rows)
        rejected_images = sum(_to_int(row.get("rejected_images")) for row in rows)
        selected_images = sum(_to_int(row.get("selected_images")) for row in rows)
        selection_discarded_images = sum(_to_int(row.get("selection_discarded_images")) for row in rows)

        avg_input_reduction_pct = _avg([_to_float(row.get("input_reduction_pct")) for row in rows])
        avg_total_processing_seconds = _avg([_to_float(row.get("total_processing_seconds")) for row in rows])

        results.append(
            {
                "variant": variant,
                "run_count": run_count,
                "completed_count": completed,
                "failed_count": failed,
                "success_rate_pct": _pct(completed, run_count),
                "failure_rate_pct": _pct(failed, run_count),
                "blocked_count": blocked,
                "blocked_rate_pct": _pct(blocked, run_count),
                "total_images_loaded": total_images_loaded,
                "accepted_images": accepted_images,
                "rejected_images": rejected_images,
                "selected_images": selected_images,
                "selection_discarded_images": selection_discarded_images,
                "acceptance_rate_pct": _pct(accepted_images, total_images_loaded),
                "rejection_rate_pct": _pct(rejected_images, total_images_loaded),
                "selected_vs_loaded_pct": _pct(selected_images, total_images_loaded),
                "selected_vs_accepted_pct": _pct(selected_images, accepted_images),
                "avg_input_reduction_pct": _round(avg_input_reduction_pct, 3),
                "avg_total_processing_seconds": _round(avg_total_processing_seconds, 3),
            }
        )
    return results


def build_scenario_variant_rows(runs_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in runs_rows:
        scenario = _value_or(row.get("scenario_label"), "unknown")
        variant = _value_or(row.get("variant"), "unknown")
        grouped[(scenario, variant)].append(row)

    results: list[dict[str, Any]] = []
    for scenario, variant in sorted(grouped):
        rows = grouped[(scenario, variant)]
        run_count = len(rows)
        completed = sum(1 for row in rows if _normalized(row.get("status")) == "completed")
        failed = sum(1 for row in rows if _normalized(row.get("status")) == "failed")
        blocked = sum(1 for row in rows if _to_bool(row.get("blocked_by_input_deficiency")))

        results.append(
            {
                "scenario": scenario,
                "variant": variant,
                "run_count": run_count,
                "success_rate_pct": _pct(completed, run_count),
                "failure_rate_pct": _pct(failed, run_count),
                "blocked_rate_pct": _pct(blocked, run_count),
                "avg_total_processing_seconds": _round(
                    _avg([_to_float(row.get("total_processing_seconds")) for row in rows]),
                    3,
                ),
                "avg_input_reduction_pct": _round(
                    _avg([_to_float(row.get("input_reduction_pct")) for row in rows]),
                    3,
                ),
                "avg_loaded_images": _round(_avg([_to_int(row.get("total_images_loaded")) for row in rows]), 3),
                "avg_accepted_images": _round(_avg([_to_int(row.get("accepted_images")) for row in rows]), 3),
                "avg_rejected_images": _round(_avg([_to_int(row.get("rejected_images")) for row in rows]), 3),
                "avg_selected_images": _round(_avg([_to_int(row.get("selected_images")) for row in rows]), 3),
            }
        )
    return results


def build_success_failure_rows(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overall = summary_payload.get("overall") if isinstance(summary_payload.get("overall"), dict) else {}
    if overall:
        rows.append(_summary_scope_row("overall", "overall", overall))

    by_variant = summary_payload.get("by_variant") if isinstance(summary_payload.get("by_variant"), dict) else {}
    for variant, values in sorted(by_variant.items()):
        if isinstance(values, dict):
            rows.append(_summary_scope_row("variant", str(variant), values))

    by_scenario = summary_payload.get("by_scenario") if isinstance(summary_payload.get("by_scenario"), dict) else {}
    for scenario, values in sorted(by_scenario.items()):
        if isinstance(values, dict):
            rows.append(_summary_scope_row("scenario", str(scenario), values))

    return rows


def build_reduction_time_rows(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_variant = summary_payload.get("by_variant") if isinstance(summary_payload.get("by_variant"), dict) else {}
    by_scenario = summary_payload.get("by_scenario") if isinstance(summary_payload.get("by_scenario"), dict) else {}

    for variant, values in sorted(by_variant.items()):
        if not isinstance(values, dict):
            continue
        rows.append(
            {
                "scope_type": "variant",
                "scope": str(variant),
                "avg_input_reduction_pct": _round(_to_float(values.get("avg_input_reduction_pct")), 3),
                "avg_total_processing_seconds": _round(_to_float(values.get("avg_total_processing_seconds")), 3),
                "median_total_processing_seconds": _round(_to_float(values.get("median_total_processing_seconds")), 3),
            }
        )

    for scenario, values in sorted(by_scenario.items()):
        if not isinstance(values, dict):
            continue
        rows.append(
            {
                "scope_type": "scenario",
                "scope": str(scenario),
                "avg_input_reduction_pct": _round(_to_float(values.get("avg_input_reduction_pct")), 3),
                "avg_total_processing_seconds": _round(_to_float(values.get("avg_total_processing_seconds")), 3),
                "median_total_processing_seconds": _round(_to_float(values.get("median_total_processing_seconds")), 3),
            }
        )

    return rows


def build_before_after_rows(
    summary_payload: dict[str, Any],
    *,
    baseline_variant: str,
    enhanced_variant: str,
) -> list[dict[str, Any]]:
    by_variant = summary_payload.get("by_variant") if isinstance(summary_payload.get("by_variant"), dict) else {}
    baseline = by_variant.get(baseline_variant) if isinstance(by_variant.get(baseline_variant), dict) else {}
    enhanced = by_variant.get(enhanced_variant) if isinstance(by_variant.get(enhanced_variant), dict) else {}

    if not baseline or not enhanced:
        return []

    metrics = [
        ("run_count", "corridas", 0),
        ("completed_count", "corridas", 0),
        ("failed_count", "corridas", 0),
        ("success_rate", "%", 4),
        ("failure_rate", "%", 4),
        ("input_blocked_rate", "%", 4),
        ("fallback_rate", "%", 4),
        ("avg_input_reduction_pct", "%", 3),
        ("avg_total_processing_seconds", "segundos", 3),
        ("median_total_processing_seconds", "segundos", 3),
    ]

    rows: list[dict[str, Any]] = []
    for key, unit, ndigits in metrics:
        baseline_value = baseline.get(key)
        enhanced_value = enhanced.get(key)

        if key.endswith("_rate") and isinstance(baseline_value, (int, float)) and isinstance(enhanced_value, (int, float)):
            baseline_value = round(float(baseline_value) * 100.0, 3)
            enhanced_value = round(float(enhanced_value) * 100.0, 3)
        elif key == "success_rate" or key == "failure_rate":
            if isinstance(baseline_value, (int, float)) and isinstance(enhanced_value, (int, float)):
                baseline_value = round(float(baseline_value) * 100.0, 3)
                enhanced_value = round(float(enhanced_value) * 100.0, 3)

        delta = None
        if isinstance(baseline_value, (int, float)) and isinstance(enhanced_value, (int, float)):
            delta = round(float(enhanced_value) - float(baseline_value), ndigits)

        rows.append(
            {
                "metric": key,
                "unit": unit,
                "baseline_value": baseline_value,
                "enhanced_value": enhanced_value,
                "delta_enhanced_minus_baseline": delta,
            }
        )

    return rows


def build_reason_rows(reason_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in reason_rows:
        variant = _value_or(row.get("variant"), "unknown")
        reason_type = _value_or(row.get("reason_type"), "unknown")
        reason = _value_or(row.get("reason"), "unknown")
        grouped[(variant, reason_type, reason)] += _to_int(row.get("count"))

    results: list[dict[str, Any]] = []
    for (variant, reason_type, reason), count in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], -item[1], item[0][2]),
    ):
        results.append(
            {
                "variant": variant,
                "reason_type": reason_type,
                "reason": reason,
                "total_count": count,
            }
        )
    return results


def build_stage_timing_rows(stage_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in stage_rows:
        variant = _value_or(row.get("variant"), "unknown")
        stage = _value_or(row.get("stage"), "unknown")
        duration = _to_float(row.get("duration_seconds"))
        if duration is None:
            continue
        grouped[(variant, stage)].append(duration)

    results: list[dict[str, Any]] = []
    for (variant, stage), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        results.append(
            {
                "variant": variant,
                "stage": stage,
                "sample_count": len(values),
                "avg_duration_seconds": _round(_avg(values), 3),
                "min_duration_seconds": _round(min(values), 3),
                "max_duration_seconds": _round(max(values), 3),
            }
        )
    return results


def generate_thesis_results_package(
    *,
    summary_payload: dict[str, Any],
    runs_rows: list[dict[str, str]],
    reason_rows: list[dict[str, str]],
    stage_rows: list[dict[str, str]],
    output_dir: Path,
    baseline_variant: str,
    enhanced_variant: str,
    source_paths: dict[str, Path],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_rows = build_variant_metrics_rows(runs_rows)
    scenario_rows = build_scenario_variant_rows(runs_rows)
    success_failure_rows = build_success_failure_rows(summary_payload)
    reduction_time_rows = build_reduction_time_rows(summary_payload)
    before_after_rows = build_before_after_rows(
        summary_payload,
        baseline_variant=baseline_variant,
        enhanced_variant=enhanced_variant,
    )
    reason_table_rows = build_reason_rows(reason_rows)
    stage_timing_rows = build_stage_timing_rows(stage_rows)

    csv_outputs = {
        "table_baseline_vs_enhanced": write_csv_table(
            output_dir / "thesis_table_baseline_vs_enhanced.csv",
            before_after_rows,
            ["metric", "unit", "baseline_value", "enhanced_value", "delta_enhanced_minus_baseline"],
        ),
        "table_scenario_metrics": write_csv_table(
            output_dir / "thesis_table_scenario_metrics.csv",
            scenario_rows,
            [
                "scenario",
                "variant",
                "run_count",
                "success_rate_pct",
                "failure_rate_pct",
                "blocked_rate_pct",
                "avg_total_processing_seconds",
                "avg_input_reduction_pct",
                "avg_loaded_images",
                "avg_accepted_images",
                "avg_rejected_images",
                "avg_selected_images",
            ],
        ),
        "table_success_failure": write_csv_table(
            output_dir / "thesis_table_success_failure_rates.csv",
            success_failure_rows,
            [
                "scope_type",
                "scope",
                "run_count",
                "completed_count",
                "failed_count",
                "success_rate_pct",
                "failure_rate_pct",
                "input_blocked_rate_pct",
                "fallback_rate_pct",
                "top_failed_stage",
                "top_reason_code",
            ],
        ),
        "table_reduction_and_time": write_csv_table(
            output_dir / "thesis_table_reduction_and_time.csv",
            reduction_time_rows,
            [
                "scope_type",
                "scope",
                "avg_input_reduction_pct",
                "avg_total_processing_seconds",
                "median_total_processing_seconds",
            ],
        ),
        "table_variant_metrics": write_csv_table(
            output_dir / "thesis_table_variant_metrics.csv",
            variant_rows,
            [
                "variant",
                "run_count",
                "completed_count",
                "failed_count",
                "success_rate_pct",
                "failure_rate_pct",
                "blocked_count",
                "blocked_rate_pct",
                "total_images_loaded",
                "accepted_images",
                "rejected_images",
                "selected_images",
                "selection_discarded_images",
                "acceptance_rate_pct",
                "rejection_rate_pct",
                "selected_vs_loaded_pct",
                "selected_vs_accepted_pct",
                "avg_input_reduction_pct",
                "avg_total_processing_seconds",
            ],
        ),
        "table_top_reasons": write_csv_table(
            output_dir / "thesis_table_top_reasons.csv",
            reason_table_rows,
            ["variant", "reason_type", "reason", "total_count"],
        ),
        "table_stage_timings": write_csv_table(
            output_dir / "thesis_table_stage_timings_summary.csv",
            stage_timing_rows,
            [
                "variant",
                "stage",
                "sample_count",
                "avg_duration_seconds",
                "min_duration_seconds",
                "max_duration_seconds",
            ],
        ),
    }

    markdown_path = output_dir / "thesis_results_chapter.md"
    markdown_path.write_text(
        _build_markdown_content(
            summary_payload=summary_payload,
            variant_rows=variant_rows,
            scenario_rows=scenario_rows,
            success_failure_rows=success_failure_rows,
            reduction_time_rows=reduction_time_rows,
            before_after_rows=before_after_rows,
            reason_rows=reason_table_rows,
            stage_timing_rows=stage_timing_rows,
            source_paths=source_paths,
            baseline_variant=baseline_variant,
            enhanced_variant=enhanced_variant,
        ),
        encoding="utf-8",
    )

    outputs = dict(csv_outputs)
    outputs["chapter_markdown"] = markdown_path
    return outputs


def _summary_scope_row(scope_type: str, scope: str, values: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope_type": scope_type,
        "scope": scope,
        "run_count": _to_int(values.get("run_count")),
        "completed_count": _to_int(values.get("completed_count")),
        "failed_count": _to_int(values.get("failed_count")),
        "success_rate_pct": _round(_to_float(values.get("success_rate")) * 100.0, 3),
        "failure_rate_pct": _round(_to_float(values.get("failure_rate")) * 100.0, 3),
        "input_blocked_rate_pct": _round(_to_float(values.get("input_blocked_rate")) * 100.0, 3),
        "fallback_rate_pct": _round(_to_float(values.get("fallback_rate")) * 100.0, 3),
        "top_failed_stage": values.get("top_failed_stage"),
        "top_reason_code": values.get("top_reason_code"),
    }


def _build_markdown_content(
    *,
    summary_payload: dict[str, Any],
    variant_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    success_failure_rows: list[dict[str, Any]],
    reduction_time_rows: list[dict[str, Any]],
    before_after_rows: list[dict[str, Any]],
    reason_rows: list[dict[str, Any]],
    stage_timing_rows: list[dict[str, Any]],
    source_paths: dict[str, Path],
    baseline_variant: str,
    enhanced_variant: str,
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    overall = summary_payload.get("overall") if isinstance(summary_payload.get("overall"), dict) else {}

    run_count = _to_int(overall.get("run_count"))
    completed_count = _to_int(overall.get("completed_count"))
    failed_count = _to_int(overall.get("failed_count"))

    baseline_row = next((row for row in variant_rows if row.get("variant") == baseline_variant), None)
    enhanced_row = next((row for row in variant_rows if row.get("variant") == enhanced_variant), None)

    key_reason_lines = _top_reason_lines(reason_rows, variant=enhanced_variant, top_n=5)

    stage_timing_note = (
        "No se registraron tiempos por etapa en este lote de evidencia, por lo que el analisis temporal por etapa queda pendiente de corridas con trazas completas."
        if not stage_timing_rows
        else "Se registraron tiempos por etapa y se resumen en la tabla correspondiente."
    )

    lines: list[str] = []
    lines.append("# Capitulo de pruebas y resultados")
    lines.append("")
    lines.append(f"Fecha de generacion: {generated_at}")
    lines.append("")
    lines.append("## 1. Introduccion de pruebas")
    lines.append("")
    lines.append(
        "Esta seccion consolida evidencia tecnica del pipeline de reconstruccion 3D en hardware limitado, "
        "a partir de los reportes estructurados generados automaticamente por el sistema."
    )
    lines.append(
        f"El conjunto analizado incluye {run_count} corridas, de las cuales {completed_count} finalizaron correctamente y {failed_count} finalizaron con fallo controlado."
    )
    lines.append("")
    lines.append("Fuentes de datos utilizadas:")
    lines.append(f"- Summary JSON: `{source_paths['summary_json']}`")
    lines.append(f"- Runs CSV: `{source_paths['runs_csv']}`")
    lines.append(f"- Reasons CSV: `{source_paths['reasons_csv']}`")
    lines.append(f"- Stage timings CSV: `{source_paths['stage_csv']}`")
    lines.append("")

    lines.append("## 2. Metodologia experimental")
    lines.append("")
    lines.append(
        "Se compararon dos variantes del pipeline: `baseline` (sin filtros estrictos de entrada) y `enhanced` "
        "(con validacion previa, seleccion automatica y control robusto de estados/errores)."
    )
    lines.append(
        "Las corridas se organizaron en tres escenarios (`good`, `mixed`, `bad`) para medir comportamiento bajo calidad de entrada heterogenea."
    )
    lines.append(
        "Las metricas analizadas abarcan calidad del lote de entrada, reduccion de redundancia, tiempo de ejecucion y trazabilidad de fallos."
    )
    lines.append("")

    lines.append("## 3. Resultados")
    lines.append("")
    lines.append("### 3.1 Comparacion baseline vs enhanced")
    lines.append("")
    lines.extend(_markdown_table(
        rows=before_after_rows,
        columns=["metric", "unit", "baseline_value", "enhanced_value", "delta_enhanced_minus_baseline"],
        headers=["Metrica", "Unidad", "Baseline", "Enhanced", "Delta (enhanced-baseline)"],
    ))
    lines.append("")

    lines.append("### 3.2 Metricas por escenario")
    lines.append("")
    lines.extend(_markdown_table(
        rows=scenario_rows,
        columns=[
            "scenario",
            "variant",
            "run_count",
            "success_rate_pct",
            "failure_rate_pct",
            "blocked_rate_pct",
            "avg_input_reduction_pct",
            "avg_total_processing_seconds",
            "avg_selected_images",
        ],
        headers=[
            "Escenario",
            "Variante",
            "Corridas",
            "Exito (%)",
            "Fallo (%)",
            "Bloqueo por entrada (%)",
            "Reduccion media entrada (%)",
            "Tiempo medio (s)",
            "Imagenes seleccionadas (prom)",
        ],
    ))
    lines.append("")

    lines.append("### 3.3 Tasas de exito y fallo")
    lines.append("")
    lines.extend(_markdown_table(
        rows=success_failure_rows,
        columns=[
            "scope_type",
            "scope",
            "run_count",
            "completed_count",
            "failed_count",
            "success_rate_pct",
            "failure_rate_pct",
            "input_blocked_rate_pct",
            "top_failed_stage",
        ],
        headers=[
            "Tipo",
            "Ambito",
            "Corridas",
            "Completadas",
            "Fallidas",
            "Exito (%)",
            "Fallo (%)",
            "Bloqueo por entrada (%)",
            "Etapa fallida principal",
        ],
    ))
    lines.append("")

    lines.append("### 3.4 Reduccion de imagenes y tiempos")
    lines.append("")
    lines.extend(_markdown_table(
        rows=reduction_time_rows,
        columns=[
            "scope_type",
            "scope",
            "avg_input_reduction_pct",
            "avg_total_processing_seconds",
            "median_total_processing_seconds",
        ],
        headers=[
            "Tipo",
            "Ambito",
            "Reduccion media (%)",
            "Tiempo medio (s)",
            "Tiempo mediano (s)",
        ],
    ))
    lines.append("")

    lines.append("## 4. Analisis tecnico")
    lines.append("")
    if baseline_row and enhanced_row:
        lines.append(
            "Los resultados muestran un trade-off explicito entre calidad de entrada y tasa de procesamiento aceptado. "
            f"La variante `enhanced` reduce el lote de entrada en promedio {enhanced_row['avg_input_reduction_pct']}%, "
            f"frente a {baseline_row['avg_input_reduction_pct']}% en `baseline`, y disminuye el tiempo medio por corrida "
            f"de {baseline_row['avg_total_processing_seconds']} s a {enhanced_row['avg_total_processing_seconds']} s."
        )
        lines.append(
            f"Sin embargo, la tasa de exito cae de {baseline_row['success_rate_pct']}% (`baseline`) a {enhanced_row['success_rate_pct']}% (`enhanced`), "
            "porque el sistema bloquea de forma preventiva lotes deficientes en vez de continuar con salidas potencialmente no confiables."
        )

    lines.append(
        "Impacto de la validacion: se incrementa el rechazo de imagenes con problemas de nitidez, exposicion, resolucion o redundancia, "
        "evitando que el motor de reconstruccion procese informacion degradada."
    )
    lines.append(
        "Impacto de la seleccion automatica: se reduce la redundancia y se conserva un subconjunto util, lo cual disminuye carga de computo "
        "en entorno de CPU limitada."
    )
    lines.append(
        "Impacto de la robustez del pipeline: los fallos quedan trazados por etapa y razon (`failed_stage`, `reason_code`), "
        "el bloqueo por entrada es explicito y no se producen estados ambiguos de exito parcial."
    )
    lines.append("")

    lines.append("Principales razones observadas en la variante enhanced:")
    for line in key_reason_lines:
        lines.append(f"- {line}")
    lines.append("")

    lines.append("## 5. Discusion tecnica y limitaciones")
    lines.append("")
    lines.append(
        "El comportamiento observado es coherente con una estrategia conservadora orientada a calidad: "
        "se aceptan menos corridas, pero se mejora la higiene de los datos de entrada."
    )
    lines.append(stage_timing_note)
    lines.append(
        "El conjunto actual de experimentos corresponde a corridas con motor mock para comparar control de entrada y trazabilidad; "
        "la validez externa sobre calidad geometrica final debe reforzarse con corridas COLMAP completas y evaluacion visual/geomtrica adicional."
    )
    lines.append(
        "Las reglas de validacion y seleccion pueden producir falsos positivos en escenas complejas o con baja textura, "
        "por lo que se recomienda calibracion de umbrales por dominio de captura."
    )
    lines.append("")

    lines.append("## 6. Propuesta de figuras y como construirlas")
    lines.append("")
    lines.append("1. Grafica de barras de tasa de exito/fallo por variante.")
    lines.append("   Fuente: `thesis_table_success_failure_rates.csv` filtrando `scope_type=variant`.")
    lines.append("2. Grafica de barras agrupadas de reduccion de imagenes por escenario y variante.")
    lines.append("   Fuente: `thesis_table_scenario_metrics.csv`, columnas `scenario`, `variant`, `avg_input_reduction_pct`.")
    lines.append("3. Grafica de tiempos medios por escenario y variante.")
    lines.append("   Fuente: `thesis_table_scenario_metrics.csv`, columna `avg_total_processing_seconds`.")
    lines.append("4. Pareto de razones de rechazo en la variante mejorada.")
    lines.append("   Fuente: `thesis_table_top_reasons.csv` filtrando `variant=enhanced` y `reason_type=validation_rejected_reason_counts`.")
    lines.append("5. Heatmap de estado final por escenario y variante.")
    lines.append("   Fuente: `processing_runs_table.csv` usando `scenario_label`, `variant`, `status`.")
    lines.append("")

    lines.append("## 7. Conclusiones")
    lines.append("")
    lines.append(
        "Las mejoras implementadas (validacion, seleccion y robustez) fortalecen la estabilidad operativa y la trazabilidad del pipeline, "
        "a costa de una mayor tasa de bloqueo preventivo en lotes deficientes."
    )
    lines.append(
        "En el contexto de hardware limitado, esta decision es tecnicamente justificable porque reduce procesamiento inutil "
        "y evita reconstrucciones con insumos de baja calidad."
    )
    lines.append("")

    lines.append("## 8. Puntos para defensa ante jurado")
    lines.append("")
    lines.append("- Mostrar el trade-off como decision de ingenieria: menos corridas aceptadas, mayor control de calidad de entrada.")
    lines.append("- Evidenciar que los fallos no son silenciosos: se reporta etapa fallida, causa y condicion de reintento.")
    lines.append("- Justificar la reduccion de tiempo con menor cardinalidad de entrada, clave para CPU sin GPU dedicada.")
    lines.append("- Destacar reproducibilidad: cada corrida deja JSON/CSV trazable y tablas reutilizables para auditoria academica.")
    lines.append("- Proponer trabajo futuro: recalibrar umbrales y repetir con COLMAP completo para medir calidad geometrica final.")
    lines.append("")

    return "\n".join(lines)


def _top_reason_lines(reason_rows: list[dict[str, Any]], *, variant: str, top_n: int) -> list[str]:
    filtered = [row for row in reason_rows if row.get("variant") == variant]
    filtered.sort(key=lambda row: (-_to_int(row.get("total_count")), str(row.get("reason_type")), str(row.get("reason"))))
    return [
        f"{row.get('reason_type')}: {row.get('reason')} -> {row.get('total_count')}"
        for row in filtered[:top_n]
    ]


def _markdown_table(*, rows: list[dict[str, Any]], columns: list[str], headers: list[str]) -> list[str]:
    if not rows:
        return ["No hay datos disponibles para esta tabla."]

    table_lines = []
    table_lines.append("| " + " | ".join(headers) + " |")
    table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        values = [str(row.get(column, "")) for column in columns]
        table_lines.append("| " + " | ".join(values) + " |")

    return table_lines


def _value_or(value: Any, fallback: str) -> str:
    text = _normalized(value)
    return text if text else fallback


def _normalized(value: Any) -> str:
    return str(value or "").strip().lower()


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _normalized(value)
    return text in {"1", "true", "yes", "si"}


def _avg(values: list[float]) -> float:
    filtered = [float(value) for value in values]
    return mean(filtered) if filtered else 0.0


def _pct(part: float, whole: float) -> float:
    if whole <= 0:
        return 0.0
    return round((float(part) / float(whole)) * 100.0, 3)


def _round(value: float, ndigits: int = 3) -> float:
    return round(float(value), ndigits)
