# EVIDENCE CHECKLIST - Sustentacion

Usa esta lista para reunir evidencia tecnica verificable del flujo end-to-end.

## 1) Entorno y hardware

- [ ] Captura de `nvidia-smi -L`
- [ ] Salida de `python scripts/check_colmap_setup.py`
- [ ] Captura de `GET /health` del backend (engine, profile, colmap, gpu)

## 2) Flujo Flutter conectado

- [ ] Pantalla Flutter conectada al backend local
- [ ] Pantalla de subida de imagenes
- [ ] Pantalla de procesamiento en curso (estado/etapa/progreso)
- [ ] Pantalla de resultado final
- [ ] Pantalla de "Detalles tecnicos"
- [ ] Visor GLB abierto (si aplica)

## 3) Artefactos JSON del backend

- [ ] `quality_report.json`
- [ ] `colmap_report.json`
- [ ] `fallback_report.json` (solo si aplica fallback)
- [ ] `preprocessing_manifest.json`

## 4) Reportes comparativos por perfil

- [ ] `profile_comparison.json`
- [ ] `profile_comparison.csv`

## 5) Modelo y logs

- [ ] Modelo final (`.glb` o `.obj`)
- [ ] Logs COLMAP (`output/logs/colmap/*.stdout.log` y `*.stderr.log`)

## 6) Comandos recomendados para evidencia

```powershell
python scripts/check_colmap_setup.py
python scripts/validate_real_dataset.py --input data/test_datasets/objeto_01
python scripts/test_pipeline.py --input data/test_datasets/objeto_01 --output-format glb
python scripts/run_experiment.py --input data/test_datasets/objeto_01 --profiles conservative balanced quality --output-format glb
python scripts/generate_defense_package.py --input data/test_datasets/objeto_01
```

## 7) Resultado esperado para sustentacion

- [ ] Existe carpeta `defense_package/` con resumen y anexos tecnicos.
- [ ] Se puede explicar claramente si el caso fue `success_real`, `success_sparse_only`, `fallback_completed` o `failed`.
- [ ] Se puede justificar la recomendacion de siguiente accion con base en reportes.
