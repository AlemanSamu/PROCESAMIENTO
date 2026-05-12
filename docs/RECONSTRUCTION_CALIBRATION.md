# RECONSTRUCTION CALIBRATION

Esta capa **no entrena COLMAP**.  
Entrena/calibra decisiones del pipeline usando historico real del proyecto.

## Que se calibra

- validacion del dataset
- prediccion de probabilidad de exito
- recomendacion de perfil (`balanced` / `quality`)
- decision de si conviene ejecutar COLMAP en ese dataset

## Historial de corridas

Se guarda automaticamente en:

- `data/experiments/reconstruction_history.ndjson`

Cada linea incluye metricas de dataset + resultado final:

- `mesh_readiness_score`
- `angular_coverage_score`
- `visual_variety_score`
- `average_feature_points`
- `quality_classification`
- `geometry_source`
- `final_success_level`

## Entrenamiento/calibracion

Comando:

```powershell
.\.venv\Scripts\python.exe scripts\train_reconstruction_calibrator.py
```

Salida:

- `data/experiments/models/reconstruction_calibrator.json`

Estrategia:

- pocos datos: umbrales estadisticos (heuristic thresholds)
- datos suficientes + `scikit-learn` disponible: regresion logistica

## Prediccion previa a reconstruccion

```powershell
.\.venv\Scripts\python.exe scripts\predict_reconstruction_success.py --dataset "C:\ruta\imagenes"
```

Entrega:

- probabilidad `sparse`
- probabilidad `approx_surface`
- probabilidad `dense_real`
- `recommended_profile`
- `should_run_colmap`
- acciones sugeridas de captura

## Integracion con validate_real_dataset

`scripts/validate_real_dataset.py` ahora agrega:

- `predicted_success_level`
- `predicted_success_probabilities`
- `recommended_profile`
- `should_run_colmap`

## Cuantas corridas se recomiendan

- minimo util: 20 corridas
- recomendado para calibracion estable: 40 a 80 corridas
- ideal si se separa por tipo de objeto/dataset

## Interpretacion

- Alta probabilidad `dense_real`: dataset fuerte para malla densa.
- Alta `approx_surface`: buena opcion para superficie aproximada si dense falla.
- Alta `sparse`: conviene mejorar captura antes de corrida final.
