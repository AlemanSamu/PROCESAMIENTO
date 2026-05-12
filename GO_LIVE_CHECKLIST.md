# Go Live Checklist (CPU / Sin CUDA)

## 1) Objetivo
- Asegurar que el backend entregue modelo 3D de forma estable en equipo sin NVIDIA.
- Prioridad operativa: intentar COLMAP real en CPU y caer a fallback si el dataset falla.

## 2) Configuracion Minima
- Confirmar en `.env`:
- `LOCAL3D_PROCESSING_ENGINE=auto`
- `LOCAL3D_COLMAP_BINARY=C:\Tools\COLMAP\COLMAP.bat`
- `LOCAL3D_COLMAP_USE_GPU=false`
- `LOCAL3D_COLMAP_ENABLE_DENSE_STAGES=false`
- `LOCAL3D_COLMAP_FALLBACK_TO_MOCK=true`
- `LOCAL3D_COLMAP_REQUIRE_DENSE_RECONSTRUCTION=false`
- `LOCAL3D_API_KEY=<clave_local_segura>`

## 3) Preflight Tecnico
- Python virtualenv existe: `.venv\Scripts\python.exe`
- Dependencias instaladas: `pip install -r requirements.txt`
- COLMAP responde: `C:\Tools\COLMAP\COLMAP.bat -h`
- Puerto backend libre (default 8000).

## 4) Validacion Antes De Salir
- Levantar backend y validar `GET /health`.
- Confirmar en `health.colmap`:
- `use_gpu=false`
- `enable_dense_stages=false`
- Correr E2E real con dataset patron.
- Aceptacion:
- `status=completed` o `completed_with_fallback`.
- Existe archivo final descargable en `/projects/{id}/model`.

## 5) Dataset Patron (obligatorio)
- Mantener 1 carpeta fija de imagenes buenas para prueba de regresion.
- Recomendado: 20-40 fotos, 60-80% traslape, buena luz, sin blur.
- Ejecutar prueba de regresion al menos semanalmente.

## 6) Operacion Diaria
- Revisar logs de COLMAP cuando falle mapper:
- `data/projects/{id}/output/logs/colmap/*.log`
- Si falla mapper:
- repetir captura con mas traslape/textura.
- Si se requiere "si o si entregar":
- mantener `PROCESSING_ENGINE=auto` y `COLMAP_FALLBACK_TO_MOCK=true`.

## 7) Criterio De Exito Realista
- Sin NVIDIA no hay denso CUDA.
- Exito en este hardware = pipeline estable CPU + entrega consistente de artefacto final.

## 8) Runbook Rapido
- Script recomendado:
- `.\scripts\start_and_validate.ps1 -Port 8000 -OutputFormat glb -MaxImages 24`
- Con dataset fijo:
- `.\scripts\start_and_validate.ps1 -ImagesDir C:\ruta\dataset_patron -Port 8000`
