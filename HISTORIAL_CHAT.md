# Historial De Continuidad Del Chat

Ultima actualizacion: 2026-04-19 (prueba real post-optimizacion + export GLB + bateria 42/42)

## Objetivo de este archivo

Este documento funciona como punto de entrada para retomar el trabajo en nuevas sesiones sin perder el hilo tecnico.

## Como usarlo al iniciar una nueva sesion

1. Leer este archivo completo.
2. Revisar el ultimo reporte tecnico detallado en `tmp_caja_eval/registro_pipeline_2026-04-18.md`.
3. Abrir el `execution_report` y `technical_evidence` del ultimo proyecto activo.
4. Continuar desde la seccion "Siguiente paso recomendado".

## Resumen cronologico (2026-04-18 a 2026-04-19)

1. Se aplicaron ajustes conservadores de seleccion y segmentacion en `.env` para mejorar robustez visual sin bloquear el pipeline.
2. Se corrio validacion E2E con dataset `CAJA_PASTILLAS` en OBJ, terminando en `completed_with_fallback` con `primitive_box`.
3. Se genero exportacion GLB para visualizacion y copia descargada local.
4. Se corrio una validacion estricta temporal (sin fallback de caja ni mock), restaurando `.env` al final.
5. Esa validacion estricta confirmo salida COLMAP + `delaunay_mesher_sparse` (sin caja), con estado `completed_with_fallback` por fallback sparse CPU.
6. Se agrego control de textura en fallback de caja para preservar el patron visual limpio durante esta etapa.
7. Se corrio nueva prueba real con `CAJA_PASTILLAS` y se confirmo `primitive_box` en GLB sin textura aplicada.
8. Se reactivo textura en `.env` y se corrio prueba comparativa real para retomar detalle visual base de etiqueta.
9. Se optimizo `box_primitive_fallback.py` removiendo duplicacion interna sin alterar la logica.
10. Se optimizaron `input_image_selector.py`, `input_object_segmenter.py` y `_count_images` en `processing_service.py` sin cambio de comportamiento.
11. Se corrio nueva prueba real con `CAJA_PASTILLAS` (proyecto `f7c048cd7641`) y se dejo export GLB actualizado para visualizacion.
12. Se valido estabilidad con bateria unitaria focalizada `unittest` (42/42 OK).

## Estado tecnico actual

- El flujo completo procesa y exporta modelos.
- La salida estable para evidencia sigue disponible.
- El dataset actual aun presenta bastante blur (advertencias altas), lo cual limita la calidad fotogrametrica pura.
- Con modo estricto ya se comprobo que puede terminar sin fallback de caja, usando malla sparse.
- Para mantener el patron visual del visor (caja limpia), el fallback box ahora puede ejecutarse sin textura.
- Para extraer detalle visual de etiqueta, se puede activar textura en fallback box (estado actual: activado).
- La corrida mas reciente confirmo: seleccion 20/24, segmentacion 20/20 y salida final canónica `primitive_box` con textura aplicada.

## Artefactos clave para retomar

- Registro consolidado principal:
  - `tmp_caja_eval/registro_pipeline_2026-04-18.md`
- Corrida GLB para visualizacion:
  - `data/projects/8361966a3fa9/output/8361966a3fa9_model.glb`
  - `data/exports/8361966a3fa9_model_downloaded.glb`
  - `tmp_caja_eval/runtime_report_8361966a3fa9_glb.json`
- Corrida estricta sin fallback de caja:
  - `data/projects/6e8a476673ba/output/6e8a476673ba_model.obj`
  - `data/projects/6e8a476673ba/output/6e8a476673ba_colmap_metadata.json`
  - `data/projects/6e8a476673ba/output/pipeline/6e8a476673ba_execution_report.json`
  - `data/projects/6e8a476673ba/output/pipeline/6e8a476673ba_technical_evidence.json`
- Corrida patron caja limpia sin textura:
  - `data/projects/24b38591a1f3/output/24b38591a1f3_model.glb`
  - `data/exports/24b38591a1f3_pattern_clean_downloaded.glb`
  - `tmp_caja_eval/runtime_pattern_clean_24b38591a1f3.json`
- Corrida patron caja texturizada (comparativa):
  - `data/projects/41e35b9032ac/output/41e35b9032ac_model.glb`
  - `data/exports/41e35b9032ac_pattern_textured_downloaded.glb`
  - `tmp_caja_eval/runtime_pattern_textured_41e35b9032ac.json`
- Corrida post-optimizacion (actual):
  - `data/projects/f7c048cd7641/output/f7c048cd7641_model.glb`
  - `data/exports/f7c048cd7641_post_optim_visual.glb`
  - `tmp_caja_eval/runtime_post_optim_2026-04-18_glb.json`
  - `data/projects/f7c048cd7641/output/pipeline/f7c048cd7641_execution_report.json`
  - `data/projects/f7c048cd7641/output/pipeline/f7c048cd7641_technical_evidence.json`

## Configuracion actual importante (.env)

- `LOCAL3D_PRIMITIVE_BOX_FALLBACK_ENABLED=true`
- `LOCAL3D_PRIMITIVE_BOX_FALLBACK_TEXTURE_ENABLED=true`
- `LOCAL3D_PRIMITIVE_BOX_FALLBACK_ON_INCOHERENT_OUTPUT=true`
- `LOCAL3D_COLMAP_FALLBACK_TO_MOCK=true`

Nota: para validaciones estrictas se puede desactivar temporalmente lo anterior y restaurar luego.

## Siguiente paso recomendado

Hacer una nueva corrida con mejores capturas (menos blur, mayor estabilidad, objeto ocupando 25%-40% del encuadre) y comparar:

1. Resultado con politica estable (fallback caja activo).
2. Resultado estricto (sin fallback caja), para medir avance real de reconstruccion geometrica.
3. Con el nuevo set de mejor calidad, priorizar vistas frontal/lateral casi ortogonales para recuperar mejor texto del empaque.
