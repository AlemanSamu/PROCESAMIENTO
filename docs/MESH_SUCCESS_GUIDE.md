# MESH SUCCESS GUIDE

Esta guia resume como capturar datasets para aumentar la probabilidad de obtener:

- `success_sparse_only`
- `success_approx_surface`
- `success_real`

## 1) Que significa cada resultado

- `success_sparse_only`: SfM real, pero solo nube de puntos util.
- `success_approx_surface`: no hubo dense real usable, pero se genero superficie aproximada desde sparse.
- `success_real`: malla densa real de COLMAP usable.

## 2) Checklist practico de captura

- Tomar entre 45 y 60 fotos.
- Usar 3 alturas: baja, media y alta.
- Cubrir 360 grados alrededor del objeto.
- Mantener overlap de 70-80% entre fotos consecutivas.
- Elegir objeto con textura (evitar superficies lisas o brillantes).
- Usar fondo con textura moderada para mejorar matching.
- Iluminacion uniforme, sin sombras duras.
- Evitar zoom digital.
- Evitar rafagas casi iguales (reducir duplicados).
- Mantener enfoque consistente (sin motion blur).

## 3) Objetivos minimos por nivel

- Para `success_sparse_only`:
  - >= 20 imagenes validas.
  - nitidez y features suficientes para registrar camaras.
- Para `success_approx_surface`:
  - >= 30 imagenes validas recomendadas.
  - buena variedad angular.
  - buen score de readiness (ideal >= 0.45).
- Para `success_real`:
  - >= 45 imagenes validas.
  - alta textura y cobertura angular.
  - readiness alto (ideal >= 0.65).

## 4) Interpretacion academica

- Sparse no es fallo del sistema: es evidencia real de SfM.
- Superficie aproximada desde sparse es una salida defendible cuando dense falla.
- Dense real es el objetivo maximo, pero depende mucho de calidad de captura.

## 5) Flujo recomendado

1. Validar dataset con `scripts/validate_real_dataset.py`.
2. Revisar `mesh_readiness_score` y recomendaciones de malla.
3. Aplicar mejoras con `scripts/suggest_capture_improvements.py`.
4. Recapturar y repetir validacion antes de lanzar pipeline completo.
