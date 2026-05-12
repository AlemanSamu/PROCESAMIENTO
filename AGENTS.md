# AGENTS.md

## Objetivo del proyecto

Sistema académico de reconstrucción 3D desde imágenes 2D capturadas por una app móvil, optimizado para hardware limitado.

## Restricciones técnicas

* Hardware objetivo: Intel i3-7020U, 8 GB RAM, Intel HD 620
* Evitar soluciones que requieran GPU dedicada
* Priorizar estabilidad, claridad arquitectónica y viabilidad académica

## Reglas de implementación

* Analizar antes de modificar
* Explicar cambios antes de aplicarlos
* Hacer cambios por fases pequeñas
* No romper compatibilidad sin justificar
* Priorizar validación de imágenes, preprocesamiento y robustez del pipeline
* Mantener código legible y documentado
* Crear pruebas cuando se modifique lógica importante

## Qué significa “hecho”

* El cambio compila o corre
* No rompe flujo existente
* Incluye justificación técnica
* Incluye forma de prueba
* Aporta mejora medible o claramente argumentada
