# CAPTURE GUIDE - Dataset Real para Reconstruccion 3D

Esta guia ayuda a capturar un dataset real util para COLMAP y para la validacion academica del proyecto.

## Objetivo de captura

- Tomar entre 20 y 40 fotos por objeto.
- Lograr buen traslape (overlap) entre imagenes consecutivas.
- Evitar fotos borrosas o con exposicion extrema.

## Recomendaciones obligatorias

1. Toma entre `20` y `40` fotos del mismo objeto.
2. Usa buena iluminacion estable (evita zonas oscuras o luces cambiantes).
3. Escoge un objeto con textura visible (patrones, bordes, detalles).
4. Manten un fondo estable (sin movimiento en segundo plano).
5. Evita reflejos fuertes (vidrio, metal pulido, luces directas sobre superficie).
6. No uses zoom digital.
7. Rodea el objeto lentamente, manteniendo distancia similar entre tomas.
8. Conserva overlap entre fotos (cada foto debe compartir contenido con la anterior y la siguiente).
9. Evita fotos borrosas (mano firme, buena velocidad de obturacion).

## Flujo recomendado de captura

1. Limpia el encuadre del objeto.
2. Da una vuelta completa al objeto en pasos pequenos.
3. Repite una segunda vuelta con angulo ligeramente mas alto o mas bajo.
4. Revisa rapidamente nitidez en galeria antes de procesar.

## Errores comunes

- Muy pocas fotos (`< 12`): COLMAP suele fallar por pocas coincidencias.
- Fotos repetidas casi iguales: baja cobertura angular.
- Iluminacion variable entre fotos: baja robustez de matching.
- Fondo en movimiento (personas, TV, ventiladores): ruido en reconstruccion.

## Ubicacion sugerida

Guarda el dataset en:

```text
data/test_datasets/objeto_01/
```

Luego valida con:

```powershell
python scripts/validate_real_dataset.py --input data/test_datasets/objeto_01
```
