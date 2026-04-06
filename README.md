# Backend Local de Reconstruccion 3D

Backend local en Python + FastAPI para recibir imagenes, ejecutar una tuberia de reconstruccion 3D por etapas y devolver un modelo final en GLB u OBJ.

El proyecto esta pensado para correr en un PC con Windows 10 y ser consumido por una app Flutter en la misma red local.

## Arquitectura General

```text
proyecto_3d (Flutter)
        |
        | HTTP local
        v
PROCESAMIENTO (FastAPI)
        |
        | almacenamiento local + tuberia algoritmica
        v
Modelo 3D final (GLB / OBJ)
```

## Rol de cada parte

### Flutter

La app Flutter es la interfaz de captura y control.

Hace lo siguiente:

- valida conexion con `GET /health`
- crea proyectos
- sube multiples imagenes
- inicia procesamiento
- consulta estado remoto
- descarga el modelo final
- visualiza el resultado

Flutter no ejecuta la reconstruccion 3D central.

### Backend local

El backend hace el trabajo pesado:

- administra proyectos
- guarda imagenes y metadatos en disco
- ejecuta el procesamiento 3D en segundo plano
- expone el estado del trabajo
- devuelve el modelo final

## Estructura principal

```text
.
|-- main.py
|-- config.py
|-- README.md
|-- app/
|   |-- api/
|   |   |-- router.py
|   |   |-- routes/
|   |   |   |-- projects.py
|   |-- algorithms/
|   |   |-- artifacts.py
|   |   |-- image_preprocessor.py
|   |   |-- feature_matcher.py
|   |   |-- pose_estimator.py
|   |   |-- point_cloud_builder.py
|   |   |-- mesh_builder.py
|   |   |-- exporter.py
|   |   |-- reconstruction_pipeline.py
|   |-- models/
|   |   |-- schemas.py
|   |-- services/
|   |   |-- project_service.py
|   |   |-- storage_service.py
|   |   |-- processing_service.py
|   |   |-- engines/
|   |   |   |-- base_engine.py
|   |   |   |-- factory.py
|   |   |   |-- mock_engine.py
|   |   |   |-- colmap_engine.py
|   |-- core/
|   |   |-- dependencies.py
|   |   |-- errors.py
|-- data/
|   |-- projects/
|-- tests/
|   |-- test_reconstruction_pipeline.py
```

## Flujo funcional

1. Flutter pregunta por `GET /health`.
2. Flutter crea un proyecto con `POST /projects`.
3. Flutter sube varias imagenes con `POST /projects/{id}/images`.
4. El backend marca el proyecto como `ready`.
5. Flutter inicia el procesamiento con `POST /projects/{id}/process`.
6. El backend lanza el job en segundo plano.
7. Flutter consulta `GET /projects/{id}/status` hasta que el estado sea `completed`.
8. Flutter descarga `GET /projects/{id}/model`.
9. La app guarda y abre el modelo en su visor 3D.

## API local

- `GET /health`
- `POST /projects`
- `POST /projects/{project_id}/images`
- `POST /projects/{project_id}/process`
- `GET /projects/{project_id}/status`
- `GET /projects/{project_id}/model`

## Flujo algoritmico

La reconstruccion esta organizada por etapas dentro de `app/algorithms/`.

### 1. Validacion y preprocesamiento

Archivo: [app/algorithms/image_preprocessor.py](app/algorithms/image_preprocessor.py)

- valida que existan imagenes
- comprueba extension y tamano
- intenta leer la imagen real con Pillow
- calcula brillo, contraste, nitidez y dimension real
- copia las imagenes a una carpeta de trabajo
- genera un manifiesto de preprocesamiento
- si la lectura real falla, usa un fallback sintetico determinista

### 2. Extraccion y emparejamiento de caracteristicas

Archivo: [app/algorithms/feature_matcher.py](app/algorithms/feature_matcher.py)

- intenta extraer keypoints reales
- usa Pillow para detectar puntos de alto gradiente cuando OpenCV no esta disponible
- si OpenCV estuviera instalado, el modulo ya esta preparado para usar ORB
- construye descriptores simples para matching
- genera emparejamientos entre imagenes consecutivas
- guarda un resumen de features y matches
- si no logra extraer suficientes datos, cae a un fallback sintetico

### 3. Estimacion de poses

Archivo: [app/algorithms/pose_estimator.py](app/algorithms/pose_estimator.py)

- transforma los matches en una distribucion de camaras
- si hay correspondencias suficientes, estima una transformacion aproximada basada en centroides, escala y angulo medio
- si no hay suficientes datos, mantiene la orbita sintetica anterior
- deja un registro en JSON de las poses estimadas

### 4. Construccion de nube de puntos

Archivo: [app/algorithms/point_cloud_builder.py](app/algorithms/point_cloud_builder.py)

- toma las poses y, cuando existen correspondencias reales, genera puntos 3D aproximados
- conserva una nube sintetica estable como fallback
- calcula limites de la nube
- guarda el resultado intermedio

### 5. Generacion de malla

Archivo: [app/algorithms/mesh_builder.py](app/algorithms/mesh_builder.py)

- ordena los puntos
- construye una malla simple y estable
- produce vertices y caras listas para exportacion

### 6. Exportacion

Archivo: [app/algorithms/exporter.py](app/algorithms/exporter.py)

- exporta a OBJ o GLB
- mantiene el contrato actual con Flutter
- produce un archivo final descargable

### 7. Orquestacion

Archivo: [app/algorithms/reconstruction_pipeline.py](app/algorithms/reconstruction_pipeline.py)

- conecta todas las etapas anteriores
- escribe un reporte final del pipeline
- devuelve el archivo final y el resumen de artefactos

## Nivel actual del algoritmo

Hoy el backend no hace reconstruccion 3D fotogrametrica completa, pero ya tiene una tuberia tecnica defendible y parcialmente real.

Cada etapa guarda su propio reporte y el pipeline central agrega `mode: real` o `mode: synthetic` para dejar clara la traza de ejecucion.

### Capas reales

- `image_preprocessor.py` lee imagenes reales con Pillow, mide brillo, contraste, nitidez y dimensiones, y guarda un manifiesto por proyecto.
- `feature_matcher.py` extrae puntos de interes basados en gradiente real de la imagen cuando puede leer el archivo.
- `exporter.py` genera archivos OBJ y GLB validos.
- `processing_service.py` y `storage_service.py` ejecutan y persisten el flujo de extremo a extremo.

### Capas aproximadas

- `pose_estimator.py` estima poses aproximadas a partir de correspondencias reales, usando centroides, escala y rotacion media.
- `point_cloud_builder.py` levanta una nube de puntos aproximada a partir de esas poses y correspondencias.
- `mesh_builder.py` sigue siendo una reconstruccion geometrica simple tipo fan de triangulos.

### Capas sinteticas

- Cuando la lectura real falla, el preprocesador cae a un fallback determinista.
- Cuando no hay suficientes caracteristicas o matches, el matcher y el estimador de poses usan respaldo sintetico.
- `mock_engine.py` sigue siendo el motor por defecto y solo delega en el pipeline.

### Evolucion futura

- Si se integra OpenCV de forma opcional, `feature_matcher.py` ya esta listo para priorizar ORB.
- `colmap_engine.py` sigue como adaptador futuro para una reconstruccion fotogrametrica completa.
- El contrato HTTP no necesita cambiar para hacer esa evolucion.

## Motores de reconstruccion

### Interfaz base

Archivo: [app/services/engines/base_engine.py](app/services/engines/base_engine.py)

Define el contrato comun para cualquier motor de reconstruccion.

### Mock actual

Archivo: [app/services/engines/mock_engine.py](app/services/engines/mock_engine.py)

- sigue siendo el motor funcional por defecto
- simula tiempo de procesamiento
- delega la reconstruccion a `ReconstructionPipeline`
- genera un modelo valido en GLB u OBJ

### COLMAP futuro

Archivo: [app/services/engines/colmap_engine.py](app/services/engines/colmap_engine.py)

- queda como adaptador futuro
- no ejecuta reconstruccion real todavia
- esta preparado para integrar comandos reales de COLMAP

### Factory de motor

Archivo: [app/services/engines/factory.py](app/services/engines/factory.py)

- decide si usar `mock` o `colmap`
- permite cambiar de motor sin tocar las rutas HTTP

## Services

### ProjectService

Archivo: [app/services/project_service.py](app/services/project_service.py)

- crea proyectos
- registra imagenes
- cambia estados
- marca completado o fallido
- valida si el modelo ya puede descargarse

### StorageService

Archivo: [app/services/storage_service.py](app/services/storage_service.py)

- crea carpetas por proyecto
- guarda metadata
- guarda imagenes
- limpia salidas previas
- devuelve la ruta del modelo final

### ProcessingService

Archivo: [app/services/processing_service.py](app/services/processing_service.py)

- coordina el trabajo en segundo plano
- no contiene la logica algoritmica pesada
- delega esa logica al engine elegido

## Estructura de almacenamiento

Cada proyecto se guarda asi:

```text
data/projects/{project_id}/
|-- images/
|-- output/
|   |-- pipeline/
|   |   |-- preprocessing_manifest.json
|   |   |-- features_and_matches.json
|   |   |-- poses.json
|   |   |-- point_cloud.json
|   |   |-- mesh.json
|   |   |-- export.json
|   |   |-- {project_id}_pipeline_report.json
|-- meta.json
```

## Estado y contrato con Flutter

La app Flutter entiende estos estados:

- `created`
- `ready`
- `processing`
- `completed`
- `failed`

Y usa estos campos:

- `model_download_url`
- `output_format`
- `error_message`

## Compatibilidad actual

El flujo remoto ya fue validado end to end con Flutter y sigue siendo funcional.

La app Flutter:

- crea proyecto
- sube multiples imagenes
- inicia procesamiento
- consulta estado
- descarga el modelo final

## Que sigue siendo mock

- La reconstruccion geometrica real aun no usa COLMAP ni otro motor fotogrametrico real.
- El motor por defecto sigue siendo una simulacion bien estructurada.
- La salida GLB/OBJ es valida y util para pruebas, pero no representa todavia una reconstruccion real de vision por computador.

## Futuro de integracion real

La arquitectura ya permite reemplazar la simulacion por una implementacion real usando COLMAP u OpenCV sin rehacer Flutter ni las rutas HTTP.

Solo habria que sustituir la logica interna del engine o del pipeline, manteniendo el contrato actual.

## Ejecucion

```powershell
cd C:\PROYECTO\PROCESAMIENTO
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Pruebas

```powershell
python -m unittest discover -s tests
```

## Nota academica

Con esta estructura, el proyecto puede defenderse asi:

- la app movil es la interfaz de captura y visualizacion
- el backend local ejecuta la reconstruccion 3D
- la reconstruccion esta separada por etapas
- el motor actual es mock, pero la arquitectura esta lista para reemplazarlo por un motor real
