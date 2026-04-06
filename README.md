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
- `mock_engine.py` sigue disponible como respaldo controlado para pruebas o modo `auto`, pero la validacion real se hace con COLMAP.

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

- sigue disponible para pruebas internas o como respaldo controlado en modo `auto`
- simula tiempo de procesamiento
- delega la reconstruccion a `ReconstructionPipeline`
- genera un modelo valido en GLB u OBJ

### COLMAP real

Archivo: [app/services/engines/colmap_engine.py](app/services/engines/colmap_engine.py)

- detecta el binario configurado o el que exista en `PATH`
- crea un workspace real por proyecto dentro de `output/workspace`
- ejecuta `feature_extractor`, `exhaustive_matcher` y `mapper` por linea de comandos
- registra tiempos por etapa y guarda `stdout/stderr` reales por comando dentro de `output/logs/colmap`
- valida de forma estricta que exista `workspace/sparse/0` con archivos de reconstruccion
- exporta el modelo sparse a TXT, genera `PLY` cuando COLMAP lo permite y siempre deja un `OBJ` usable en `output`
- genera un artefacto descargable en `OBJ` o `GLB` a partir de la nube sparse
- guarda metricas reales de reconstruccion y no cae automaticamente a mock cuando el modo configurado es `colmap`

### Factory de motor

Archivo: [app/services/engines/factory.py](app/services/engines/factory.py)

- decide si usar `mock` o `colmap`
- en modo `colmap`, mantiene ese motor incluso si el binario falla, para que la validacion real falle de forma visible
- en modo `auto`, puede usar `mock` solo cuando COLMAP no esta disponible o cuando el fallback fue habilitado explicitamente
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

## Integracion COLMAP

El backend ya puede usar COLMAP como motor real de reconstruccion sparse sin cambiar las rutas HTTP ni el flujo de Flutter.

### Flujo del motor COLMAP

1. `ProcessingService` usa COLMAP cuando `processing_engine=colmap`; en `auto` puede volver a `mock` solo cuando COLMAP no esta disponible.
2. Se limpia `output/` del proyecto antes de cada corrida.
3. `ColmapReconstructionEngine` crea:

```text
data/projects/{project_id}/output/
|-- workspace/
|   |-- database.db
|   |-- sparse/
|   |-- dense/
|-- logs/
|   |-- colmap/
|   |   |-- feature_extractor.stdout.log
|   |   |-- feature_extractor.stderr.log
|   |   |-- exhaustive_matcher.stdout.log
|   |   |-- exhaustive_matcher.stderr.log
|   |   |-- mapper.stdout.log
|   |   |-- mapper.stderr.log
|-- colmap_sparse_txt/
|   |-- cameras.txt
|   |-- images.txt
|   |-- points3D.txt
|-- {project_id}_sparse.ply
|-- {project_id}_model.obj | {project_id}_model.glb
|-- {project_id}_colmap_metadata.json
```

4. Se ejecutan estos comandos:

```powershell
colmap feature_extractor --database_path <workspace/database.db> --image_path <images> --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_RADIAL --SiftExtraction.use_gpu 0
colmap exhaustive_matcher --database_path <workspace/database.db> --SiftMatching.use_gpu 0
colmap mapper --database_path <workspace/database.db> --image_path <images> --output_path <workspace/sparse>
colmap model_converter --input_path <workspace/sparse/0> --output_path <output/colmap_sparse_txt> --output_type TXT
colmap model_converter --input_path <workspace/sparse/0> --output_path <output/{project_id}_sparse.ply> --output_type PLY
```

5. El backend parsea `points3D.txt`, genera el artefacto final descargable y guarda metadata adicional en `meta.json` y en `{project_id}_colmap_metadata.json`.

### Variables utiles

Se pueden configurar por `.env` con prefijo `LOCAL3D_`:

- `LOCAL3D_PROCESSING_ENGINE=auto|mock|colmap`
- `LOCAL3D_COLMAP_BINARY=C:\\ruta\\a\\COLMAP.bat` o `C:\\ruta\\a\\colmap.exe`
- `LOCAL3D_COLMAP_TIMEOUT_SECONDS=1800`
- `LOCAL3D_COLMAP_USE_GPU=false`
- `LOCAL3D_COLMAP_CAMERA_MODEL=SIMPLE_RADIAL`
- `LOCAL3D_COLMAP_SINGLE_CAMERA=true`
- `LOCAL3D_COLMAP_FALLBACK_TO_MOCK=false`

### Windows 10

- COLMAP no se instala desde este backend; debe estar instalado previamente en Windows 10.
- Si el ejecutable no esta en `PATH`, configura `LOCAL3D_COLMAP_BINARY` con la ruta absoluta.
- Si tu instalacion de COLMAP no tiene soporte CUDA, deja `LOCAL3D_COLMAP_USE_GPU=false`.
- El `GLB` que genera este backend desde COLMAP representa una nube de puntos sparse en modo `POINTS`, no una malla densa.

## Que sigue siendo mock

- La reconstruccion geometrica real aun no usa COLMAP ni otro motor fotogrametrico real.
- El motor por defecto sigue siendo una simulacion bien estructurada.
- La salida GLB/OBJ es valida y util para pruebas, pero no representa todavia una reconstruccion real de vision por computador.

## Limitaciones actuales

- La reconstruccion real actual llega hasta `sparse reconstruction`.
- No hay pipeline denso ni meshing real de COLMAP todavia.
- El artefacto `OBJ` exportado desde COLMAP es una nube de puntos, no una malla cerrada.
- El `GLB` exportado desde COLMAP es valido, pero algunos visores renderizan mejor mallas que nubes de puntos.
- La seleccion del mejor modelo sparse se hace por presencia de archivos y tamano aproximado; no hay ranking fotogrametrico avanzado.
- Si COLMAP produce un modelo sparse vacio o no crea `sparse/0`, el proceso queda en `failed` con error explicito para no ocultar la falla real.

## Futuro de integracion real

La arquitectura ya permite seguir evolucionando la reconstruccion real usando COLMAP sin rehacer Flutter ni las rutas HTTP.

Los siguientes pasos naturales son:

- agregar `image_undistorter` y `patch_match_stereo` para una fase densa
- convertir el resultado real a una malla mas util
- incorporar una conversion posterior a `GLB` de malla cuando exista ese artefacto

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

## Validacion real end-to-end

### Que entrega ahora `/status`

`GET /projects/{project_id}/status` mantiene el endpoint actual y ahora agrega estos campos utiles para validacion real:

- `engine`
- `current_stage`
- `progress`
- `message`
- `metrics`
- `processing_metadata` enriquecido con logs, tiempos, workspace, artefactos y fallback

### Script automatico sin mock

El script [tests/run_real_colmap_e2e.py](tests/run_real_colmap_e2e.py) ejecuta este flujo real:

1. consulta `GET /health`
2. crea un proyecto
3. sube imagenes reales
4. inicia `POST /process`
5. consulta `GET /status` hasta estado terminal
6. falla si el engine final no es `colmap` o si hubo fallback
7. descarga `GET /model`
8. valida que exista el archivo de salida reportado por el backend

Ejecucion recomendada:

```powershell
cd C:\PROYECTO\PROCESAMIENTO
python tests\run_real_colmap_e2e.py --base-url http://127.0.0.1:8000 --output-format obj --images-dir C:\ruta\a\imagenes_reales
```

Si no pasas `--images-dir`, el script intenta autodetectar un dataset local dentro de `data/projects/*/images` con al menos 3 imagenes legibles por COLMAP. Si no encuentra uno valido, falla rapido y te pide una carpeta real de fotos con overlap.

### Pasos exactos para prueba manual

1. Configura el backend para validacion real:

```powershell
$env:LOCAL3D_PROCESSING_ENGINE = 'colmap'
$env:LOCAL3D_COLMAP_BINARY = 'C:\Tools\COLMAP\COLMAP.bat'
$env:LOCAL3D_COLMAP_FALLBACK_TO_MOCK = 'false'
```

2. Inicia FastAPI:

```powershell
cd C:\PROYECTO\PROCESAMIENTO
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

3. Verifica el motor:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Debes ver `engine = colmap`.

4. Crea el proyecto:

```powershell
$project = Invoke-RestMethod -Method POST -Uri http://127.0.0.1:8000/projects -ContentType 'application/json' -Body '{"name":"validacion-colmap-real"}'
$project.id
```

5. Sube imagenes reales con overlap. No uses los `.jpg` dummy de pruebas del repo si no son legibles por COLMAP:

```powershell
$projectId = $project.id
curl.exe -X POST "http://127.0.0.1:8000/projects/$projectId/images" ^
  -F "files=@C:\ruta\img_01.jpg" ^
  -F "files=@C:\ruta\img_02.jpg" ^
  -F "files=@C:\ruta\img_03.jpg"
```

6. Inicia el procesamiento:

```powershell
Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/projects/$projectId/process" -ContentType 'application/json' -Body '{"output_format":"obj"}'
```

7. Consulta estado hasta `completed` o `failed`:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/projects/$projectId/status"
```

Durante el proceso revisa:

- `engine` debe seguir siendo `colmap`
- `current_stage` debe avanzar por `starting`, `feature_extractor`, `exhaustive_matcher`, `mapper`, `export`, `completed`
- `progress` debe incrementarse
- `processing_metadata.logs.directory` debe apuntar a `output/logs/colmap`
- `metrics` debe incluir tiempo total, imagenes procesadas, camaras reconstruidas y puntos 3D

8. Descarga el modelo final:

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/projects/$projectId/model" -OutFile "C:\Temp\$projectId.obj"
```

9. Verifica artefactos locales del proyecto:

```powershell
Get-ChildItem "C:\PROYECTO\PROCESAMIENTO\data\projects\$projectId\output" -Recurse
```

Debes ver al menos:

- `{project_id}_model.obj` o `{project_id}_model.glb`
- `{project_id}_sparse.ply` cuando COLMAP logra exportarlo
- `colmap_sparse_txt\cameras.txt`
- `colmap_sparse_txt\images.txt`
- `colmap_sparse_txt\points3D.txt`
- `workspace\sparse\0\...`
- `logs\colmap\*.stdout.log`
- `logs\colmap\*.stderr.log`
## Nota academica

Con esta estructura, el proyecto puede defenderse asi:

- la app movil es la interfaz de captura y visualizacion
- el backend local ejecuta la reconstruccion 3D
- la reconstruccion esta separada por etapas
- el backend ya puede ejecutar SfM real con COLMAP y dejar trazas verificables para la defensa academica
