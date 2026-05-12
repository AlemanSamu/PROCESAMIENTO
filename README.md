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
|   |   |-- input_image_validator.py
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
6. El backend valida automaticamente el lote (`apta`, `advertida`, `rechazada`) antes del motor 3D.
7. El backend lanza el job en segundo plano solo con imagenes aceptadas.
8. Flutter consulta `GET /projects/{id}/status` hasta que el estado sea `completed`.
9. Flutter descarga `GET /projects/{id}/model`.
10. La app guarda y abre el modelo en su visor 3D.

## API local

- `GET /health`
- `POST /projects`
- `POST /projects/{project_id}/images`
- `POST /projects/{project_id}/process`
- `GET /projects/{project_id}/status`
- `GET /projects/{project_id}/result`
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

### Cuando COLMAP genera solo nube sparse

Si COLMAP logra SfM pero no produce una malla densa utilizable, el resultado final puede verse como puntos.

- `sparse` significa que hay camaras y puntos 3D reconstruidos, pero sin una superficie cerrada con caras.
- ver puntos no es un error del sistema: es evidencia real parcial de fotogrametria.
- el pipeline no debe reportarlo como `success_real`; se clasifica como `success_sparse_only` o `fallback_completed` segun densidad y cobertura.
- `fallback_completed` con SfM real parcial indica continuidad academica honesta: hubo reconstruccion real, pero insuficiente para malla defendible como resultado final.

Para mejorar la captura y subir de sparse a malla:

- tomar mas fotos con mayor overlap
- variar altura y angulo entre tomas
- mejorar iluminacion uniforme
- evitar superficies lisas o brillantes
- buscar mas textura visual en el objeto/escena

En tesis, esta salida se puede usar como evidencia de robustez del pipeline y de trazabilidad tecnica (camaras, puntos, reportes), dejando explicito que no es reconstruccion densa final.

### Generacion de superficie aproximada desde sparse

Cuando la etapa densa de COLMAP no produce una malla utilizable pero existe SfM sparse real, el backend intenta reconstruir una superficie aproximada:

- limpia outliers y normaliza la nube sparse
- intenta metodos progresivos (`ball_pivoting` con Open3D si esta disponible, `alpha_shape`/`convex_hull` con SciPy/Trimesh)
- si falla todo lo anterior, cae a una malla de contencion (bounding mesh)
- conserva la nube sparse como evidencia secundaria

Esta salida no se reporta como `success_real`. Se clasifica como `success_approx_surface` cuando la malla aproximada es visible y suficientemente consistente.

Nota importante sobre metricas:

- `visual_faces_count` / `visual_vertices_count` pueden ser altos en modos como `point_spheres`.
- esas caras son de visualizacion, no de reconstruccion real.
- la clasificacion usa `real_geometry_metrics` (`dense_faces_count_real`, `surface_faces_count_real`, etc.), no la geometria visual.

Interpretacion academica:

- no equivale a dense real de COLMAP
- si mantiene forma general y continuidad volumetrica, es defendible como reconstruccion aproximada basada en SfM
- permite entregar un GLB visible y entendible en lugar de solo puntos cuando hay informacion sparse suficiente

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
- ejecuta `input_image_validator.py` antes de reconstruccion
- bloquea el lote si no cumple minimos y expone razones en `processing_metadata.input_validation`
- filtra imagenes rechazadas y entrega al engine solo `accepted_images`
- no contiene la logica algoritmica pesada
- delega esa logica al engine elegido

## Estructura de almacenamiento

Cada proyecto se guarda asi:

```text
data/projects/{project_id}/
|-- images/
|-- output/
|   |-- preprocessed_images/
|   |-- validation/
|   |   |-- input_image_validation_report.json
|   |   |-- accepted_images/
|   |-- pipeline/
|   |   |-- preprocessing_manifest.json
|   |   |-- fallback_report.json
|   |   |-- features_and_matches.json
|   |   |-- poses.json
|   |   |-- point_cloud.json
|   |   |-- mesh.json
|   |   |-- export.json
|   |   |-- {project_id}_execution_report.json
|   |   |-- {project_id}_technical_evidence.json
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
- `LOCAL3D_IMAGE_VALIDATION_ENABLED=true|false`
- `LOCAL3D_IMAGE_VALIDATION_MIN_IMAGES_REQUIRED=6`
- `LOCAL3D_IMAGE_VALIDATION_MIN_WIDTH=640`
- `LOCAL3D_IMAGE_VALIDATION_MIN_HEIGHT=480`
- `LOCAL3D_IMAGE_VALIDATION_MIN_PIXELS=307200`
- `LOCAL3D_IMAGE_VALIDATION_MIN_SHARPNESS_WARN=0.06`
- `LOCAL3D_IMAGE_VALIDATION_MIN_SHARPNESS_REJECT=0.04`
- `LOCAL3D_IMAGE_VALIDATION_MIN_BRIGHTNESS=0.15`
- `LOCAL3D_IMAGE_VALIDATION_MAX_BRIGHTNESS=0.9`
- `LOCAL3D_IMAGE_VALIDATION_EXPOSURE_WARN_MARGIN=0.07`
- `LOCAL3D_IMAGE_VALIDATION_NEAR_DUPLICATE_WARN_HAMMING=6`
- `LOCAL3D_IMAGE_VALIDATION_NEAR_DUPLICATE_REJECT_HAMMING=2`
- `LOCAL3D_IMAGE_VALIDATION_COVERAGE_MIN_UNIQUE_RATIO=0.55`
- `LOCAL3D_IMAGE_VALIDATION_COVERAGE_MIN_MEDIAN_HAMMING=8`
- `LOCAL3D_IMAGE_VALIDATION_COVERAGE_MAX_NEIGHBOR_SIMILARITY_RATIO=0.7`
- `LOCAL3D_IMAGE_VALIDATION_BLOCK_ON_LOW_COVERAGE=false`
- `LOCAL3D_COLMAP_BINARY=C:\\ruta\\a\\COLMAP.bat` o `C:\\ruta\\a\\colmap.exe`
- `LOCAL3D_COLMAP_TIMEOUT_SECONDS=1800`
- `LOCAL3D_COLMAP_USE_GPU=false`
- `LOCAL3D_COLMAP_ENABLE_DENSE_STAGES=true|false`
- `LOCAL3D_COLMAP_CAMERA_MODEL=SIMPLE_RADIAL`
- `LOCAL3D_COLMAP_SINGLE_CAMERA=true`
- `LOCAL3D_COLMAP_FALLBACK_TO_MOCK=false`
- `LOCAL3D_COLMAP_REQUIRE_DENSE_RECONSTRUCTION=false`
- `LOCAL3D_METRICS_EVIDENCE_ENABLED=true|false`
- `LOCAL3D_METRICS_EVIDENCE_ROOT=data/experiments`
- `LOCAL3D_METRICS_EXPERIMENT_VARIANT=baseline|enhanced`
- `LOCAL3D_METRICS_EXPERIMENT_SCENARIO=auto|good|mixed|bad`

### Windows 10

- COLMAP no se instala desde este backend; debe estar instalado previamente en Windows 10.
- Si el ejecutable no esta en `PATH`, configura `LOCAL3D_COLMAP_BINARY` con la ruta absoluta.
- Si tu instalacion de COLMAP no tiene soporte CUDA, deja `LOCAL3D_COLMAP_USE_GPU=false`.
- Si quieres evitar por completo las etapas densas/CUDA, configura `LOCAL3D_COLMAP_ENABLE_DENSE_STAGES=false`.
- Si quieres permitir procesamiento en CPU (por ejemplo Intel HD), deja `LOCAL3D_COLMAP_REQUIRE_DENSE_RECONSTRUCTION=false` para habilitar fallback sparse.
- Si quieres exigir reconstruccion densa real, configura `LOCAL3D_COLMAP_REQUIRE_DENSE_RECONSTRUCTION=true`; en ese caso el backend falla rapido con mensaje explicito cuando el binario no tenga CUDA.

## Que sigue siendo mock

- La reconstruccion geometrica real aun no usa COLMAP ni otro motor fotogrametrico real.
- El motor por defecto ahora es `auto`: intenta COLMAP y, segun configuracion, puede caer a mock de forma controlada.
- La salida GLB/OBJ es valida y util para pruebas, pero no representa todavia una reconstruccion real de vision por computador.

## Limitaciones actuales

- Si COLMAP no tiene CUDA o si `LOCAL3D_COLMAP_ENABLE_DENSE_STAGES=false`, el backend puede terminar en `completed_with_fallback` con una malla aproximada desde sparse (convex hull o bounding box).
- Si `LOCAL3D_COLMAP_REQUIRE_DENSE_RECONSTRUCTION=true` y no hay CUDA, el proyecto termina en `failed` con `reason_code=dense_reconstruction_unavailable`.
- La seleccion del mejor modelo sparse se hace por presencia de archivos y tamano aproximado; no hay ranking fotogrametrico avanzado.
- Si COLMAP produce un modelo sparse vacio o no crea `sparse/0`, el proceso queda en `failed` con error explicito para no ocultar la falla real.

## Futuro de integracion real

La arquitectura ya permite seguir evolucionando la reconstruccion real usando COLMAP sin rehacer Flutter ni las rutas HTTP.

Los siguientes pasos naturales son:

- agregar `image_undistorter` y `patch_match_stereo` para una fase densa
- convertir el resultado real a una malla mas util
- incorporar una conversion posterior a `GLB` de malla cuando exista ese artefacto

## Ejecucion

### Recrear entorno Windows desde cero

Si `.venv` quedo roto por cambio de usuario, ruta o version de Python, recrealo:

```powershell
cd C:\GRADO\PROYECTO\PROCESAMIENTO
Rename-Item .venv .venv_broken_2026_05_02 -ErrorAction SilentlyContinue
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Validacion rapida:

```powershell
python -c "from main import app; print(app.title)"
python -m pytest -q
```

```powershell
cd C:\GRADO\PROYECTO\PROCESAMIENTO
.\.venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Notas de conectividad:

- Usa `--host 0.0.0.0` para aceptar conexiones desde APK/dispositivos en LAN.
- `GET /health` incluye `network.preferred_base_url` y `network.advertised_urls` para ayudar a configurar la app Android.
- CORS ahora es configurable por `.env`:
  - `LOCAL3D_CORS_ALLOWED_ORIGINS=*`
  - `LOCAL3D_CORS_ALLOW_CREDENTIALS=false`

## Perfiles de procesamiento

El backend soporta `LOCAL3D_PROFILE`:

- `conservative`: usa menos imagenes, ancho maximo menor y GPU en modo automatico/CPU. Util para pruebas rapidas.
- `balanced`: recomendado para la ASUS V3607V. Usa RTX 4050 si esta disponible, maximo de imagenes medio y validacion estandar.
- `quality`: usa mas imagenes y mayor ancho objetivo. Recomendado cuando el dataset tiene buen overlap y buena iluminacion.

Ejemplo recomendado para tu PC:

```powershell
$env:LOCAL3D_PROFILE = "balanced"
$env:LOCAL3D_PROCESSING_ENGINE = "auto"
$env:LOCAL3D_COLMAP_USE_GPU = "true"
$env:LOCAL3D_COLMAP_GPU_MODE = "auto"
$env:LOCAL3D_COLMAP_ENABLE_DENSE_STAGES = "false"
$env:LOCAL3D_COLMAP_FALLBACK_TO_MOCK = "false"
$env:LOCAL3D_PRIMITIVE_BOX_FALLBACK_ENABLED = "true"
```

Con esta configuracion, si COLMAP no esta instalado o falla, el backend intenta generar un fallback academico tipo caja con metadata `fallback_used=true`. No se reporta como reconstruccion fotogrametrica real.

## COLMAP y RTX 4050

1. Instala COLMAP para Windows.
2. Si `colmap` no esta en `PATH`, configura la ruta:

```powershell
$env:LOCAL3D_COLMAP_BINARY = "C:\Tools\COLMAP\COLMAP.bat"
```

3. Verifica GPU:

```powershell
nvidia-smi -L
```

4. Inicia backend:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000
```

Para pruebas estables se deja `LOCAL3D_COLMAP_ENABLE_DENSE_STAGES=false` por defecto. Activalo solo cuando COLMAP tenga soporte CUDA correcto:

```powershell
$env:LOCAL3D_COLMAP_ENABLE_DENSE_STAGES = "true"
```

## Prueba rapida del pipeline

Sin COLMAP, con fallback academico:

```powershell
$env:LOCAL3D_PROCESSING_ENGINE = "auto"
$env:LOCAL3D_PRIMITIVE_BOX_FALLBACK_ENABLED = "true"
$env:LOCAL3D_COLMAP_FALLBACK_TO_MOCK = "false"
python scripts\test_pipeline.py --input C:\ruta\imagenes --output-format glb
```

Con COLMAP:

```powershell
$env:LOCAL3D_PROCESSING_ENGINE = "colmap"
$env:LOCAL3D_COLMAP_BINARY = "C:\Tools\COLMAP\COLMAP.bat"
$env:LOCAL3D_COLMAP_USE_GPU = "true"
python scripts\test_pipeline.py --input C:\ruta\imagenes --output-format obj
```

El script consulta `/health`, crea proyecto, sube imagenes, procesa, consulta `/status`, lee `/result` y descarga el modelo si existe.

## Fase 2 - Preprocesamiento real y fallback académico

Antes de entregar imagenes a COLMAP, el backend ejecuta una etapa real de preprocesamiento cuando esta activo el sistema de perfiles (`LOCAL3D_PROFILE=conservative|balanced|quality`).

El preprocesamiento:

- corrige orientacion EXIF;
- redimensiona manteniendo proporcion segun `LOCAL3D_IMAGE_PREPROCESSING_MAX_WIDTH` y el perfil activo;
- normaliza las imagenes a JPG/PNG;
- aplica CLAHE cuando detecta contraste bajo;
- aplica denoise suave con OpenCV;
- calcula ancho, alto, brillo, contraste, nitidez, `blur_score`, pixeles y tamano del archivo;
- clasifica cada imagen como `accepted`, `warning` o `rejected`.

Los originales no se modifican y permanecen en:

```text
data/projects/{project_id}/images/
```

Las imagenes normalizadas quedan en:

```text
data/projects/{project_id}/output/preprocessed_images/
```

El manifiesto tecnico queda en:

```text
data/projects/{project_id}/output/pipeline/preprocessing_manifest.json
```

COLMAP usa preferiblemente `output/preprocessed_images/` cuando existe. Si una configuracion legacy no activa perfiles, el servicio conserva el flujo anterior y puede usar `selected_images` o `segmented_images`.

Si COLMAP no esta instalado, falla, no encuentra suficientes coincidencias, genera sparse vacio o no puede hacer reconstruccion densa, el backend puede recuperar el proceso con fallback academico tipo caja parametrica cuando `LOCAL3D_PRIMITIVE_BOX_FALLBACK_ENABLED=true`.

El fallback queda declarado de forma explicita en:

```text
data/projects/{project_id}/output/pipeline/fallback_report.json
```

Ese reporte incluye `reason_code`, `reason_message`, motor intentado, disponibilidad de COLMAP, imagenes usadas, limitaciones, explicacion academica y ruta del modelo generado. Para el informe de grado, usa estos archivos como evidencia de trazabilidad:

- `preprocessing_manifest.json`: demuestra preparacion real de imagenes y metricas de calidad.
- `fallback_report.json`: demuestra control de fallo y resultado minimo defendible.
- `execution_report.json` y `technical_evidence.json`: muestran tiempos, estados, artefactos y decision final.

Tambien puedes consultar todo desde:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/projects/$projectId/result"
```

Campos utiles de `/result`:

- `preprocessing_summary`
- `fallback_report`
- `artifact_paths`
- `warnings`
- `recommended_next_action`

## Errores comunes

- `ModuleNotFoundError: fastapi`: activa `.venv` e instala `pip install -r requirements.txt`.
- `No Python at ...`: recrea `.venv`; quedo apuntando a una ruta antigua.
- `colmap no se reconoce`: instala COLMAP o configura `LOCAL3D_COLMAP_BINARY`.
- `API key faltante`: define `LOCAL3D_API_KEY` o envia header `X-API-Key`.
- `insufficient_valid_images`: sube mas fotos nitidas, con overlap y buena iluminacion.
- `fallback_used=true`: COLMAP fallo o no estaba disponible; el modelo es aproximado academico, no SfM real.
- Procesamiento muy lento: usa `LOCAL3D_PROFILE=conservative` o reduce cantidad de imagenes.

## Pruebas

```powershell
python -m unittest discover -s tests
```

## Runbook Automatico

Para operar y validar en un solo comando:

```powershell
.\scripts\start_and_validate.ps1 -Port 8000 -OutputFormat glb -MaxImages 24
```

Si `LOCAL3D_API_KEY` esta configurada en `.env`, el script la detecta automaticamente y envia `X-API-Key`.  
Puedes forzar una clave puntual con:

```powershell
.\scripts\start_and_validate.ps1 -Port 8000 -ApiKey "TU_API_KEY"
```

Opcional con dataset fijo:

```powershell
.\scripts\start_and_validate.ps1 -Port 8000 -ImagesDir C:\ruta\dataset_patron -OutputFormat glb
```

Checklist de salida a produccion:

```text
GO_LIVE_CHECKLIST.md
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
$env:LOCAL3D_COLMAP_ENABLE_DENSE_STAGES = 'false' # evita etapas densas/CUDA
$env:LOCAL3D_COLMAP_FALLBACK_TO_MOCK = 'false'
$env:LOCAL3D_COLMAP_REQUIRE_DENSE_RECONSTRUCTION = 'false' # Intel/CPU profile
# $env:LOCAL3D_COLMAP_REQUIRE_DENSE_RECONSTRUCTION = 'true' # exige denso real y falla sin CUDA
$env:LOCAL3D_API_KEY = 'TU_API_KEY'
```

2. Inicia FastAPI:

```powershell
cd C:\PROYECTO\PROCESAMIENTO
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

3. Verifica el motor:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health -Headers @{ "X-API-Key" = "TU_API_KEY" }
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
- `pipeline/{project_id}_execution_report.json`
- `pipeline/{project_id}_technical_evidence.json`
- `colmap_sparse_txt\cameras.txt`
- `colmap_sparse_txt\images.txt`
- `colmap_sparse_txt\points3D.txt`
- `workspace\sparse\0\...`
- `logs\colmap\*.stdout.log`
- `logs\colmap\*.stderr.log`

## Reportes comparativos para tesis

Cada corrida puede anexar evidencia normalizada en:

- `output/pipeline/{project_id}_technical_evidence.json`
- `data/experiments/processing_runs.ndjson`

Con ese historial, puedes generar reporte comparativo reutilizable (JSON + CSV):

```powershell
python scripts\generate_experiment_report.py --before-variant baseline --after-variant enhanced
```

Salida esperada:

- `data/experiments/reports/processing_experiment_summary.json`
- `data/experiments/reports/processing_runs_table.csv`
- `data/experiments/reports/processing_stage_timings_table.csv`
- `data/experiments/reports/processing_reason_frequencies_table.csv`

## Fase 3 - COLMAP real, RTX 4050 y métricas comparativas

Esta fase configura COLMAP como camino real de reconstruccion SfM en Windows, manteniendo fallback academico cuando el motor no logra una salida defendible.

### Instalar COLMAP en Windows

1. Descarga una version de COLMAP para Windows desde el proyecto oficial.
2. Descomprime COLMAP en una ruta estable, por ejemplo `C:\Tools\COLMAP`.
3. Verifica que exista `COLMAP.bat` o `colmap.exe`.
4. Configura la ruta explicita para que la evidencia sea reproducible:

```powershell
$env:LOCAL3D_COLMAP_BINARY = "C:\Tools\COLMAP\COLMAP.bat"
```

Si `colmap` ya esta en `PATH`, el backend puede detectarlo, pero para el informe conviene fijar `LOCAL3D_COLMAP_BINARY`.

### Verificar RTX 4050

Ejecuta:

```powershell
nvidia-smi -L
python scripts\check_colmap_setup.py
```

El diagnostico reporta si `nvidia-smi` detecta GPU, si aparece RTX 4050, que binario COLMAP responde, version detectada, perfiles recomendados y si conviene usar GPU o CPU.

Tambien lista comandos disponibles (`feature_extractor`, `matcher`, `mapper`, `model_converter`, etapas densas), version detectada y salida de `colmap -h` para trazabilidad.

### Perfiles COLMAP

- `conservative`: `SiftExtraction.use_gpu=0`, `SiftMatching.use_gpu=0`, denso deshabilitado. Usalo para pruebas rapidas y para aislar problemas.
- `balanced`: usa GPU si `nvidia-smi` detecta GPU NVIDIA, denso deshabilitado por defecto. Es el perfil recomendado para sustentacion estable en ASUS V3607V con RTX 4050.
- `quality`: usa GPU si esta disponible, timeout mayor y denso opcional. Usalo para evidencia final cuando tengas mas tiempo y buenas fotos.

### Probar GPU y fallback CPU

Forzar intento de GPU:

```powershell
$env:LOCAL3D_COLMAP_USE_GPU = "true"
$env:LOCAL3D_COLMAP_GPU_MODE = "enabled"
```

Forzar CPU:

```powershell
$env:LOCAL3D_COLMAP_USE_GPU = "false"
$env:LOCAL3D_COLMAP_GPU_MODE = "disabled"
```

En perfil `balanced` y `conservative`, si hay error CUDA en runtime, el backend puede caer a CPU para mantener continuidad. En `quality`, el flujo prioriza evidenciar el fallo para no ocultar limitaciones de hardware/configuracion.

Configuracion sugerida para ASUS V3607V:

```powershell
$env:LOCAL3D_PROFILE = "balanced"
$env:LOCAL3D_PROCESSING_ENGINE = "colmap"
$env:LOCAL3D_COLMAP_BINARY = "C:\Tools\COLMAP\COLMAP.bat"
$env:LOCAL3D_COLMAP_ENABLE_DENSE_STAGES = "false"
$env:LOCAL3D_COLMAP_REQUIRE_DENSE_RECONSTRUCTION = "false"
$env:LOCAL3D_PRIMITIVE_BOX_FALLBACK_ENABLED = "true"
```

### Interpretar `colmap_report.json`

Cada ejecucion deja:

```text
data/projects/{project_id}/output/pipeline/colmap_report.json
```

Campos clave:

- `colmap_binary`, `colmap_version`: prueban que se uso COLMAP real.
- `gpu_detected`, `rtx_4050_detected`, `gpu_used`: evidencian disponibilidad y uso de GPU.
- `profile`: documenta el perfil experimental.
- `commands_executed`, `command_durations`: trazabilidad por etapa.
- `sparse_created`, `cameras_reconstructed`, `images_registered`, `points3D_count`: calidad minima de la reconstruccion real.
- `model_outputs`: rutas de TXT, PLY, OBJ/GLB y logs.
- `fallback_used`, `failure_reason`: separan SfM real de recuperacion academica.

Metadatos GPU clave:

- `gpu_requested`
- `gpu_used`
- `gpu_fallback_to_cpu`
- `gpu_error_message`

Si COLMAP genera sparse real, `fallback_used=false` aunque el OBJ final se construya desde la nube sparse porque las etapas densas esten deshabilitadas. Si COLMAP falla y se recupera con caja academica, `fallback_used=true` y tambien se genera `fallback_report.json`.

### Interpretar `quality_report.json`

Cada corrida deja:

```text
data/projects/{project_id}/output/pipeline/quality_report.json
```

Clasificaciones:

- `success_real`: reconstruccion real utilizable.
- `success_sparse_only`: resultado sparse utilizable sin denso completo.
- `fallback_completed`: se recupero con fallback academico.
- `failed`: no se obtuvo resultado usable.

En la tesis, usa este archivo para justificar estado final, limitaciones y siguiente accion recomendada.

### Comparación de perfiles

Para comparar `conservative`, `balanced` y `quality` sobre el mismo dataset:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py --input C:\ruta\imagenes --profiles conservative balanced quality --output-format glb
```

Se generan:

- `data/experiments/reports/profile_comparison.json`
- `data/experiments/reports/profile_comparison.csv`

Campos comparados: tiempo total, imagenes aceptadas, puntos 3D, camaras reconstruidas, uso de fallback, tamaño de modelo y clasificacion final.

### Evidencias para el informe

Guarda capturas o anexos de:

- `python scripts\check_colmap_setup.py`
- `nvidia-smi -L`
- `output/pipeline/colmap_report.json`
- `output/pipeline/fallback_report.json` cuando exista
- `output/colmap_sparse_txt/points3D.txt`
- `output/logs/colmap/*.stdout.log` y `*.stderr.log`
- captura del modelo OBJ/GLB abierto desde Flutter o visor externo

### Prueba real con fotos

Recomendacion de captura:

- 20 a 40 fotos del mismo objeto
- objeto con textura visible
- fondo quieto
- buena luz
- rodear el objeto en circulo
- no usar zoom
- no usar fotos borrosas
- cada foto debe solaparse con la anterior

Comandos base:

```powershell
python scripts\check_colmap_setup.py
uvicorn main:app --reload
```

Luego crea proyecto, sube fotos y procesa como en la seccion de validacion end-to-end COLMAP.
## Nota academica

Con esta estructura, el proyecto puede defenderse asi:

- la app movil es la interfaz de captura y visualizacion
- el backend local ejecuta la reconstruccion 3D
- la reconstruccion esta separada por etapas
- el backend ya puede ejecutar SfM real con COLMAP y dejar trazas verificables para la defensa academica

## Fase 4 - Integracion Flutter con reportes tecnicos

La app Flutter ahora consume `GET /projects/{project_id}/result` para mostrar evidencia tecnica del backend sin romper el flujo actual de crear proyecto, subir imagenes, procesar y abrir visor.

### Ejecutar la app Flutter

```powershell
cd C:\GRADO\PROYECTO\PROCESAMIENTO\flutter_integration
flutter pub get
flutter run
```

### Configurar IP y backend URL

Opciones recomendadas:

1. Pantalla **Configuracion del backend** dentro de la app.
2. `dart-define` al ejecutar:

```powershell
flutter run --dart-define=LOCAL_BACKEND_URL=http://192.168.1.100:8000 --dart-define=LOCAL_BACKEND_API_KEY=TU_API_KEY
```

### Conectar con backend local

1. Inicia FastAPI:

```powershell
cd C:\GRADO\PROYECTO\PROCESAMIENTO
.\.venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

2. En Flutter ingresa URL + API key (si aplica).
3. Usa **Probar conexion**; la app valida `/health`.
4. La pantalla de configuracion muestra motor, perfil, disponibilidad de COLMAP y estado de GPU cuando el backend lo reporta.

### Interpretar resultados en Flutter

La pantalla de estado muestra:

- estado general, etapa actual y progreso
- motor usado (`colmap`, `mock` o fallback)
- perfil usado
- uso de GPU y fallback
- clasificacion final de calidad
- recomendacion tecnica del backend

Mensajes de clasificacion:

- `success_real`: reconstruccion real completada con COLMAP.
- `success_sparse_only`: reconstruccion real parcial desde nube sparse.
- `fallback_completed`: modelo aproximado porque no fue posible la reconstruccion real.
- `failed`: no se pudo generar un modelo usable.

Si el resultado final es GLB, el visor principal se abre desde la app. Si es OBJ, la app muestra ruta/descarga y avisa que el visor principal esta optimizado para GLB.

### Generar APK

```powershell
cd C:\GRADO\PROYECTO\PROCESAMIENTO\flutter_integration
flutter build apk --release
```

## Fase 5 - Prueba real completa y evidencias para sustentacion

Esta fase valida el flujo completo backend + Flutter con fotos reales y deja evidencia lista para la tesis.

> Nota de integridad academica:
> Los modelos 3D previos fueron descartados por no representar reconstrucciones validas.
> La evidencia final debe generarse nuevamente con el flujo de Fase 5 y dataset real.

### 1) Dataset recomendado

```text
data/test_datasets/objeto_01/
```

Guia de captura:

```text
docs/CAPTURE_GUIDE.md
```

### 2) Validar calidad del dataset real

```powershell
python scripts/validate_real_dataset.py --input data/test_datasets/objeto_01
```

Salida esperada:

- cantidad de imagenes
- resolucion promedio
- nitidez promedio
- brillo promedio
- posibles duplicados
- advertencias
- recomendacion: `apto`, `mejorar` o `no_apto`

### 3) Prueba E2E backend con fotos reales

```powershell
python scripts/test_pipeline.py --input data/test_datasets/objeto_01 --output-format glb
```

### 4) Comparacion por perfiles

```powershell
python scripts/run_experiment.py --input data/test_datasets/objeto_01 --profiles conservative balanced quality --output-format glb
```

Artefactos esperados:

- `data/experiments/reports/profile_comparison.json`
- `data/experiments/reports/profile_comparison.csv`

### 5) Checklist de evidencia para tesis

```text
docs/evidence/EVIDENCE_CHECKLIST.md
```

### 6) Generar paquete final de defensa

```powershell
python scripts/generate_defense_package.py --input data/test_datasets/objeto_01
```

Salida:

```text
defense_package/{project_id}/
|-- DEFENSE_SUMMARY.md
|-- package_manifest.json
|-- reports/
|   |-- quality_report.json
|   |-- colmap_report.json
|   |-- fallback_report.json (si aplica)
|   |-- preprocessing_manifest.json
|-- experiments/
|   |-- profile_comparison.json (si existe)
|   |-- profile_comparison.csv (si existe)
|-- logs/colmap/
|-- model/
```

### 7) Interpretacion rapida de resultado

- `success_real`: evidencia principal de reconstruccion real con COLMAP.
- `success_sparse_only`: evidencia real parcial (sparse), util con limitaciones.
- `fallback_completed`: evidencia de robustez del pipeline (salida aproximada academica).
- `failed`: evidencia diagnostica para justificar recaptura o ajuste tecnico.

## Correccion de modelos tipo caja o textura deformada

Este problema aparece cuando COLMAP logra una reconstruccion sparse pobre (pocas camaras/puntos) o cuando no hay malla densa usable y el sistema cae en geometria aproximada.

### Como detectarlo rapido

- Revisa `data/projects/{project_id}/output/pipeline/quality_report.json`.
- Si `quality_classification` es `fallback_completed`, no es reconstruccion fotogrametrica real final.
- Revisa:
  - `points3D_count`
  - `cameras_reconstructed`
  - `geometry_source` (`colmap_dense`, `colmap_sparse`, `primitive_box`, `fallback`)
  - `texture_source` (`real_projection`, `best_image_projection`, `none`)

### Reglas de calidad aplicadas

- `success_real`: solo si hay malla densa real (`geometry_source=colmap_dense`) y reconstruccion suficiente.
- `success_sparse_only`: sparse real utilizable sin denso final.
- `fallback_completed`: caja/fallback/mesh debil o sparse insuficiente para defensa como reconstruccion real.
- `failed`: sin modelo utilizable.

Umbrales minimos de robustez sparse:

- `points3D_count >= 500`
- `cameras_reconstructed >= 6`

### Configuracion recomendada para RTX 4050

```powershell
$env:LOCAL3D_COLMAP_USE_GPU = "true"
$env:LOCAL3D_COLMAP_GPU_MODE = "enabled"
$env:LOCAL3D_COLMAP_ENABLE_DENSE_STAGES = "true"
$env:LOCAL3D_COLMAP_REQUIRE_DENSE_RECONSTRUCTION = "false"
```

Si el denso falla, el pipeline no se rompe: mantiene `success_sparse_only` si el sparse es suficiente, o cae a `fallback_completed` si la reconstruccion sigue debil.

### Diagnostico automatico por proyecto

```powershell
python scripts/inspect_reconstruction_output.py --project-id <id>
```

El script reporta:

- si fue COLMAP real o fallback
- cantidad de camaras y puntos 3D
- disponibilidad densa
- si el modelo final es `primitive_box`
- recomendacion concreta (`repetir fotos`, `activar dense`, `usar perfil quality`, `dataset aceptable`, `resultado defendible solo como fallback`)

## Mejora visual mediante texturizado multi-vista

El pipeline aplica una capa de texturizado liviana para mejorar la presentacion de modelos aproximados (`surface_from_sparse` o `geometric_prior`) cuando la reconstruccion densa real no es usable.

Que hace:
- Prioriza imagenes segmentadas y luego preprocesadas para extraer color visual estable.
- Selecciona vistas candidatas por nitidez, features y area visible del objeto.
- En objetos `box_like`, reparte color por caras principales (frontal, trasera, laterales, superior e inferior) para evitar mallas "lavadas" o grises.
- Si no hay evidencia visual suficiente, usa fallback de `average_image_color`.

Importante para sustentacion:
- Esta mejora **no** convierte un fallback en reconstruccion densa real.
- La clasificacion academica sigue siendo honesta (`success_real`, `success_approx_surface`, `success_sparse_only`, `fallback_completed`, `failed`).
- Si la geometria real no alcanza umbral, el resultado se reporta como aproximado/fallback aunque visualmente se vea mejor.
