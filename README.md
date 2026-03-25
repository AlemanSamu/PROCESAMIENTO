# Backend Local de Procesamiento 3D (Python + FastAPI)

Backend minimo para ejecutar en un PC con Windows 10 como modulo local de procesamiento.
No es un servidor en internet. La app Flutter envia imagenes por red local y este backend procesa localmente.

## Estructura

```text
.
|-- main.py
|-- config.py
|-- app/
|   |-- api/
|   |   |-- router.py
|   |   |-- routes/
|   |   |   |-- projects.py
|   |-- models/
|   |   |-- schemas.py
|   |-- services/
|   |   |-- project_service.py
|   |   |-- storage_service.py
|   |   |-- processing_service.py
|   |   |-- engines/
|   |   |   |-- base_engine.py
|   |   |   |-- mock_engine.py
|   |   |   |-- colmap_engine.py
|   |-- core/
|   |   |-- dependencies.py
|   |   |-- errors.py
|-- data/
|   |-- projects/
```

## Almacenamiento local por proyecto

Cada proyecto se guarda asi:

```text
data/projects/{project_id}/
|-- images/
|-- output/
|-- meta.json
```

## Endpoints locales

- `GET /projects`
- `POST /projects`
- `POST /projects/{project_id}/images`
- `POST /projects/{project_id}/process`
- `GET /projects/{project_id}/status`
- `GET /projects/{project_id}/model`

## Requisitos (Windows 10)

1. Python 3.10 o superior
2. PowerShell
3. (Opcional) COLMAP instalado y en PATH

## Ejecucion

```powershell
cd C:\PROYECTO\PROCESAMIENTO
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Prueba de salud:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Swagger:

`http://127.0.0.1:8000/docs`

## Flujo rapido

Crear proyecto:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/projects" ^
  -H "Content-Type: application/json" ^
  -d "{\"name\":\"ProyectoDemo\"}"
```

Subir imagenes:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/projects/{project_id}/images" ^
  -F "files=@C:/imagenes/img1.jpg" ^
  -F "files=@C:/imagenes/img2.jpg"
```

Iniciar procesamiento:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/projects/{project_id}/process" ^
  -H "Content-Type: application/json" ^
  -d "{\"output_format\":\"glb\"}"
```

Consultar estado:

```powershell
curl.exe "http://127.0.0.1:8000/projects/{project_id}/status"
```

Descargar modelo:

```powershell
curl.exe -L "http://127.0.0.1:8000/projects/{project_id}/model" -o modelo.glb
```

## Motor de procesamiento desacoplado

- Si COLMAP no esta disponible, el backend usa simulacion (`mock_engine.py`).
- Punto de integracion real: `app/services/engines/colmap_engine.py`.
- Para cambiar a COLMAP, implementa ese adaptador y configura `.env`:
  - `LOCAL3D_PROCESSING_ENGINE=colmap`
