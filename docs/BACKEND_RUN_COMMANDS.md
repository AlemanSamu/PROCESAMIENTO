# Backend Run Commands (Prueba Real APK + COLMAP)

Ejecutar exactamente en PowerShell:

```powershell
cd C:\GRADO\PROYECTO\PROCESAMIENTO
.\.venv\Scripts\Activate.ps1

$env:LOCAL3D_PROFILE="quality"
$env:LOCAL3D_PROCESSING_ENGINE="colmap"
$env:LOCAL3D_COLMAP_BINARY="C:\Tools\COLMAP\COLMAP.bat"
$env:LOCAL3D_COLMAP_USE_GPU="true"
$env:LOCAL3D_COLMAP_GPU_MODE="enabled"
$env:LOCAL3D_COLMAP_ENABLE_DENSE_STAGES="true"
$env:LOCAL3D_COLMAP_REQUIRE_DENSE_RECONSTRUCTION="false"
$env:LOCAL3D_PRIMITIVE_BOX_FALLBACK_ENABLED="true"
$env:LOCAL3D_API_KEY="contrasena"

uvicorn main:app --host 0.0.0.0 --port 8000
```

## Verificacion rapida de health

Desde el PC:

```powershell
curl http://127.0.0.1:8000/health
```

Desde otro dispositivo en la red (ajustar IP):

```text
http://<IP_LOCAL_PC>:8000/health
```

## Notas operativas

- Si el celular no conecta, revisar firewall y que ambos equipos esten en la misma subred.
- Si usas API key en app, debe coincidir con `LOCAL3D_API_KEY`.
- Para evidencia de entorno GPU, ejecutar:

```powershell
nvidia-smi -L
.\.venv\Scripts\python.exe scripts\check_colmap_setup.py
```
