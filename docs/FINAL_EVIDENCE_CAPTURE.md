# Final Evidence Capture (Tesis)

Capturar evidencia minima obligatoria del flujo real E2E.

## A. Evidencia visual (capturas/pantallas)

- [ ] APK instalado en celular.
- [ ] Pantalla de configuracion backend en app.
- [ ] Prueba de conexion exitosa.
- [ ] Modo captura profesional.
- [ ] Checklist de captura.
- [ ] Fotos capturadas.
- [ ] Procesamiento en app.
- [ ] Resultado final.
- [ ] Visor GLB.

## B. Evidencia tecnica (archivos)

- [ ] `quality_report.json`
- [ ] `colmap_report.json`
- [ ] `capture_metadata.json`
- [ ] `dataset_validation_report.json`
- [ ] `fallback_report.json` (si existe)
- [ ] `preprocessing_manifest.json`
- [ ] GLB final
- [ ] logs COLMAP

## C. Evidencia de entorno

- [ ] `nvidia-smi -L`
- [ ] `scripts/check_colmap_setup.py`

## D. Empaquetado final

```powershell
cd C:\GRADO\PROYECTO\PROCESAMIENTO
.\.venv\Scripts\python.exe scripts\collect_final_evidence.py --project-id <project_id>
```

Salida esperada:

`defense_package/final_e2e_<project_id>/`
