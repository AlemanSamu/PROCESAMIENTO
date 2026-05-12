# Real Device E2E Checklist (APK + Backend + COLMAP)

Checklist operativo para la prueba real completa de tesis.

## 1. Red y dispositivos

- [ ] PC y celular conectados a la misma red WiFi.
- [ ] Firewall de Windows permite puerto `8000` para Python/Uvicorn.
- [ ] Confirmar IP local del PC (ejemplo: `192.168.1.120`).

Comando sugerido en PC:

```powershell
ipconfig
```

## 2. Backend local listo

- [ ] Entorno virtual activado en `C:\GRADO\PROYECTO\PROCESAMIENTO`.
- [ ] Variables de entorno configuradas (perfil quality + COLMAP + API key).
- [ ] Backend levantado en `0.0.0.0:8000`.
- [ ] Endpoint de salud responde desde PC: `http://127.0.0.1:8000/health`.

## 3. App Android (APK)

- [ ] APK instalado en celular (`app-debug.apk` o `app-release.apk`).
- [ ] Permiso de camara concedido en Android.
- [ ] URL backend configurada en app: `http://<IP_PC>:8000`.
- [ ] API key configurada en app si aplica.
- [ ] Prueba de conexion en app: OK.

## 4. Captura profesional 3D

- [ ] Perfil seleccionado: `maxima_calidad`.
- [ ] Minimo `45` fotos capturadas.
- [ ] Niveles `bajo / medio / alto` completos.
- [ ] Sin zoom digital.
- [ ] Iluminacion estable y uniforme.
- [ ] Objeto cubierto a 360 grados.
- [ ] Checklist de fin de captura revisado.

## 5. Exportacion / subida / procesamiento

- [ ] Exportacion o carga de imagenes completada sin error.
- [ ] Procesamiento remoto/local iniciado en app.
- [ ] Pipeline COLMAP ejecutado (sparse/dense segun configuracion).
- [ ] Resultado GLB generado.
- [ ] Resultado descargado/visualizado en app (visor GLB).

## 6. Evidencia obligatoria

- [ ] Capturas de pantalla del flujo completo (ver `FINAL_EVIDENCE_CAPTURE.md`).
- [ ] Artefactos JSON y reportes copiados a paquete final.
- [ ] Logs relevantes de COLMAP preservados.

## 7. Criterio de exito E2E

- [ ] Captura real en celular finalizada.
- [ ] Backend procesa dataset real sin caida.
- [ ] Existe modelo final GLB utilizable.
- [ ] Existe paquete de evidencia final para tesis.
