# Integración Flutter con Backend Local

Esta carpeta contiene la app Flutter que consume el backend local FastAPI para crear proyectos, procesar imágenes y mostrar resultados técnicos de reconstrucción.

## Estructura

```text
lib/
|-- main.dart
|-- config/
|   |-- local_backend_config.dart
|-- models/
|   |-- project_models.dart
|-- services/
|   |-- local_api_service.dart
|-- screens/
|   |-- project_history_screen.dart
|   |-- project_status_screen.dart
|   |-- model_viewer_screen.dart
|-- widgets/
|   |-- loading_state.dart
|   |-- error_state.dart
|   |-- empty_state.dart
|   |-- project_card.dart
```

## Funcionalidades incluidas

1. Crear proyectos desde Flutter.
2. Subida multiple de imagenes.
3. Inicio de procesamiento.
4. Pantalla de estado con polling automatico.
5. Historial de proyectos.
6. Visor 3D del modelo final.
7. Consumo de `GET /projects/{id}/result` con:
   - `quality_report`
   - `preprocessing_summary`
   - `fallback_report`
   - `artifact_paths`
   - `warnings`
   - `recommended_next_action`
8. Detalles técnicos (cámaras, puntos 3D, tiempo, perfil, GPU, rutas de reportes).
9. Manejo de loading, errores y estados vacíos.

## Dependencias

Agrega en `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  http: ^1.2.2
  image_picker: ^1.1.2
  model_viewer_plus: ^1.9.3
```

## Ejecutar la app

```powershell
cd C:\GRADO\PROYECTO\PROCESAMIENTO\flutter_integration
flutter pub get
flutter run
```

## Configuración de IP / backend

Tienes dos opciones:

1. Definir valor por defecto en `lib/config/local_backend_config.dart`.
2. Cambiar URL/API key desde la pantalla **Configuración del backend** en la app.

También puedes usar `dart-define`:

```powershell
flutter run --dart-define=LOCAL_BACKEND_URL=http://192.168.1.100:8000 --dart-define=LOCAL_BACKEND_API_KEY=TU_API_KEY
```

Si editas código, ejemplo:

```dart
static const String baseUrl = 'http://192.168.1.100:8000';
```

## Conectar con backend local

1. Levanta FastAPI en el PC:

```powershell
cd C:\GRADO\PROYECTO\PROCESAMIENTO
.\.venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

2. En Flutter abre **Configuración del backend**.
3. Ingresa URL (`http://IP_DEL_PC:8000`) y API key si aplica.
4. Usa **Probar conexión** y confirma `/health`.

## Android: permitir tráfico HTTP local

En `android/app/src/main/AndroidManifest.xml`:

- Asegura permiso de internet:

```xml
<uses-permission android:name="android.permission.INTERNET" />
```

- En `<application ...>` agrega:

```xml
android:usesCleartextTraffic="true"
```

## Flujo de uso en la app

1. Crear proyecto.
2. Subir imagenes.
3. Iniciar procesamiento.
4. Revisar estado y clasificación:
   - `success_real`
   - `success_sparse_only`
   - `fallback_completed`
   - `failed`
5. Abrir visor 3D cuando hay GLB.
6. Si el resultado es OBJ, revisar ruta/descarga y recomendaciones técnicas.

## Interpretar resultados

- `success_real`: reconstrucción real completada con COLMAP.
- `success_sparse_only`: reconstrucción real parcial basada en sparse.
- `fallback_completed`: modelo aproximado por fallback académico.
- `failed`: no se obtuvo modelo usable.

La pantalla de estado incluye recomendaciones del backend para reintentar con `quality` o capturar más fotos.

## Flutter analyze y tests

```powershell
flutter analyze
flutter test
```

## Generar APK

```powershell
flutter build apk --release
```
