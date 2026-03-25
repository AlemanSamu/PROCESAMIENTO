# Integracion Flutter con Backend Local

Esta carpeta contiene una primera version funcional del cliente Flutter para consumir el backend local FastAPI.

## Arquitectura propuesta

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
7. Manejo de loading, errores y estados vacios.

## Dependencias recomendadas

Agrega en `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  http: ^1.2.2
  image_picker: ^1.1.2
  model_viewer_plus: ^1.9.3
```

## Configuracion de URL del backend

Edita `lib/config/local_backend_config.dart` con la IP LAN del PC:

```dart
static const String baseUrl = 'http://192.168.1.100:8000';
```

## Android: permitir trafico HTTP local

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
4. Revisar estado hasta `Completed`.
5. Abrir visor 3D.
