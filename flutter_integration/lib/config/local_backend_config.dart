class LocalBackendConfig {
  const LocalBackendConfig._();

  // Cambia este valor por la IP LAN del PC donde corre FastAPI.
  static const String baseUrl = String.fromEnvironment(
    'LOCAL_BACKEND_URL',
    defaultValue: 'http://192.168.1.100:8000',
  );
}
