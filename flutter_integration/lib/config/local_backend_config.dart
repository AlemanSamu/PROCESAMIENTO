class LocalBackendConfig {
  const LocalBackendConfig._();

  static const String defaultBaseUrl = 'http://192.168.1.100:8000';
  static const String dartDefineBaseUrl = String.fromEnvironment(
    'LOCAL_BACKEND_URL',
    defaultValue: '',
  );
  static const String dartDefineApiKey = String.fromEnvironment(
    'LOCAL_BACKEND_API_KEY',
    defaultValue: '',
  );

  static String get fallbackBaseUrl {
    final trimmed = dartDefineBaseUrl.trim();
    if (trimmed.isNotEmpty) {
      return trimmed;
    }
    return defaultBaseUrl;
  }

  static String? get fallbackApiKey {
    final trimmed = dartDefineApiKey.trim();
    if (trimmed.isEmpty) {
      return null;
    }
    return trimmed;
  }

  static String get baseUrl => fallbackBaseUrl;
}
