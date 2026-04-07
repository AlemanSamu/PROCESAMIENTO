import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../config/local_backend_config.dart';
import 'local_api_service.dart';

class BackendConnectionConfig {
  const BackendConnectionConfig({
    required this.baseUrl,
    this.apiKey,
  });

  final String baseUrl;
  final String? apiKey;

  BackendConnectionConfig copyWith({
    String? baseUrl,
    String? apiKey,
    bool clearApiKey = false,
  }) {
    return BackendConnectionConfig(
      baseUrl: baseUrl ?? this.baseUrl,
      apiKey: clearApiKey ? null : (apiKey ?? this.apiKey),
    );
  }
}

class BackendUrlStore {
  const BackendUrlStore._();

  static const String _baseUrlPrefsKey = 'local_backend_base_url';
  static const String _apiKeyPrefsKey = 'local_backend_api_key';

  static Future<String?> loadSavedBaseUrl() async {
    final prefs = await SharedPreferences.getInstance();
    final savedValue = prefs.getString(_baseUrlPrefsKey);
    if (savedValue == null) {
      _logDebug('No hay URL guardada en SharedPreferences.');
      return null;
    }

    final normalized = normalizeBaseUrl(savedValue);
    if (normalized == null) {
      _logDebug('Se ignora URL guardada invalida: "$savedValue"');
      await prefs.remove(_baseUrlPrefsKey);
      return null;
    }

    _logDebug('URL guardada encontrada: $normalized');
    return normalized;
  }

  static Future<String?> loadSavedApiKey() async {
    final prefs = await SharedPreferences.getInstance();
    final savedValue = prefs.getString(_apiKeyPrefsKey);
    if (savedValue == null) {
      _logDebug('No hay API key guardada en SharedPreferences.');
      return null;
    }

    final normalized = normalizeApiKey(savedValue);
    if (normalized == null) {
      _logDebug('Se elimina API key guardada vacia o invalida.');
      await prefs.remove(_apiKeyPrefsKey);
      return null;
    }

    _logDebug('API key guardada encontrada. configured=true');
    return normalized;
  }

  static Future<BackendConnectionConfig> loadResolvedConnectionConfig() async {
    final savedBaseUrl = await loadSavedBaseUrl();
    final resolvedBaseUrl = savedBaseUrl ??
        normalizeBaseUrl(LocalBackendConfig.fallbackBaseUrl) ??
        LocalBackendConfig.defaultBaseUrl;

    final savedApiKey = await loadSavedApiKey();
    final resolvedApiKey = savedApiKey ?? LocalBackendConfig.fallbackApiKey;

    _logDebug(
      'Configuracion resuelta. baseUrl=$resolvedBaseUrl apiKeyConfigured=${resolvedApiKey != null}',
    );
    return BackendConnectionConfig(
      baseUrl: resolvedBaseUrl,
      apiKey: resolvedApiKey,
    );
  }

  static Future<void> saveConnectionConfig({
    required String baseUrl,
    String? apiKey,
  }) async {
    final normalizedBaseUrl = normalizeBaseUrl(baseUrl);
    if (normalizedBaseUrl == null) {
      throw const FormatException('La URL del backend no es valida.');
    }

    final normalizedApiKey = normalizeApiKey(apiKey);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_baseUrlPrefsKey, normalizedBaseUrl);
    if (normalizedApiKey == null) {
      await prefs.remove(_apiKeyPrefsKey);
    } else {
      await prefs.setString(_apiKeyPrefsKey, normalizedApiKey);
    }
    _logDebug(
      'Configuracion guardada. baseUrl=$normalizedBaseUrl apiKeyConfigured=${normalizedApiKey != null}',
    );
  }

  static Future<BackendConnectionConfig>
      restoreDefaultConnectionConfig() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_baseUrlPrefsKey);
    await prefs.remove(_apiKeyPrefsKey);
    final restoredConfig = BackendConnectionConfig(
      baseUrl: normalizeBaseUrl(LocalBackendConfig.fallbackBaseUrl) ??
          LocalBackendConfig.defaultBaseUrl,
      apiKey: LocalBackendConfig.fallbackApiKey,
    );
    _logDebug(
      'Se restauro la configuracion por defecto. baseUrl=${restoredConfig.baseUrl} apiKeyConfigured=${restoredConfig.apiKey != null}',
    );
    return restoredConfig;
  }

  static Future<String> loadResolvedBaseUrl() async {
    return (await loadResolvedConnectionConfig()).baseUrl;
  }

  static Future<void> saveBaseUrl(String value) async {
    final currentConfig = await loadResolvedConnectionConfig();
    await saveConnectionConfig(baseUrl: value, apiKey: currentConfig.apiKey);
  }

  static Future<String> restoreDefaultBaseUrl() async {
    return (await restoreDefaultConnectionConfig()).baseUrl;
  }

  static String? normalizeBaseUrl(String value) {
    final trimmed = value.trim();
    if (trimmed.isEmpty) {
      return null;
    }
    if (!LocalApiService.isValidBaseUrl(trimmed)) {
      return null;
    }
    return trimmed.endsWith('/')
        ? trimmed.substring(0, trimmed.length - 1)
        : trimmed;
  }

  static String? normalizeApiKey(String? value) {
    final trimmed = (value ?? '').trim();
    if (trimmed.isEmpty) {
      return null;
    }
    return trimmed;
  }

  static void _logDebug(String message) {
    if (kDebugMode) {
      debugPrint('[BackendUrlStore] $message');
    }
  }
}
