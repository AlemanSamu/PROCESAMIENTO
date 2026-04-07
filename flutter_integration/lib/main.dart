import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import 'screens/project_history_screen.dart';
import 'services/backend_url_store.dart';
import 'services/local_api_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final initialConfig = await BackendUrlStore.loadResolvedConnectionConfig();
  runApp(LocalProcessingClientApp(initialConfig: initialConfig));
}

class LocalProcessingClientApp extends StatefulWidget {
  const LocalProcessingClientApp({
    super.key,
    required this.initialConfig,
  });

  final BackendConnectionConfig initialConfig;

  @override
  State<LocalProcessingClientApp> createState() =>
      _LocalProcessingClientAppState();
}

class _LocalProcessingClientAppState extends State<LocalProcessingClientApp> {
  late LocalApiService _apiService;
  late BackendConnectionConfig _currentConfig;

  @override
  void initState() {
    super.initState();
    _currentConfig = widget.initialConfig;
    _apiService = LocalApiService(
      baseUrl: _currentConfig.baseUrl,
      apiKey: _currentConfig.apiKey,
    );
    _logDebug(
      'Aplicacion iniciada con backend=${_currentConfig.baseUrl} apiKeyConfigured=${_currentConfig.apiKey != null}',
    );
  }

  @override
  void dispose() {
    _apiService.close();
    super.dispose();
  }

  Future<void> _handleBackendConfigChanged(
    BackendConnectionConfig nextConfig,
  ) async {
    if (nextConfig.baseUrl == _currentConfig.baseUrl &&
        nextConfig.apiKey == _currentConfig.apiKey) {
      _logDebug('La configuracion del backend no cambio.');
      return;
    }

    _logDebug(
      'Actualizando backend de ${_currentConfig.baseUrl} a ${nextConfig.baseUrl} apiKeyConfigured=${nextConfig.apiKey != null}',
    );
    final previousApiService = _apiService;
    final nextApiService = LocalApiService(
      baseUrl: nextConfig.baseUrl,
      apiKey: nextConfig.apiKey,
    );

    setState(() {
      _currentConfig = nextConfig;
      _apiService = nextApiService;
    });

    previousApiService.close();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Local 3D Client',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: ProjectHistoryScreen(
        apiService: _apiService,
        currentBaseUrl: _currentConfig.baseUrl,
        currentApiKey: _currentConfig.apiKey,
        onBackendConfigChanged: _handleBackendConfigChanged,
      ),
    );
  }

  void _logDebug(String message) {
    if (kDebugMode) {
      debugPrint('[LocalProcessingClientApp] $message');
    }
  }
}
