import 'package:flutter/material.dart';

import 'config/local_backend_config.dart';
import 'screens/project_history_screen.dart';
import 'services/local_api_service.dart';

void main() {
  runApp(const LocalProcessingClientApp());
}

class LocalProcessingClientApp extends StatefulWidget {
  const LocalProcessingClientApp({super.key});

  @override
  State<LocalProcessingClientApp> createState() => _LocalProcessingClientAppState();
}

class _LocalProcessingClientAppState extends State<LocalProcessingClientApp> {
  late final LocalApiService _apiService;

  @override
  void initState() {
    super.initState();
    _apiService = LocalApiService(baseUrl: LocalBackendConfig.baseUrl);
  }

  @override
  void dispose() {
    _apiService.close();
    super.dispose();
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
      home: ProjectHistoryScreen(apiService: _apiService),
    );
  }
}
