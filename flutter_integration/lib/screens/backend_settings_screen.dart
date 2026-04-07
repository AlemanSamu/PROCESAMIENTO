import 'package:flutter/material.dart';

import '../config/local_backend_config.dart';
import '../services/backend_url_store.dart';
import '../services/local_api_service.dart';

class BackendSettingsScreen extends StatefulWidget {
  const BackendSettingsScreen({
    super.key,
    required this.currentConfig,
  });

  final BackendConnectionConfig currentConfig;

  @override
  State<BackendSettingsScreen> createState() => _BackendSettingsScreenState();
}

class _BackendSettingsScreenState extends State<BackendSettingsScreen> {
  final GlobalKey<FormState> _formKey = GlobalKey<FormState>();
  late final TextEditingController _urlController;
  late final TextEditingController _apiKeyController;

  HealthCheckResult? _healthResult;
  bool _isTestingConnection = false;
  bool _isSaving = false;
  bool _obscureApiKey = true;

  @override
  void initState() {
    super.initState();
    _urlController = TextEditingController(text: widget.currentConfig.baseUrl);
    _apiKeyController =
        TextEditingController(text: widget.currentConfig.apiKey ?? '');
  }

  @override
  void dispose() {
    _urlController.dispose();
    _apiKeyController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Configuracion del backend'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'URL actual del backend',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 8),
                    SelectableText(widget.currentConfig.baseUrl),
                    const SizedBox(height: 12),
                    Text(
                      'API key actual: ${_describeApiKey(widget.currentConfig.apiKey)}',
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'Valor por defecto actual: ${LocalBackendConfig.fallbackBaseUrl}',
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Prioridad: valores guardados > dart-define > valor por defecto.',
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            Form(
              key: _formKey,
              child: Column(
                children: [
                  TextFormField(
                    controller: _urlController,
                    keyboardType: TextInputType.url,
                    textInputAction: TextInputAction.next,
                    decoration: const InputDecoration(
                      labelText: 'URL base del backend',
                      hintText: 'http://10.221.168.227:8000',
                      border: OutlineInputBorder(),
                      helperText: 'Ejemplo: http://NOMBRE-PC:8000',
                    ),
                    validator: _validateUrl,
                    onChanged: (_) => _clearHealthResult(),
                  ),
                  const SizedBox(height: 16),
                  TextFormField(
                    controller: _apiKeyController,
                    obscureText: _obscureApiKey,
                    textInputAction: TextInputAction.done,
                    decoration: InputDecoration(
                      labelText: 'API key',
                      hintText: 'Ingresa la API key del backend local',
                      border: const OutlineInputBorder(),
                      helperText: 'Se enviara en el header X-API-Key.',
                      suffixIcon: IconButton(
                        onPressed: () {
                          setState(() {
                            _obscureApiKey = !_obscureApiKey;
                          });
                        },
                        icon: Icon(
                          _obscureApiKey
                              ? Icons.visibility
                              : Icons.visibility_off,
                        ),
                      ),
                    ),
                    onChanged: (_) => _clearHealthResult(),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            Wrap(
              spacing: 12,
              runSpacing: 12,
              children: [
                OutlinedButton.icon(
                  onPressed: _isTestingConnection || _isSaving
                      ? null
                      : _testConnection,
                  icon: _isTestingConnection
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.health_and_safety_outlined),
                  label: const Text('Probar conexion'),
                ),
                FilledButton.icon(
                  onPressed: _isSaving ? null : _saveConfig,
                  icon: _isSaving
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.save_outlined),
                  label: const Text('Guardar'),
                ),
                TextButton.icon(
                  onPressed: _isSaving ? null : _restoreDefault,
                  icon: const Icon(Icons.restore),
                  label: const Text('Restaurar valor por defecto'),
                ),
              ],
            ),
            if (_healthResult != null) ...[
              const SizedBox(height: 16),
              _HealthResultCard(result: _healthResult!),
            ],
          ],
        ),
      ),
    );
  }

  String? _validateUrl(String? value) {
    final trimmed = value?.trim() ?? '';
    if (trimmed.isEmpty) {
      return 'La URL no puede estar vacia.';
    }
    if (!trimmed.startsWith('http://') && !trimmed.startsWith('https://')) {
      return 'La URL debe empezar con http:// o https://.';
    }
    if (!LocalApiService.isValidBaseUrl(trimmed)) {
      return 'Ingresa una URL valida.';
    }
    return null;
  }

  Future<void> _testConnection() async {
    FocusScope.of(context).unfocus();
    final validationMessage = _validateUrl(_urlController.text);
    if (validationMessage != null) {
      setState(() {
        _healthResult = HealthCheckResult.invalidUrl(validationMessage);
      });
      return;
    }

    setState(() {
      _isTestingConnection = true;
      _healthResult = null;
    });

    final result = await LocalApiService.testConnectionToUrl(
      _urlController.text.trim(),
      apiKey: _apiKeyController.text.trim(),
    );
    if (!mounted) {
      return;
    }
    setState(() {
      _isTestingConnection = false;
      _healthResult = result;
    });
  }

  Future<void> _saveConfig() async {
    FocusScope.of(context).unfocus();
    final formState = _formKey.currentState;
    if (formState == null || !formState.validate()) {
      return;
    }

    final normalizedBaseUrl =
        BackendUrlStore.normalizeBaseUrl(_urlController.text);
    if (normalizedBaseUrl == null) {
      setState(() {
        _healthResult = HealthCheckResult.invalidUrl(
          'La URL no se pudo normalizar correctamente.',
        );
      });
      return;
    }

    setState(() {
      _isSaving = true;
    });

    try {
      final normalizedApiKey =
          BackendUrlStore.normalizeApiKey(_apiKeyController.text);
      await BackendUrlStore.saveConnectionConfig(
        baseUrl: normalizedBaseUrl,
        apiKey: normalizedApiKey,
      );
      if (!mounted) {
        return;
      }
      Navigator.of(context).pop(
        BackendConnectionConfig(
          baseUrl: normalizedBaseUrl,
          apiKey: normalizedApiKey,
        ),
      );
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _isSaving = false;
        _healthResult =
            HealthCheckResult.connectionError(details: error.toString());
      });
    }
  }

  Future<void> _restoreDefault() async {
    FocusScope.of(context).unfocus();
    setState(() {
      _isSaving = true;
    });

    try {
      final restoredConfig =
          await BackendUrlStore.restoreDefaultConnectionConfig();
      if (!mounted) {
        return;
      }
      Navigator.of(context).pop(restoredConfig);
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _isSaving = false;
        _healthResult =
            HealthCheckResult.connectionError(details: error.toString());
      });
    }
  }

  void _clearHealthResult() {
    if (_healthResult != null) {
      setState(() {
        _healthResult = null;
      });
    }
  }

  String _describeApiKey(String? apiKey) {
    if (apiKey == null || apiKey.isEmpty) {
      return 'No configurada';
    }
    return 'Configurada';
  }
}

class _HealthResultCard extends StatelessWidget {
  const _HealthResultCard({required this.result});

  final HealthCheckResult result;

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final accentColor = switch (result.status) {
      HealthCheckStatus.connected => Colors.green,
      HealthCheckStatus.unauthorized => Colors.redAccent,
      HealthCheckStatus.timeout => Colors.orange,
      HealthCheckStatus.invalidUrl => Colors.amber.shade800,
      HealthCheckStatus.connectionError => colorScheme.error,
    };
    final backgroundColor = switch (result.status) {
      HealthCheckStatus.connected => Colors.green.shade50,
      HealthCheckStatus.unauthorized => Colors.red.shade50,
      HealthCheckStatus.timeout => Colors.orange.shade50,
      HealthCheckStatus.invalidUrl => Colors.amber.shade50,
      HealthCheckStatus.connectionError => colorScheme.errorContainer,
    };
    final icon = switch (result.status) {
      HealthCheckStatus.connected => Icons.check_circle_outline,
      HealthCheckStatus.unauthorized => Icons.lock_outline,
      HealthCheckStatus.timeout => Icons.timer_off_outlined,
      HealthCheckStatus.invalidUrl => Icons.link_off,
      HealthCheckStatus.connectionError => Icons.error_outline,
    };

    return Card(
      color: backgroundColor,
      child: ListTile(
        leading: Icon(icon, color: accentColor),
        title: Text(result.message),
        subtitle: Text(
          switch (result.status) {
            HealthCheckStatus.connected =>
              'El backend respondio correctamente en /health.',
            HealthCheckStatus.unauthorized =>
              'El backend es alcanzable, pero la API key es incorrecta o falta.',
            HealthCheckStatus.timeout =>
              'El backend no respondio dentro del tiempo esperado.',
            HealthCheckStatus.invalidUrl =>
              'Corrige la URL antes de guardarla o probarla.',
            HealthCheckStatus.connectionError =>
              'No fue posible establecer la conexion con el backend.',
          },
        ),
      ),
    );
  }
}
