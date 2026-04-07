import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../models/project_models.dart';

class ApiException implements Exception {
  ApiException(this.message);

  final String message;

  @override
  String toString() => message;
}

enum HealthCheckStatus {
  connected,
  unauthorized,
  connectionError,
  timeout,
  invalidUrl,
}

class HealthCheckResult {
  const HealthCheckResult({
    required this.status,
    required this.message,
    this.statusCode,
  });

  final HealthCheckStatus status;
  final String message;
  final int? statusCode;

  bool get isConnected => status == HealthCheckStatus.connected;

  factory HealthCheckResult.connected({int? statusCode}) {
    final suffix = statusCode == null ? '' : ' (HTTP $statusCode)';
    return HealthCheckResult(
      status: HealthCheckStatus.connected,
      message: 'Conectado$suffix',
      statusCode: statusCode,
    );
  }

  factory HealthCheckResult.unauthorized({
    String? details,
    int? statusCode,
  }) {
    final suffix = details == null || details.isEmpty ? '' : ': $details';
    return HealthCheckResult(
      status: HealthCheckStatus.unauthorized,
      message: 'API key incorrecta o faltante$suffix',
      statusCode: statusCode,
    );
  }

  factory HealthCheckResult.connectionError({
    String? details,
    int? statusCode,
  }) {
    final suffix = details == null || details.isEmpty ? '' : ': $details';
    return HealthCheckResult(
      status: HealthCheckStatus.connectionError,
      message: 'Error de conexion$suffix',
      statusCode: statusCode,
    );
  }

  factory HealthCheckResult.timeout() {
    return const HealthCheckResult(
      status: HealthCheckStatus.timeout,
      message: 'Timeout',
    );
  }

  factory HealthCheckResult.invalidUrl([String? details]) {
    final suffix = details == null || details.isEmpty ? '' : ': $details';
    return HealthCheckResult(
      status: HealthCheckStatus.invalidUrl,
      message: 'URL invalida$suffix',
    );
  }
}

class ModelViewerAsset {
  const ModelViewerAsset({
    required this.sourceUri,
    required this.localUri,
    required this.localPath,
    required this.detectedType,
    required this.byteCount,
    this.contentType,
  });

  final Uri sourceUri;
  final Uri localUri;
  final String localPath;
  final String detectedType;
  final int byteCount;
  final String? contentType;
}

class LocalApiService {
  LocalApiService({
    required String baseUrl,
    String? apiKey,
    http.Client? client,
  })  : baseUrl = normalizeBaseUrl(baseUrl) ?? baseUrl.trim(),
        apiKey = normalizeApiKey(apiKey),
        _client = client ?? http.Client() {
    _logDebug(
      'Usando backend: ${this.baseUrl} apiKeyConfigured=${this.apiKey != null}',
    );
  }

  static const String _apiKeyHeader = 'X-API-Key';
  static const String _viewerCacheDirectoryName =
      'local_processing_client_models';

  final String baseUrl;
  final String? apiKey;
  final http.Client _client;

  static bool isValidBaseUrl(String value) {
    final trimmed = value.trim();
    if (trimmed.isEmpty) {
      return false;
    }
    if (!trimmed.startsWith('http://') && !trimmed.startsWith('https://')) {
      return false;
    }

    final parsed = Uri.tryParse(trimmed);
    return parsed != null && parsed.hasScheme && parsed.host.isNotEmpty;
  }

  static String? normalizeBaseUrl(String value) {
    final trimmed = value.trim();
    if (!isValidBaseUrl(trimmed)) {
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

  static Future<HealthCheckResult> testConnectionToUrl(
    String baseUrl, {
    String? apiKey,
    Duration timeout = const Duration(seconds: 5),
    http.Client? client,
  }) async {
    final normalizedBaseUrl = normalizeBaseUrl(baseUrl);
    if (normalizedBaseUrl == null) {
      _logDebug('No se puede probar /health: URL invalida "$baseUrl"');
      return HealthCheckResult.invalidUrl(
        'Debe empezar con http:// o https:// y tener un host valido.',
      );
    }

    final normalizedApiKey = normalizeApiKey(apiKey);
    final ownsClient = client == null;
    final activeClient = client ?? http.Client();
    final uri = Uri.parse('$normalizedBaseUrl/health');

    try {
      _logDebug(
        'Probando /health en $uri apiKeyConfigured=${normalizedApiKey != null}',
      );
      final response = await activeClient
          .get(uri, headers: _buildHeaders(apiKey: normalizedApiKey))
          .timeout(timeout);
      final decoded = _decodeResponseBody(response.body);
      _logDebug(
        'Resultado /health ${response.statusCode} desde $normalizedBaseUrl: ${response.body}',
      );

      if (response.statusCode == 401) {
        _logDebug('Respuesta 401 en /health para $normalizedBaseUrl');
        return HealthCheckResult.unauthorized(
          details: _extractErrorMessage(response.statusCode, decoded),
          statusCode: response.statusCode,
        );
      }

      if (response.statusCode >= 200 && response.statusCode < 300) {
        final auth = decoded is Map<String, dynamic> ? decoded['auth'] : null;
        if (auth is Map<String, dynamic>) {
          final required = auth['required'] == true;
          final valid = auth['valid'] != false;
          if (required && !valid) {
            _logDebug(
              'La API key fue rechazada por /health para $normalizedBaseUrl',
            );
            return HealthCheckResult.unauthorized(statusCode: 401);
          }
        }
        return HealthCheckResult.connected(statusCode: response.statusCode);
      }

      return HealthCheckResult.connectionError(
        details: _extractErrorMessage(response.statusCode, decoded),
        statusCode: response.statusCode,
      );
    } on TimeoutException catch (error) {
      _logDebug('Timeout en /health para $normalizedBaseUrl: $error');
      return HealthCheckResult.timeout();
    } on SocketException catch (error) {
      _logDebug('Error de socket en /health para $normalizedBaseUrl: $error');
      return HealthCheckResult.connectionError(details: error.message);
    } on http.ClientException catch (error) {
      _logDebug('ClientException en /health para $normalizedBaseUrl: $error');
      return HealthCheckResult.connectionError(details: error.message);
    } on FormatException catch (error) {
      _logDebug('FormatException en /health para $normalizedBaseUrl: $error');
      return HealthCheckResult.invalidUrl(error.message);
    } catch (error) {
      _logDebug('Fallo inesperado en /health para $normalizedBaseUrl: $error');
      return HealthCheckResult.connectionError(details: error.toString());
    } finally {
      if (ownsClient) {
        activeClient.close();
      }
    }
  }

  Future<HealthCheckResult> testConnection({
    Duration timeout = const Duration(seconds: 5),
  }) {
    return LocalApiService.testConnectionToUrl(
      baseUrl,
      apiKey: apiKey,
      timeout: timeout,
      client: _client,
    );
  }

  Uri _uri(String path) {
    final normalizedPath = path.startsWith('/') ? path : '/$path';
    return Uri.parse('$baseUrl$normalizedPath');
  }

  Future<List<ProjectModel>> getProjects() {
    return _guardRequest('GET /projects', () async {
      final response = await _client.get(
        _uri('/projects'),
        headers: _headers(),
      );
      final decoded = _decodeBody(response);
      _throwIfError(response.statusCode, decoded);

      if (decoded is! List<dynamic>) {
        throw ApiException('Respuesta invalida para historial de proyectos.');
      }

      return decoded
          .whereType<Map<String, dynamic>>()
          .map(ProjectModel.fromJson)
          .toList();
    });
  }

  Future<ProjectModel> createProject({String? name}) {
    return _guardRequest('POST /projects', () async {
      final response = await _client.post(
        _uri('/projects'),
        headers: _headers(jsonContentType: true),
        body: jsonEncode({'name': name}),
      );
      return ProjectModel.fromJson(_decodeMap(response));
    });
  }

  Future<ImageUploadResult> uploadImages({
    required String projectId,
    required List<String> imagePaths,
  }) {
    return _guardRequest('POST /projects/$projectId/images', () async {
      if (imagePaths.isEmpty) {
        throw ApiException('Selecciona al menos una imagen.');
      }

      final request = http.MultipartRequest(
        'POST',
        _uri('/projects/$projectId/images'),
      );
      request.headers.addAll(_headers());
      for (final path in imagePaths) {
        request.files.add(await http.MultipartFile.fromPath('files', path));
      }

      final streamed = await _client.send(request);
      final response = await http.Response.fromStream(streamed);
      return ImageUploadResult.fromJson(_decodeMap(response));
    });
  }

  Future<ProcessStartResult> startProcessing({
    required String projectId,
    String outputFormat = 'glb',
  }) {
    return _guardRequest('POST /projects/$projectId/process', () async {
      final response = await _client.post(
        _uri('/projects/$projectId/process'),
        headers: _headers(jsonContentType: true),
        body: jsonEncode({'output_format': outputFormat}),
      );
      return ProcessStartResult.fromJson(_decodeMap(response));
    });
  }

  Future<ProjectModel> getProjectStatus(String projectId) {
    return _guardRequest('GET /projects/$projectId/status', () async {
      final response = await _client.get(
        _uri('/projects/$projectId/status'),
        headers: _headers(),
      );
      return ProjectModel.fromJson(_decodeMap(response));
    });
  }

  Uri getModelUri(String projectId) {
    return _uri('/projects/$projectId/model');
  }

  Uri resolveModelUri(
    ProjectModel project, {
    bool cacheBust = false,
  }) {
    final configuredDownloadUrl = (project.modelDownloadUrl ?? '').trim();

    Uri resolved;
    if (configuredDownloadUrl.isEmpty) {
      resolved = getModelUri(project.id);
    } else {
      final candidate = Uri.tryParse(configuredDownloadUrl);
      if (candidate != null && candidate.hasScheme) {
        resolved = candidate;
      } else {
        resolved = Uri.parse('$baseUrl/').resolve(configuredDownloadUrl);
      }
    }

    if (!cacheBust) {
      _logDebug('Model URI resuelta para ${project.id}: $resolved');
      return resolved;
    }

    final cacheBustToken =
        project.updatedAt?.millisecondsSinceEpoch.toString() ??
            DateTime.now().millisecondsSinceEpoch.toString();
    final queryParameters = Map<String, String>.from(resolved.queryParameters)
      ..['ts'] = cacheBustToken;

    if ((project.currentStage ?? '').isNotEmpty) {
      queryParameters['stage'] = project.currentStage!;
    }
    if ((project.modelFilename ?? '').isNotEmpty) {
      queryParameters['file'] = project.modelFilename!;
    }

    final cacheBusted = resolved.replace(queryParameters: queryParameters);
    _logDebug('Model URI cache-busted para ${project.id}: $cacheBusted');
    return cacheBusted;
  }

  Future<ModelViewerAsset> downloadModelForViewer({
    required ProjectModel project,
    Duration timeout = const Duration(seconds: 20),
  }) {
    return _guardRequest('GET /projects/${project.id}/model [viewer]',
        () async {
      final sourceUri = resolveModelUri(project, cacheBust: true);
      final response =
          await _client.get(sourceUri, headers: _headers()).timeout(timeout);

      if (response.statusCode < 200 || response.statusCode >= 300) {
        final decoded = _tryDecodeResponseBody(response.body);
        final message = _extractErrorMessage(response.statusCode, decoded);
        throw ApiException(message);
      }

      final bytes = response.bodyBytes;
      final contentType = response.headers['content-type'];
      final detectedType = _detectModelType(
        project: project,
        sourceUri: sourceUri,
        contentType: contentType,
      );

      _logDebug(
        'Descarga del visor project=${project.id} url=$sourceUri type=$detectedType contentType=$contentType bytes=${bytes.length}',
      );

      if (bytes.isEmpty) {
        throw ApiException('El modelo descargado esta vacio.');
      }
      if (detectedType != 'glb') {
        throw ApiException(
          'El visor esperaba un GLB y recibio "$detectedType".',
        );
      }

      final cacheDir = await _ensureViewerCacheDirectory();
      final modelFile = await _writeViewerModelFile(
        cacheDir: cacheDir,
        project: project,
        bytes: bytes,
        extension: detectedType,
      );
      final localUri = Uri.file(modelFile.path);

      _logDebug(
        'Modelo para visor guardado en ${modelFile.path} desde $sourceUri',
      );

      return ModelViewerAsset(
        sourceUri: sourceUri,
        localUri: localUri,
        localPath: modelFile.path,
        detectedType: detectedType,
        byteCount: bytes.length,
        contentType: contentType,
      );
    });
  }

  void close() {
    _client.close();
  }

  Map<String, String> _headers({bool jsonContentType = false}) {
    return _buildHeaders(
      apiKey: apiKey,
      jsonContentType: jsonContentType,
    );
  }

  Future<R> _guardRequest<R>(
    String action,
    Future<R> Function() request,
  ) async {
    _logDebug('$action usando $baseUrl apiKeyConfigured=${apiKey != null}');
    try {
      return await request();
    } on SocketException catch (error) {
      _logDebug('$action fallo por socket: $error');
      throw ApiException('Error de conexion con el backend.');
    } on http.ClientException catch (error) {
      _logDebug('$action fallo por client exception: $error');
      throw ApiException('Error de conexion con el backend.');
    } on TimeoutException catch (error) {
      _logDebug('$action excedio el tiempo de espera: $error');
      throw ApiException('Timeout al conectar con el backend.');
    } on ApiException catch (error) {
      _logDebug('$action fallo: ${error.message}');
      rethrow;
    } catch (error) {
      _logDebug('$action fallo de forma inesperada: $error');
      rethrow;
    }
  }

  Future<Directory> _ensureViewerCacheDirectory() async {
    final cacheDirectory = Directory(
      '${Directory.systemTemp.path}${Platform.pathSeparator}$_viewerCacheDirectoryName',
    );
    if (!cacheDirectory.existsSync()) {
      await cacheDirectory.create(recursive: true);
    }
    return cacheDirectory;
  }

  Future<File> _writeViewerModelFile({
    required Directory cacheDir,
    required ProjectModel project,
    required List<int> bytes,
    required String extension,
  }) async {
    final timestamp = project.updatedAt?.millisecondsSinceEpoch ??
        DateTime.now().millisecondsSinceEpoch;
    final baseFileName = _sanitizeFileName(
      (project.modelFilename ?? '').trim().isNotEmpty
          ? project.modelFilename!
          : '${project.id}_model.$extension',
    );
    final normalizedFileName =
        baseFileName.toLowerCase().endsWith('.$extension')
            ? baseFileName
            : '$baseFileName.$extension';
    final targetFileName = '${project.id}_${timestamp}_$normalizedFileName';

    await _deleteStaleViewerModelsForProject(
      cacheDir,
      keepFileName: targetFileName,
      projectId: project.id,
    );

    final outputFile = File(
      '${cacheDir.path}${Platform.pathSeparator}$targetFileName',
    );
    await outputFile.writeAsBytes(bytes, flush: true);
    return outputFile;
  }

  Future<void> _deleteStaleViewerModelsForProject(
    Directory cacheDir, {
    required String keepFileName,
    required String projectId,
  }) async {
    await for (final entity in cacheDir.list()) {
      if (entity is! File) {
        continue;
      }
      final fileName = entity.uri.pathSegments.isEmpty
          ? entity.path
          : entity.uri.pathSegments.last;
      if (!fileName.startsWith('${projectId}_') || fileName == keepFileName) {
        continue;
      }
      try {
        await entity.delete();
      } catch (error) {
        _logDebug(
          'No se pudo limpiar cache previa del visor para $projectId: $error',
        );
      }
    }
  }

  Map<String, dynamic> _decodeMap(http.Response response) {
    final decoded = _decodeBody(response);
    _throwIfError(response.statusCode, decoded);
    if (decoded is! Map<String, dynamic>) {
      throw ApiException('Respuesta invalida del backend.');
    }
    return decoded;
  }

  dynamic _decodeBody(http.Response response) {
    return _decodeResponseBody(response.body);
  }

  static dynamic _decodeResponseBody(String body) {
    if (body.isEmpty) {
      return <String, dynamic>{};
    }
    try {
      return jsonDecode(body);
    } catch (_) {
      throw ApiException('No se pudo interpretar la respuesta del backend.');
    }
  }

  static dynamic _tryDecodeResponseBody(String body) {
    if (body.isEmpty) {
      return <String, dynamic>{};
    }
    try {
      return jsonDecode(body);
    } catch (_) {
      return null;
    }
  }

  void _throwIfError(int statusCode, dynamic decoded) {
    if (statusCode >= 200 && statusCode < 300) {
      return;
    }

    final message = _extractErrorMessage(statusCode, decoded);
    if (statusCode == 401) {
      _logDebug('Respuesta 401 desde $baseUrl');
    }
    throw ApiException(message);
  }

  static Map<String, String> _buildHeaders({
    String? apiKey,
    bool jsonContentType = false,
  }) {
    final headers = <String, String>{};
    if (jsonContentType) {
      headers['Content-Type'] = 'application/json';
    }
    final normalizedApiKey = normalizeApiKey(apiKey);
    if (normalizedApiKey != null) {
      headers[_apiKeyHeader] = normalizedApiKey;
    }
    return headers;
  }

  static String _extractErrorMessage(int statusCode, dynamic decoded) {
    String message = 'Error HTTP $statusCode';
    if (statusCode == 401) {
      message = 'API key incorrecta o faltante.';
    }
    if (decoded is Map<String, dynamic>) {
      final error = decoded['error'];
      if (error is Map<String, dynamic> && error['message'] != null) {
        message = error['message'].toString();
      }
    }
    return message;
  }

  static String _detectModelType({
    required ProjectModel project,
    required Uri sourceUri,
    String? contentType,
  }) {
    final candidates = <String?>[
      project.finalModelType,
      _contentTypeToModelType(contentType),
      _extensionFromValue(project.modelFilename),
      _extensionFromValue(project.finalModelPath),
      _extensionFromValue(sourceUri.path),
    ];

    for (final candidate in candidates) {
      final normalized = (candidate ?? '').trim().toLowerCase();
      if (normalized == 'glb' || normalized == 'gltf') {
        return normalized;
      }
    }
    return 'unknown';
  }

  static String? _contentTypeToModelType(String? contentType) {
    final normalized = (contentType ?? '').toLowerCase();
    if (normalized.contains('model/gltf-binary')) {
      return 'glb';
    }
    if (normalized.contains('model/gltf+json')) {
      return 'gltf';
    }
    return null;
  }

  static String? _extensionFromValue(String? value) {
    final normalized = (value ?? '').trim();
    if (normalized.isEmpty) {
      return null;
    }
    final withoutQuery = normalized.split('?').first;
    final dotIndex = withoutQuery.lastIndexOf('.');
    if (dotIndex < 0 || dotIndex == withoutQuery.length - 1) {
      return null;
    }
    return withoutQuery.substring(dotIndex + 1).toLowerCase();
  }

  static String _sanitizeFileName(String value) {
    return value.replaceAll(RegExp(r'[^A-Za-z0-9._-]'), '_');
  }

  static void _logDebug(String message) {
    if (kDebugMode) {
      debugPrint('[LocalApiService] $message');
    }
  }
}
