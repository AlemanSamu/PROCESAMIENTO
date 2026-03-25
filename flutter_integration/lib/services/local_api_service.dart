import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/project_models.dart';

class ApiException implements Exception {
  ApiException(this.message);

  final String message;

  @override
  String toString() => message;
}

class LocalApiService {
  LocalApiService({
    required this.baseUrl,
    http.Client? client,
  }) : _client = client ?? http.Client();

  final String baseUrl;
  final http.Client _client;

  Uri _uri(String path) {
    final normalizedPath = path.startsWith('/') ? path : '/$path';
    return Uri.parse('$baseUrl$normalizedPath');
  }

  Future<List<ProjectModel>> getProjects() async {
    final response = await _client.get(_uri('/projects'));
    final decoded = _decodeBody(response);
    _throwIfError(response.statusCode, decoded);

    if (decoded is! List<dynamic>) {
      throw ApiException('Respuesta invalida para historial de proyectos.');
    }

    return decoded
        .whereType<Map<String, dynamic>>()
        .map(ProjectModel.fromJson)
        .toList();
  }

  Future<ProjectModel> createProject({String? name}) async {
    final response = await _client.post(
      _uri('/projects'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'name': name}),
    );
    return ProjectModel.fromJson(_decodeMap(response));
  }

  Future<ImageUploadResult> uploadImages({
    required String projectId,
    required List<String> imagePaths,
  }) async {
    if (imagePaths.isEmpty) {
      throw ApiException('Selecciona al menos una imagen.');
    }

    final request = http.MultipartRequest('POST', _uri('/projects/$projectId/images'));
    for (final path in imagePaths) {
      request.files.add(await http.MultipartFile.fromPath('files', path));
    }

    final streamed = await _client.send(request);
    final response = await http.Response.fromStream(streamed);
    return ImageUploadResult.fromJson(_decodeMap(response));
  }

  Future<ProcessStartResult> startProcessing({
    required String projectId,
    String outputFormat = 'glb',
  }) async {
    final response = await _client.post(
      _uri('/projects/$projectId/process'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'output_format': outputFormat}),
    );
    return ProcessStartResult.fromJson(_decodeMap(response));
  }

  Future<ProjectModel> getProjectStatus(String projectId) async {
    final response = await _client.get(_uri('/projects/$projectId/status'));
    return ProjectModel.fromJson(_decodeMap(response));
  }

  Uri getModelUri(String projectId) {
    return _uri('/projects/$projectId/model');
  }

  void close() {
    _client.close();
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
    if (response.body.isEmpty) {
      return <String, dynamic>{};
    }
    try {
      return jsonDecode(response.body);
    } catch (_) {
      throw ApiException('No se pudo interpretar la respuesta del backend.');
    }
  }

  void _throwIfError(int statusCode, dynamic decoded) {
    if (statusCode >= 200 && statusCode < 300) {
      return;
    }

    String message = 'Error HTTP $statusCode';
    if (decoded is Map<String, dynamic>) {
      final error = decoded['error'];
      if (error is Map<String, dynamic> && error['message'] != null) {
        message = error['message'].toString();
      }
    }
    throw ApiException(message);
  }
}
