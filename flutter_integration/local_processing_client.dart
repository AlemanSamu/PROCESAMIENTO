import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

class LocalProcessingClient {
  LocalProcessingClient({
    required this.baseUrl,
    http.Client? httpClient,
  }) : _httpClient = httpClient ?? http.Client();

  final String baseUrl;
  final http.Client _httpClient;

  Future<Map<String, dynamic>> createProject({String? name}) async {
    final uri = Uri.parse('$baseUrl/projects');
    final response = await _httpClient.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'name': name}),
    );
    return _decodeJson(response);
  }

  Future<Map<String, dynamic>> uploadImages({
    required String projectId,
    required List<String> imagePaths,
  }) async {
    final uri = Uri.parse('$baseUrl/projects/$projectId/images');
    final request = http.MultipartRequest('POST', uri);

    for (final path in imagePaths) {
      request.files.add(await http.MultipartFile.fromPath('files', path));
    }

    final streamed = await _httpClient.send(request);
    final response = await http.Response.fromStream(streamed);
    return _decodeJson(response);
  }

  Future<Map<String, dynamic>> startProcessing({
    required String projectId,
    String outputFormat = 'glb',
  }) async {
    final uri = Uri.parse('$baseUrl/projects/$projectId/process');
    final response = await _httpClient.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'output_format': outputFormat}),
    );
    return _decodeJson(response);
  }

  Future<Map<String, dynamic>> getProjectStatus(String projectId) async {
    final uri = Uri.parse('$baseUrl/projects/$projectId/status');
    final response = await _httpClient.get(uri);
    return _decodeJson(response);
  }

  Uri modelDownloadUri(String projectId) {
    return Uri.parse('$baseUrl/projects/$projectId/model');
  }

  void close() {
    _httpClient.close();
  }

  Map<String, dynamic> _decodeJson(http.Response response) {
    final body = response.body.isEmpty ? '{}' : response.body;
    final decoded = jsonDecode(body) as Map<String, dynamic>;

    if (response.statusCode < 200 || response.statusCode >= 300) {
      final message = decoded['error']?['message']?.toString() ?? 'HTTP error ${response.statusCode}';
      throw HttpException(message);
    }

    return decoded;
  }
}
