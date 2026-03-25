import 'package:http/http.dart' as http;

import '../models/project_models.dart';
import 'local_api_service.dart';

class LocalProcessingApiService {
  LocalProcessingApiService({
    required this.baseUrl,
    http.Client? httpClient,
  }) : _delegate = LocalApiService(baseUrl: baseUrl, client: httpClient);

  final String baseUrl;
  final LocalApiService _delegate;

  Future<List<ProjectItem>> listProjects() {
    return _delegate.getProjects();
  }

  Future<ProjectItem> createProject({String? name}) {
    return _delegate.createProject(name: name);
  }

  Future<ImageUploadResult> uploadImages({
    required String projectId,
    required List<String> imagePaths,
  }) {
    return _delegate.uploadImages(projectId: projectId, imagePaths: imagePaths);
  }

  Future<ProcessStartResult> startProcessing({
    required String projectId,
    OutputFormat outputFormat = OutputFormat.glb,
  }) {
    return _delegate.startProcessing(
      projectId: projectId,
      outputFormat: outputFormat.name,
    );
  }

  Future<ProjectStatusResult> getProjectStatus(String projectId) {
    return _delegate.getProjectStatus(projectId);
  }

  Uri modelUrl(String projectId) {
    return _delegate.getModelUri(projectId);
  }

  void close() {
    _delegate.close();
  }
}
