enum ProjectStatus {
  created,
  ready,
  processing,
  completed,
  failed,
  unknown,
}

enum OutputFormat {
  glb,
  obj,
}

ProjectStatus projectStatusFromString(String? raw) {
  switch ((raw ?? '').toLowerCase()) {
    case 'created':
      return ProjectStatus.created;
    case 'ready':
      return ProjectStatus.ready;
    case 'processing':
      return ProjectStatus.processing;
    case 'completed':
      return ProjectStatus.completed;
    case 'failed':
      return ProjectStatus.failed;
    default:
      return ProjectStatus.unknown;
  }
}

extension ProjectStatusText on ProjectStatus {
  String get label {
    switch (this) {
      case ProjectStatus.created:
        return 'Created';
      case ProjectStatus.ready:
        return 'Ready';
      case ProjectStatus.processing:
        return 'Processing';
      case ProjectStatus.completed:
        return 'Completed';
      case ProjectStatus.failed:
        return 'Failed';
      case ProjectStatus.unknown:
        return 'Unknown';
    }
  }

  bool get isTerminal {
    return this == ProjectStatus.completed || this == ProjectStatus.failed;
  }
}

class ProjectModel {
  const ProjectModel({
    required this.id,
    required this.name,
    required this.status,
    required this.imageCount,
    this.outputFormat,
    this.modelFilename,
    this.modelDownloadUrl,
    this.errorMessage,
    this.currentStage,
    this.progress,
    this.message,
    this.metrics,
    this.fallbackUsed = false,
    this.finalModelType,
    this.finalModelPath,
    this.methodUsed,
    this.processingMetadata,
    this.createdAt,
    this.updatedAt,
  });

  final String id;
  final String name;
  final ProjectStatus status;
  final int imageCount;
  final String? outputFormat;
  final String? modelFilename;
  final String? modelDownloadUrl;
  final String? errorMessage;
  final String? currentStage;
  final double? progress;
  final String? message;
  final Map<String, dynamic>? metrics;
  final bool fallbackUsed;
  final String? finalModelType;
  final String? finalModelPath;
  final String? methodUsed;
  final Map<String, dynamic>? processingMetadata;
  final DateTime? createdAt;
  final DateTime? updatedAt;

  int? get meshVertexCount => _toNullableInt(metrics?['mesh_vertex_count']);

  int? get meshFaceCount => _toNullableInt(metrics?['mesh_face_count']);

  Map<String, dynamic>? get sparseFallback =>
      _toStringKeyedMap(processingMetadata?['sparse_fallback']);

  factory ProjectModel.fromJson(Map<String, dynamic> json) {
    return ProjectModel(
      id: (json['id'] ?? json['project_id'] ?? '').toString(),
      name: (json['name'] ?? 'Untitled').toString(),
      status: projectStatusFromString(json['status']?.toString()),
      imageCount: _toInt(json['image_count']),
      outputFormat: json['output_format']?.toString(),
      modelFilename: json['model_filename']?.toString(),
      modelDownloadUrl: json['model_download_url']?.toString(),
      errorMessage: json['error_message']?.toString(),
      currentStage: json['current_stage']?.toString(),
      progress: _toDouble(json['progress']),
      message: json['message']?.toString(),
      metrics: _toStringKeyedMap(json['metrics']),
      fallbackUsed: _toBool(json['fallback_used']),
      finalModelType: json['final_model_type']?.toString(),
      finalModelPath: json['final_model_path']?.toString(),
      methodUsed: json['method_used']?.toString(),
      processingMetadata: _toStringKeyedMap(json['processing_metadata']),
      createdAt: _toDateTime(json['created_at']),
      updatedAt: _toDateTime(json['updated_at']),
    );
  }
}

class ImageUploadResult {
  const ImageUploadResult({
    required this.projectId,
    required this.status,
    required this.uploadedCount,
    required this.totalImages,
  });

  final String projectId;
  final ProjectStatus status;
  final int uploadedCount;
  final int totalImages;

  factory ImageUploadResult.fromJson(Map<String, dynamic> json) {
    return ImageUploadResult(
      projectId: (json['project_id'] ?? '').toString(),
      status: projectStatusFromString(json['status']?.toString()),
      uploadedCount: _toInt(json['uploaded_count']),
      totalImages: _toInt(json['total_images']),
    );
  }
}

class ProcessStartResult {
  const ProcessStartResult({
    required this.projectId,
    required this.status,
    required this.engine,
    required this.message,
  });

  final String projectId;
  final ProjectStatus status;
  final String engine;
  final String message;

  factory ProcessStartResult.fromJson(Map<String, dynamic> json) {
    return ProcessStartResult(
      projectId: (json['project_id'] ?? '').toString(),
      status: projectStatusFromString(json['status']?.toString()),
      engine: (json['engine'] ?? '').toString(),
      message: (json['message'] ?? '').toString(),
    );
  }
}

// Compatibilidad con nombres usados en versiones previas.
typedef ProjectItem = ProjectModel;
typedef ProjectStatusResult = ProjectModel;

int _toInt(dynamic value) {
  if (value is int) {
    return value;
  }
  return int.tryParse(value?.toString() ?? '') ?? 0;
}

int? _toNullableInt(dynamic value) {
  if (value == null) {
    return null;
  }
  if (value is int) {
    return value;
  }
  return int.tryParse(value.toString());
}

double? _toDouble(dynamic value) {
  if (value == null) {
    return null;
  }
  if (value is double) {
    return value;
  }
  if (value is int) {
    return value.toDouble();
  }
  return double.tryParse(value.toString());
}

bool _toBool(dynamic value) {
  if (value is bool) {
    return value;
  }
  final normalized = value?.toString().toLowerCase();
  return normalized == 'true' || normalized == '1' || normalized == 'yes';
}

DateTime? _toDateTime(dynamic value) {
  if (value == null) {
    return null;
  }
  return DateTime.tryParse(value.toString());
}

Map<String, dynamic>? _toStringKeyedMap(dynamic value) {
  if (value is Map<String, dynamic>) {
    return value;
  }
  if (value is Map) {
    return value.map(
      (key, item) => MapEntry(key.toString(), item),
    );
  }
  return null;
}
