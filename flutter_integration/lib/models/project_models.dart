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
    this.engine,
    this.workflowStage,
    this.stageStatus,
    this.profileUsed,
    this.preprocessingSummary,
    this.fallbackReport,
    this.qualityReport,
    this.colmapReport,
    this.artifactPaths,
    this.warnings = const <String>[],
    this.recommendedNextAction,
    this.qualityClassification,
    this.gpuRequested,
    this.gpuUsed,
    this.gpuFallbackToCpu,
    this.gpuErrorMessage,
    this.camerasReconstructed,
    this.points3DCount,
    this.totalProcessingTime,
    this.imagesProcessed,
    this.imagesAccepted,
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
  final String? engine;
  final String? workflowStage;
  final String? stageStatus;
  final String? profileUsed;
  final Map<String, dynamic>? preprocessingSummary;
  final Map<String, dynamic>? fallbackReport;
  final Map<String, dynamic>? qualityReport;
  final Map<String, dynamic>? colmapReport;
  final Map<String, dynamic>? artifactPaths;
  final List<String> warnings;
  final String? recommendedNextAction;
  final String? qualityClassification;
  final bool? gpuRequested;
  final bool? gpuUsed;
  final bool? gpuFallbackToCpu;
  final String? gpuErrorMessage;
  final int? camerasReconstructed;
  final int? points3DCount;
  final double? totalProcessingTime;
  final int? imagesProcessed;
  final int? imagesAccepted;

  int? get meshVertexCount => _toNullableInt(metrics?['mesh_vertex_count']);

  int? get meshFaceCount => _toNullableInt(metrics?['mesh_face_count']);

  Map<String, dynamic>? get sparseFallback =>
      _toStringKeyedMap(processingMetadata?['sparse_fallback']);

  bool get isObjOnly =>
      (finalModelType ?? '').toLowerCase() == 'obj' ||
      (outputFormat ?? '').toLowerCase() == 'obj';

  String get qualityLabel {
    switch ((qualityClassification ?? '').toLowerCase()) {
      case 'success_real':
        return 'success_real';
      case 'success_sparse_only':
        return 'success_sparse_only';
      case 'fallback_completed':
        return 'fallback_completed';
      case 'failed':
        return 'failed';
      default:
        return 'sin_clasificar';
    }
  }

  String get qualityMessage {
    switch ((qualityClassification ?? '').toLowerCase()) {
      case 'success_real':
        return 'Reconstrucción real completada con COLMAP.';
      case 'success_sparse_only':
        return 'Reconstrucción real parcial generada desde nube sparse.';
      case 'fallback_completed':
        return 'Se generó un modelo aproximado porque la reconstrucción real no fue posible.';
      case 'failed':
        return 'No se pudo generar un modelo usable.';
      default:
        return message ?? 'Sin reporte de calidad.';
    }
  }

  factory ProjectModel.fromJson(Map<String, dynamic> json) {
    final processingMetadata = _toStringKeyedMap(json['processing_metadata']);
    final metrics = _toStringKeyedMap(json['metrics']) ??
        _toStringKeyedMap(processingMetadata?['metrics']);
    final artifactPaths = _toStringKeyedMap(json['artifact_paths']) ??
        _toStringKeyedMap(processingMetadata?['artifacts']);
    final preprocessingSummary = _toStringKeyedMap(json['preprocessing_summary']) ??
        _toStringKeyedMap(processingMetadata?['preprocessing']);
    final fallbackReport = _toStringKeyedMap(json['fallback_report']) ??
        _toStringKeyedMap(processingMetadata?['fallback_report']);
    final qualityReport = _toStringKeyedMap(json['quality_report']) ??
        _toStringKeyedMap(processingMetadata?['quality_report']);
    final colmapReport = _toStringKeyedMap(json['colmap_report']) ??
        _toStringKeyedMap(processingMetadata?['colmap_report']);
    final qualityMetrics = _toStringKeyedMap(qualityReport?['metrics']);
    final warnings = _toStringList(json['warnings']) ??
        _toStringList(processingMetadata?['warnings']) ??
        const <String>[];
    final gpuRequested = _toNullableBool(
      json['gpu_requested'] ??
          processingMetadata?['gpu_requested'] ??
          colmapReport?['gpu_requested'],
    );
    final gpuUsed = _toNullableBool(
      json['gpu_used'] ?? processingMetadata?['gpu_used'] ?? colmapReport?['gpu_used'],
    );
    final gpuFallbackToCpu = _toNullableBool(
      json['gpu_fallback_to_cpu'] ??
          processingMetadata?['gpu_fallback_to_cpu'] ??
          colmapReport?['gpu_fallback_to_cpu'],
    );
    final gpuErrorMessage = _toNullableString(
      json['gpu_error_message'] ??
          processingMetadata?['gpu_error_message'] ??
          colmapReport?['gpu_error_message'],
    );
    final camerasReconstructed = _toNullableInt(
      json['cameras_reconstructed'] ??
          qualityMetrics?['cameras_reconstructed'] ??
          metrics?['reconstructed_camera_count'] ??
          processingMetadata?['registered_image_count'],
    );
    final points3DCount = _toNullableInt(
      json['points3D_count'] ??
          json['points_3d_count'] ??
          qualityMetrics?['points_3d_count'] ??
          metrics?['point_3d_count'] ??
          processingMetadata?['point_count'],
    );
    final totalProcessingTime = _toDouble(
      json['total_processing_time'] ??
          qualityMetrics?['total_processing_seconds'] ??
          metrics?['total_processing_seconds'],
    );
    final imagesProcessed = _toNullableInt(
      qualityMetrics?['image_count_processed'] ??
          metrics?['image_count_processed'] ??
          metrics?['image_count_preprocessed'] ??
          metrics?['image_count_selected'],
    );
    final imagesAccepted = _toNullableInt(
      qualityMetrics?['image_count_accepted'] ??
          metrics?['image_count_accepted'] ??
          metrics?['image_count_selected'],
    );
    final qualityClassification = _toNullableString(
      json['quality_classification'] ??
          qualityReport?['quality_classification'] ??
          processingMetadata?['quality_classification'],
    );
    final profileUsed = _toNullableString(
      qualityReport?['profile'] ?? processingMetadata?['profile'] ?? json['profile'],
    );

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
      metrics: metrics,
      fallbackUsed: _toBool(json['fallback_used'] ?? processingMetadata?['fallback_used']),
      finalModelType: json['final_model_type']?.toString(),
      finalModelPath: json['final_model_path']?.toString(),
      methodUsed: json['method_used']?.toString(),
      processingMetadata: processingMetadata,
      createdAt: _toDateTime(json['created_at']),
      updatedAt: _toDateTime(json['updated_at']),
      engine: _toNullableString(json['engine'] ?? processingMetadata?['engine']),
      workflowStage:
          _toNullableString(json['workflow_stage'] ?? processingMetadata?['workflow_stage']),
      stageStatus:
          _toNullableString(json['stage_status'] ?? processingMetadata?['stage_status']),
      profileUsed: profileUsed,
      preprocessingSummary: preprocessingSummary,
      fallbackReport: fallbackReport,
      qualityReport: qualityReport,
      colmapReport: colmapReport,
      artifactPaths: artifactPaths,
      warnings: warnings,
      recommendedNextAction: _toNullableString(json['recommended_next_action']),
      qualityClassification: qualityClassification,
      gpuRequested: gpuRequested,
      gpuUsed: gpuUsed,
      gpuFallbackToCpu: gpuFallbackToCpu,
      gpuErrorMessage: gpuErrorMessage,
      camerasReconstructed: camerasReconstructed,
      points3DCount: points3DCount,
      totalProcessingTime: totalProcessingTime,
      imagesProcessed: imagesProcessed,
      imagesAccepted: imagesAccepted,
    );
  }

  ProjectModel mergeWith(ProjectModel other) {
    return ProjectModel(
      id: id.isNotEmpty ? id : other.id,
      name: name.isNotEmpty ? name : other.name,
      status: other.status != ProjectStatus.unknown ? other.status : status,
      imageCount: imageCount > 0 ? imageCount : other.imageCount,
      outputFormat: outputFormat ?? other.outputFormat,
      modelFilename: modelFilename ?? other.modelFilename,
      modelDownloadUrl: modelDownloadUrl ?? other.modelDownloadUrl,
      errorMessage: errorMessage ?? other.errorMessage,
      currentStage: currentStage ?? other.currentStage,
      progress: progress ?? other.progress,
      message: message ?? other.message,
      metrics: _mergeMaps(metrics, other.metrics),
      fallbackUsed: fallbackUsed || other.fallbackUsed,
      finalModelType: finalModelType ?? other.finalModelType,
      finalModelPath: finalModelPath ?? other.finalModelPath,
      methodUsed: methodUsed ?? other.methodUsed,
      processingMetadata: _mergeMaps(processingMetadata, other.processingMetadata),
      createdAt: createdAt ?? other.createdAt,
      updatedAt: updatedAt ?? other.updatedAt,
      engine: engine ?? other.engine,
      workflowStage: workflowStage ?? other.workflowStage,
      stageStatus: stageStatus ?? other.stageStatus,
      profileUsed: profileUsed ?? other.profileUsed,
      preprocessingSummary: _mergeMaps(preprocessingSummary, other.preprocessingSummary),
      fallbackReport: _mergeMaps(fallbackReport, other.fallbackReport),
      qualityReport: _mergeMaps(qualityReport, other.qualityReport),
      colmapReport: _mergeMaps(colmapReport, other.colmapReport),
      artifactPaths: _mergeMaps(artifactPaths, other.artifactPaths),
      warnings: warnings.isNotEmpty ? warnings : other.warnings,
      recommendedNextAction: recommendedNextAction ?? other.recommendedNextAction,
      qualityClassification: qualityClassification ?? other.qualityClassification,
      gpuRequested: gpuRequested ?? other.gpuRequested,
      gpuUsed: gpuUsed ?? other.gpuUsed,
      gpuFallbackToCpu: gpuFallbackToCpu ?? other.gpuFallbackToCpu,
      gpuErrorMessage: gpuErrorMessage ?? other.gpuErrorMessage,
      camerasReconstructed: camerasReconstructed ?? other.camerasReconstructed,
      points3DCount: points3DCount ?? other.points3DCount,
      totalProcessingTime: totalProcessingTime ?? other.totalProcessingTime,
      imagesProcessed: imagesProcessed ?? other.imagesProcessed,
      imagesAccepted: imagesAccepted ?? other.imagesAccepted,
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

bool? _toNullableBool(dynamic value) {
  if (value == null) {
    return null;
  }
  return _toBool(value);
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

List<String>? _toStringList(dynamic value) {
  if (value is List) {
    return value.map((item) => item.toString()).toList();
  }
  return null;
}

String? _toNullableString(dynamic value) {
  if (value == null) {
    return null;
  }
  final text = value.toString().trim();
  if (text.isEmpty || text.toLowerCase() == 'null') {
    return null;
  }
  return text;
}

Map<String, dynamic>? _mergeMaps(
  Map<String, dynamic>? first,
  Map<String, dynamic>? second,
) {
  if (first == null && second == null) {
    return null;
  }
  final merged = <String, dynamic>{};
  if (second != null) {
    merged.addAll(second);
  }
  if (first != null) {
    merged.addAll(first);
  }
  return merged;
}
