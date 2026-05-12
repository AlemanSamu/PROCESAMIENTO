import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../models/project_models.dart';
import '../services/local_api_service.dart';
import '../widgets/error_state.dart';
import '../widgets/loading_state.dart';
import 'model_viewer_screen.dart';

class ProjectStatusScreen extends StatefulWidget {
  const ProjectStatusScreen({
    super.key,
    required this.apiService,
    required this.projectId,
    required this.projectName,
  });

  final LocalApiService apiService;
  final String projectId;
  final String projectName;

  @override
  State<ProjectStatusScreen> createState() => _ProjectStatusScreenState();
}

class _ProjectStatusScreenState extends State<ProjectStatusScreen> {
  Timer? _pollTimer;
  ProjectModel? _statusProject;
  ProjectModel? _resultProject;
  bool _isLoading = true;
  bool _showTechnicalDetails = false;
  String? _errorMessage;

  ProjectModel? get _project {
    final status = _statusProject;
    final result = _resultProject;
    if (status == null && result == null) {
      return null;
    }
    if (status == null) {
      return result;
    }
    if (result == null) {
      return status;
    }
    return status.mergeWith(result);
  }

  @override
  void initState() {
    super.initState();
    _loadStatus(initialLoad: true);
    _pollTimer = Timer.periodic(
      const Duration(seconds: 3),
      (_) => _loadStatus(),
    );
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    super.dispose();
  }

  Future<void> _loadStatus({bool initialLoad = false}) async {
    if (initialLoad) {
      setState(() {
        _isLoading = true;
        _errorMessage = null;
      });
    }

    try {
      final statusProject = await widget.apiService.getProjectStatus(widget.projectId);
      ProjectModel? resultProject;
      try {
        resultProject = await widget.apiService.getProjectResult(widget.projectId);
      } catch (_) {
        resultProject = null;
      }

      if (!mounted) {
        return;
      }
      setState(() {
        _statusProject = statusProject;
        _resultProject = resultProject;
        _errorMessage = null;
      });

      if (statusProject.status.isTerminal) {
        _pollTimer?.cancel();
      }
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() {
        _errorMessage = e.toString();
      });
    } finally {
      if (!mounted) {
        return;
      }
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final project = _project;

    if (_isLoading && project == null) {
      return const Scaffold(
        body: LoadingState(message: 'Consultando estado...'),
      );
    }

    if (_errorMessage != null && project == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Estado del proyecto')),
        body: ErrorState(
          message: _errorMessage!,
          onRetry: () => _loadStatus(initialLoad: true),
        ),
      );
    }

    if (project == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Estado del proyecto')),
        body: ErrorState(
          message: 'No se pudo cargar el estado.',
          onRetry: () => _loadStatus(initialLoad: true),
        ),
      );
    }

    final classificationColor = _classificationColor(project.qualityClassification, context);
    final canOpenViewer = project.status == ProjectStatus.completed && !project.isObjOnly;
    final shouldShowObjNote = project.status == ProjectStatus.completed && project.isObjOnly;
    final retryActionLabel = _retryActionLabel(project.recommendedNextAction);

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.projectName),
        actions: [
          IconButton(
            onPressed: () => _loadStatus(initialLoad: true),
            icon: const Icon(Icons.refresh),
          ),
        ],
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
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            'Estado general',
                            style: Theme.of(context).textTheme.titleMedium,
                          ),
                        ),
                        Chip(
                          label: Text(project.qualityLabel),
                          backgroundColor: classificationColor.withValues(alpha: 0.14),
                          side: BorderSide(color: classificationColor),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    _detailLine('Estado', project.status.label),
                    _detailLine('Etapa actual', project.currentStage ?? 'sin reporte'),
                    _detailLine('Workflow', project.workflowStage ?? 'sin reporte'),
                    _detailLine(
                      'Motor usado',
                      project.fallbackUsed
                          ? 'fallback'
                          : (project.engine ?? 'sin reporte'),
                    ),
                    if (project.fallbackUsed && (project.engine ?? '').isNotEmpty)
                      _detailLine('Motor intentado', project.engine!),
                    _detailLine('Perfil', project.profileUsed ?? 'sin reporte'),
                    _detailLine('GPU usada', _formatBool(project.gpuUsed)),
                    _detailLine('Fallback', project.fallbackUsed ? 'sí' : 'no'),
                    _detailLine('Clasificación', project.qualityLabel),
                    if (project.progress != null)
                      _detailLine('Progreso', '${(project.progress! * 100).toStringAsFixed(0)}%'),
                    if (project.outputFormat != null)
                      _detailLine('Formato salida', project.outputFormat!.toUpperCase()),
                    const SizedBox(height: 8),
                    Text(project.qualityMessage),
                    if ((project.message ?? '').isNotEmpty) ...[
                      const SizedBox(height: 8),
                      Text(project.message!),
                    ],
                    if (project.errorMessage != null && project.errorMessage!.isNotEmpty) ...[
                      const SizedBox(height: 8),
                      Text(
                        project.errorMessage!,
                        style: const TextStyle(color: Colors.redAccent),
                      ),
                    ],
                    if (project.status == ProjectStatus.processing) ...[
                      const SizedBox(height: 12),
                      const LinearProgressIndicator(),
                    ],
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Resultado',
                      style: Theme.of(context).textTheme.titleSmall,
                    ),
                    const SizedBox(height: 10),
                    _detailLine('Tipo modelo', project.finalModelType ?? 'sin reporte'),
                    _detailLine('Ruta final', project.finalModelPath ?? 'sin reporte'),
                    _detailLine('Cámaras reconstruidas', _formatInt(project.camerasReconstructed)),
                    _detailLine('Puntos 3D', _formatInt(project.points3DCount)),
                    _detailLine('Tiempo total (s)', _formatDouble(project.totalProcessingTime)),
                    if ((project.recommendedNextAction ?? '').isNotEmpty) ...[
                      const SizedBox(height: 8),
                      Text('Recomendación: ${project.recommendedNextAction}'),
                    ],
                    if (shouldShowObjNote) ...[
                      const SizedBox(height: 8),
                      Text(
                        'El proyecto terminó en OBJ. El visor principal está optimizado para GLB.',
                        style: TextStyle(color: Theme.of(context).colorScheme.secondary),
                      ),
                      const SizedBox(height: 8),
                      if ((project.modelDownloadUrl ?? '').isNotEmpty)
                        SelectableText('Descarga OBJ: ${project.modelDownloadUrl}'),
                      if ((project.finalModelPath ?? '').isNotEmpty)
                        SelectableText('Ruta local OBJ: ${project.finalModelPath}'),
                    ],
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 10,
              runSpacing: 10,
              children: [
                if (canOpenViewer)
                  FilledButton.icon(
                    onPressed: _openModelViewer,
                    icon: const Icon(Icons.view_in_ar),
                    label: const Text('Abrir visor 3D'),
                  ),
                OutlinedButton.icon(
                  onPressed: () {
                    setState(() {
                      _showTechnicalDetails = !_showTechnicalDetails;
                    });
                  },
                  icon: const Icon(Icons.analytics_outlined),
                  label: Text(_showTechnicalDetails ? 'Ocultar detalles técnicos' : 'Ver detalles técnicos'),
                ),
                if (retryActionLabel != null)
                  TextButton.icon(
                    onPressed: () => _showRetryRecommendation(project),
                    icon: const Icon(Icons.replay),
                    label: Text(retryActionLabel),
                  ),
              ],
            ),
            if (_showTechnicalDetails) ...[
              const SizedBox(height: 12),
              _buildTechnicalDetailsCard(project),
            ],
            if (_errorMessage != null) ...[
              const SizedBox(height: 12),
              Card(
                color: Colors.red.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Text(
                    _errorMessage!,
                    style: const TextStyle(color: Colors.redAccent),
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildTechnicalDetailsCard(ProjectModel project) {
    final warnings = project.warnings;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Detalles técnicos',
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: 10),
            _detailLine('Imágenes procesadas', _formatInt(project.imagesProcessed)),
            _detailLine('Imágenes aceptadas', _formatInt(project.imagesAccepted)),
            _detailLine('Cámaras reconstruidas', _formatInt(project.camerasReconstructed)),
            _detailLine('Puntos 3D', _formatInt(project.points3DCount)),
            _detailLine('Tiempo total (s)', _formatDouble(project.totalProcessingTime)),
            _detailLine('Perfil', project.profileUsed ?? 'sin reporte'),
            _detailLine('GPU solicitada', _formatBool(project.gpuRequested)),
            _detailLine('GPU usada', _formatBool(project.gpuUsed)),
            _detailLine('GPU fallback a CPU', _formatBool(project.gpuFallbackToCpu)),
            if ((project.gpuErrorMessage ?? '').isNotEmpty)
              _detailLine('Error GPU', project.gpuErrorMessage!),
            _detailLine(
              'Reportes',
              _summarizeReportPaths(project),
            ),
            if (warnings.isNotEmpty) ...[
              const SizedBox(height: 8),
              const Text('Advertencias:'),
              const SizedBox(height: 4),
              ...warnings.map((item) => Text('• $item')),
            ],
          ],
        ),
      ),
    );
  }

  Widget _detailLine(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(top: 4),
      child: Text('$label: $value'),
    );
  }

  String _formatBool(bool? value) {
    if (value == null) {
      return 'sin reporte';
    }
    return value ? 'sí' : 'no';
  }

  String _formatInt(int? value) {
    if (value == null) {
      return 'sin reporte';
    }
    return value.toString();
  }

  String _formatDouble(double? value) {
    if (value == null) {
      return 'sin reporte';
    }
    return value.toStringAsFixed(2);
  }

  String _summarizeReportPaths(ProjectModel project) {
    final artifacts = project.artifactPaths ?? const <String, dynamic>{};
    final reportKeys = <String>[
      'preprocessing_manifest',
      'fallback_report',
      'quality_report',
      'colmap_report',
      'execution_report',
      'technical_evidence_report',
    ];
    final values = <String>[];
    for (final key in reportKeys) {
      final raw = artifacts[key];
      if (raw == null) {
        continue;
      }
      final text = raw.toString().trim();
      if (text.isNotEmpty) {
        values.add('$key=$text');
      }
    }
    if (values.isEmpty) {
      return 'sin reporte';
    }
    return values.join(' | ');
  }

  Color _classificationColor(String? classification, BuildContext context) {
    switch ((classification ?? '').toLowerCase()) {
      case 'success_real':
        return Colors.green;
      case 'success_sparse_only':
        return Colors.blue;
      case 'fallback_completed':
        return Colors.orange;
      case 'failed':
        return Colors.red;
      default:
        return Theme.of(context).colorScheme.outline;
    }
  }

  String? _retryActionLabel(String? recommendation) {
    final text = (recommendation ?? '').toLowerCase();
    if (text.contains('quality')) {
      return 'Reintentar con perfil quality';
    }
    if (text.contains('foto') || text.contains('recaptur')) {
      return 'Reintentar con más fotos';
    }
    if (text.isNotEmpty) {
      return 'Ver recomendación';
    }
    return null;
  }

  Future<void> _showRetryRecommendation(ProjectModel project) async {
    final recommendation = project.recommendedNextAction ?? 'Sin recomendación específica.';
    await showDialog<void>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Recomendación del backend'),
          content: Text(recommendation),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Cerrar'),
            ),
            if (project.profileUsed != null)
              TextButton(
                onPressed: () async {
                  await Clipboard.setData(ClipboardData(text: recommendation));
                  if (mounted) {
                    Navigator.of(context).pop();
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Recomendación copiada.')),
                    );
                  }
                },
                child: const Text('Copiar'),
              ),
          ],
        );
      },
    );
  }

  void _openModelViewer() {
    final project = _project;
    if (project == null) {
      return;
    }
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => ModelViewerScreen(
          apiService: widget.apiService,
          project: project,
        ),
      ),
    );
  }
}
