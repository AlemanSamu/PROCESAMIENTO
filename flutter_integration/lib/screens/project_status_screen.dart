import 'dart:async';

import 'package:flutter/material.dart';

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
  ProjectModel? _project;
  bool _isLoading = true;
  String? _errorMessage;

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
      final project =
          await widget.apiService.getProjectStatus(widget.projectId);
      if (!mounted) {
        return;
      }
      setState(() {
        _project = project;
        _errorMessage = null;
      });

      if (project.status.isTerminal) {
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
    if (_isLoading && _project == null) {
      return const Scaffold(
        body: LoadingState(message: 'Consultando estado...'),
      );
    }

    if (_errorMessage != null && _project == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Estado del proyecto')),
        body: ErrorState(
          message: _errorMessage!,
          onRetry: () => _loadStatus(initialLoad: true),
        ),
      );
    }

    final project = _project;
    if (project == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Estado del proyecto')),
        body: ErrorState(
          message: 'No se pudo cargar el estado.',
          onRetry: () => _loadStatus(initialLoad: true),
        ),
      );
    }

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
                    Text(
                      'Proyecto: ${project.name}',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                    const SizedBox(height: 10),
                    _detailLine('Estado', project.status.label),
                    _detailLine(
                      'Etapa actual',
                      project.currentStage ?? 'sin reporte',
                    ),
                    _detailLine('Imagenes', '${project.imageCount}'),
                    if (project.progress != null)
                      _detailLine(
                        'Progreso',
                        '${(project.progress! * 100).toStringAsFixed(0)}%',
                      ),
                    if (project.outputFormat != null)
                      _detailLine('Formato salida', project.outputFormat!),
                    if (project.message != null && project.message!.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.only(top: 8),
                        child: Text(project.message!),
                      ),
                    if (project.errorMessage != null &&
                        project.errorMessage!.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.only(top: 8),
                        child: Text(
                          project.errorMessage!,
                          style: const TextStyle(color: Colors.redAccent),
                        ),
                      ),
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
                      'Artefacto final esperado',
                      style: Theme.of(context).textTheme.titleSmall,
                    ),
                    const SizedBox(height: 10),
                    _detailLine(
                      'Fallback sparse',
                      project.fallbackUsed ? 'si' : 'no',
                    ),
                    _detailLine(
                      'Metodo mesh',
                      project.methodUsed ??
                          project.sparseFallback?['mesh_method']?.toString() ??
                          'sin reporte',
                    ),
                    _detailLine(
                      'Tipo modelo',
                      project.finalModelType ?? 'sin reporte',
                    ),
                    _detailLine(
                      'Vertices mesh',
                      project.meshVertexCount?.toString() ?? 'sin reporte',
                    ),
                    _detailLine(
                      'Caras mesh',
                      project.meshFaceCount?.toString() ?? 'sin reporte',
                    ),
                    if (project.finalModelPath != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 8),
                        child: SelectableText(
                          'Archivo final: ${project.finalModelPath}',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                      ),
                    if (project.modelDownloadUrl != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 8),
                        child: SelectableText(
                          'Endpoint del modelo: ${project.modelDownloadUrl}',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                      ),
                    if ((project.meshFaceCount ?? 0) > 0 &&
                        (project.meshFaceCount ?? 0) <= 60)
                      Padding(
                        padding: const EdgeInsets.only(top: 8),
                        child: Text(
                          'La malla final es bastante simple. Si el backend uso convex_hull o bounding_box, el volumen puede verse correcto pero muy basico.',
                          style: TextStyle(
                            color: Theme.of(context).colorScheme.secondary,
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ),
            if (_errorMessage != null)
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
            const SizedBox(height: 12),
            if (project.status == ProjectStatus.completed)
              FilledButton.icon(
                onPressed: _openModelViewer,
                icon: const Icon(Icons.view_in_ar),
                label: const Text('Abrir visor 3D'),
              ),
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
