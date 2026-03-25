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
      final project = await widget.apiService.getProjectStatus(widget.projectId);
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
                    Text('Proyecto: ${project.name}', style: Theme.of(context).textTheme.titleMedium),
                    const SizedBox(height: 10),
                    Text('Estado: ${project.status.label}'),
                    Text('Imagenes: ${project.imageCount}'),
                    if (project.outputFormat != null) Text('Formato salida: ${project.outputFormat}'),
                    if (project.errorMessage != null && project.errorMessage!.isNotEmpty)
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
            if (_errorMessage != null)
              Card(
                color: Colors.red.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Text(_errorMessage!, style: const TextStyle(color: Colors.redAccent)),
                ),
              ),
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

  void _openModelViewer() {
    final modelUrl = widget.apiService.getModelUri(widget.projectId).toString();
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => ModelViewerScreen(
          modelUrl: modelUrl,
          projectName: widget.projectName,
        ),
      ),
    );
  }
}
