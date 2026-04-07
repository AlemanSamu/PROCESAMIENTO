import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../models/project_models.dart';
import '../services/backend_url_store.dart';
import '../services/local_api_service.dart';
import '../widgets/empty_state.dart';
import '../widgets/error_state.dart';
import '../widgets/loading_state.dart';
import '../widgets/project_card.dart';
import 'backend_settings_screen.dart';
import 'project_status_screen.dart';

class ProjectHistoryScreen extends StatefulWidget {
  const ProjectHistoryScreen({
    super.key,
    required this.apiService,
    required this.currentBaseUrl,
    required this.currentApiKey,
    required this.onBackendConfigChanged,
  });

  final LocalApiService apiService;
  final String currentBaseUrl;
  final String? currentApiKey;
  final Future<void> Function(BackendConnectionConfig config)
      onBackendConfigChanged;

  @override
  State<ProjectHistoryScreen> createState() => _ProjectHistoryScreenState();
}

class _ProjectHistoryScreenState extends State<ProjectHistoryScreen> {
  final ImagePicker _imagePicker = ImagePicker();

  bool _isLoading = true;
  String? _errorMessage;
  String? _busyProjectId;
  List<ProjectModel> _projects = const [];

  @override
  void initState() {
    super.initState();
    _loadProjects();
  }

  @override
  void didUpdateWidget(covariant ProjectHistoryScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.currentBaseUrl != widget.currentBaseUrl ||
        oldWidget.currentApiKey != widget.currentApiKey) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) {
          _loadProjects();
        }
      });
    }
  }

  Future<void> _loadProjects() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final projects = await widget.apiService.getProjects();
      if (!mounted) {
        return;
      }
      setState(() {
        _projects = projects;
      });
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

  Future<void> _createProject() async {
    final name = await _askProjectName();
    if (name == null) {
      return;
    }

    try {
      await widget.apiService.createProject(name: name);
      if (!mounted) {
        return;
      }
      _showInfo('Proyecto creado');
      await _loadProjects();
    } catch (e) {
      if (!mounted) {
        return;
      }
      _showError(e.toString());
    }
  }

  Future<String?> _askProjectName() async {
    final controller = TextEditingController();
    final result = await showDialog<String>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Nuevo proyecto'),
          content: TextField(
            controller: controller,
            decoration: const InputDecoration(
              hintText: 'Ejemplo: Escaneo sala 1',
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancelar'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(context, controller.text.trim()),
              child: const Text('Crear'),
            ),
          ],
        );
      },
    );
    controller.dispose();

    if (result == null) {
      return null;
    }
    if (result.isEmpty) {
      return 'Proyecto sin nombre';
    }
    return result;
  }

  Future<void> _uploadImages(ProjectModel project) async {
    final selected = await _imagePicker.pickMultiImage(imageQuality: 85);
    if (selected.isEmpty) {
      return;
    }

    setState(() {
      _busyProjectId = project.id;
    });

    try {
      final paths = selected.map((item) => item.path).toList();
      final result = await widget.apiService.uploadImages(
        projectId: project.id,
        imagePaths: paths,
      );
      if (!mounted) {
        return;
      }
      _showInfo('Subidas ${result.uploadedCount} imagenes');
      await _loadProjects();
    } catch (e) {
      if (!mounted) {
        return;
      }
      _showError(e.toString());
    } finally {
      if (!mounted) {
        return;
      }
      setState(() {
        _busyProjectId = null;
      });
    }
  }

  Future<void> _startProcess(ProjectModel project) async {
    setState(() {
      _busyProjectId = project.id;
    });

    try {
      final result =
          await widget.apiService.startProcessing(projectId: project.id);
      if (!mounted) {
        return;
      }
      _showInfo('Procesamiento iniciado con ${result.engine}');
      await _loadProjects();
    } catch (e) {
      if (!mounted) {
        return;
      }
      _showError(e.toString());
    } finally {
      if (!mounted) {
        return;
      }
      setState(() {
        _busyProjectId = null;
      });
    }
  }

  Future<void> _openStatus(ProjectModel project) async {
    await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => ProjectStatusScreen(
          apiService: widget.apiService,
          projectId: project.id,
          projectName: project.name,
        ),
      ),
    );
    if (!mounted) {
      return;
    }
    _loadProjects();
  }

  Future<void> _openBackendSettings() async {
    final updatedConfig =
        await Navigator.of(context).push<BackendConnectionConfig>(
      MaterialPageRoute(
        builder: (_) => BackendSettingsScreen(
          currentConfig: BackendConnectionConfig(
            baseUrl: widget.currentBaseUrl,
            apiKey: widget.currentApiKey,
          ),
        ),
      ),
    );
    if (!mounted || updatedConfig == null) {
      return;
    }
    if (updatedConfig.baseUrl == widget.currentBaseUrl &&
        updatedConfig.apiKey == widget.currentApiKey) {
      return;
    }

    await widget.onBackendConfigChanged(updatedConfig);
    if (!mounted) {
      return;
    }
    _showInfo('Backend actualizado: ${updatedConfig.baseUrl}');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Historial de proyectos'),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(40),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
            child: Row(
              children: [
                const Icon(Icons.dns_outlined, size: 18),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    widget.currentBaseUrl,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                ),
              ],
            ),
          ),
        ),
        actions: [
          IconButton(
            onPressed: _openBackendSettings,
            icon: const Icon(Icons.settings),
            tooltip: 'Configurar backend',
          ),
          IconButton(
            onPressed: _loadProjects,
            icon: const Icon(Icons.refresh),
            tooltip: 'Recargar',
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _createProject,
        icon: const Icon(Icons.add),
        label: const Text('Nuevo proyecto'),
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_isLoading) {
      return const LoadingState(message: 'Cargando proyectos...');
    }

    if (_errorMessage != null) {
      return ErrorState(
        message: _errorMessage!,
        onRetry: _loadProjects,
      );
    }

    if (_projects.isEmpty) {
      return EmptyState(
        title: 'No hay proyectos',
        subtitle: 'Crea tu primer proyecto para comenzar.',
        actionLabel: 'Crear proyecto',
        onAction: _createProject,
      );
    }

    return RefreshIndicator(
      onRefresh: _loadProjects,
      child: ListView.builder(
        itemCount: _projects.length,
        itemBuilder: (context, index) {
          final project = _projects[index];
          return ProjectCard(
            project: project,
            isBusy: _busyProjectId == project.id,
            onUploadImages: () => _uploadImages(project),
            onStartProcessing: () => _startProcess(project),
            onOpenStatus: () => _openStatus(project),
          );
        },
      ),
    );
  }

  void _showInfo(String message) {
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.redAccent,
      ),
    );
  }
}
