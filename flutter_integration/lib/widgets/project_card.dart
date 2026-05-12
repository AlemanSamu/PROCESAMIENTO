import 'package:flutter/material.dart';

import '../models/project_models.dart';

class ProjectCard extends StatelessWidget {
  const ProjectCard({
    super.key,
    required this.project,
    required this.onUploadImages,
    required this.onStartProcessing,
    required this.onOpenStatus,
    this.isBusy = false,
  });

  final ProjectModel project;
  final VoidCallback onUploadImages;
  final VoidCallback onStartProcessing;
  final VoidCallback onOpenStatus;
  final bool isBusy;

  @override
  Widget build(BuildContext context) {
    final statusColor = _statusColor(project.status, context);
    final qualityColor = _qualityColor(project.qualityClassification, context);
    final canUpload = !isBusy && project.status != ProjectStatus.processing;
    final canStartProcessing =
        !isBusy &&
        project.status != ProjectStatus.processing &&
        project.imageCount > 0;
    final guidance = _guidanceMessage(project);

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    project.name,
                    style: Theme.of(context).textTheme.titleMedium,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                Chip(
                  label: Text(project.status.label),
                  backgroundColor: statusColor.withValues(alpha: 0.15),
                  side: BorderSide(color: statusColor),
                ),
              ],
            ),
            const SizedBox(height: 6),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                Chip(
                  label: Text(project.qualityLabel),
                  backgroundColor: qualityColor.withValues(alpha: 0.12),
                  side: BorderSide(color: qualityColor),
                ),
                Chip(
                  label: Text(project.fallbackUsed ? 'fallback' : 'sin fallback'),
                  backgroundColor: (project.fallbackUsed ? Colors.orange : Colors.green)
                      .withValues(alpha: 0.12),
                ),
                if ((project.outputFormat ?? '').isNotEmpty)
                  Chip(
                    label: Text('Formato ${project.outputFormat!.toUpperCase()}'),
                  ),
              ],
            ),
            const SizedBox(height: 6),
            Text('ID: ${project.id}'),
            Text('Imagenes: ${project.imageCount}'),
            Text(
              'Fecha: ${_formatDate(project.updatedAt ?? project.createdAt)}',
            ),
            if (guidance != null)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(
                  guidance,
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.secondary,
                  ),
                ),
              ),
            if (project.errorMessage != null && project.errorMessage!.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(
                  project.errorMessage!,
                  style: const TextStyle(color: Colors.redAccent),
                ),
              ),
            const SizedBox(height: 10),
            if (isBusy) const LinearProgressIndicator(),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                OutlinedButton.icon(
                  onPressed: canUpload ? onUploadImages : null,
                  icon: const Icon(Icons.add_photo_alternate_outlined),
                  label: const Text('Subir imagenes'),
                ),
                FilledButton.icon(
                  onPressed: canStartProcessing ? onStartProcessing : null,
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Procesar'),
                ),
                TextButton.icon(
                  onPressed: onOpenStatus,
                  icon: const Icon(Icons.info_outline),
                  label: const Text('Estado'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Color _statusColor(ProjectStatus status, BuildContext context) {
    switch (status) {
      case ProjectStatus.created:
        return Colors.blueGrey;
      case ProjectStatus.ready:
        return Colors.indigo;
      case ProjectStatus.processing:
        return Colors.orange;
      case ProjectStatus.completed:
        return Colors.green;
      case ProjectStatus.failed:
        return Colors.red;
      case ProjectStatus.unknown:
        return Theme.of(context).colorScheme.outline;
    }
  }

  Color _qualityColor(String? classification, BuildContext context) {
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

  String _formatDate(DateTime? value) {
    if (value == null) {
      return 'sin fecha';
    }
    final local = value.toLocal();
    final day = local.day.toString().padLeft(2, '0');
    final month = local.month.toString().padLeft(2, '0');
    final year = local.year.toString();
    final hour = local.hour.toString().padLeft(2, '0');
    final minute = local.minute.toString().padLeft(2, '0');
    return '$year-$month-$day $hour:$minute';
  }

  String? _guidanceMessage(ProjectModel project) {
    if (project.status == ProjectStatus.processing) {
      return 'El proyecto esta procesando. Espera a que termine para volver a subir imagenes.';
    }
    if (project.imageCount <= 0) {
      return 'Sube imagenes para habilitar el procesamiento.';
    }
    if (project.status == ProjectStatus.failed) {
      return 'Puedes volver a subir imagenes y reprocesar este proyecto.';
    }
    return null;
  }
}
