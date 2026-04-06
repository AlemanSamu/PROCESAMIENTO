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
            Text('ID: ${project.id}'),
            Text('Imagenes: ${project.imageCount}'),
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
                  onPressed: isBusy ? null : onUploadImages,
                  icon: const Icon(Icons.add_photo_alternate_outlined),
                  label: const Text('Subir imagenes'),
                ),
                FilledButton.icon(
                  onPressed: isBusy ? null : onStartProcessing,
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
}
