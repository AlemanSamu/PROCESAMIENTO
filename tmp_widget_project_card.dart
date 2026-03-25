import 'package:flutter/material.dart';

import '../models/project_models.dart';
import 'status_chip.dart';

class ProjectCard extends StatelessWidget {
  const ProjectCard({
    super.key,
    required this.project,
    required this.onTap,
  });

  final ProjectItem project;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        onTap: onTap,
        title: Text(project.name),
        subtitle: Text(
          '${project.id} • ${project.imageCount} imagen(es)',
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
        ),
        trailing: StatusChip(status: project.status),
      ),
    );
  }
}
