import 'package:flutter/material.dart';

import '../models/project_models.dart';

class StatusChip extends StatelessWidget {
  const StatusChip({
    super.key,
    required this.status,
  });

  final ProjectStatus status;

  @override
  Widget build(BuildContext context) {
    return Chip(
      label: Text(_label(status)),
      backgroundColor: _color(status).withValues(alpha: 0.15),
      side: BorderSide(color: _color(status)),
      labelStyle: TextStyle(color: _color(status)),
    );
  }

  static String _label(ProjectStatus status) {
    switch (status) {
      case ProjectStatus.created:
        return 'Creado';
      case ProjectStatus.ready:
        return 'Listo';
      case ProjectStatus.processing:
        return 'Procesando';
      case ProjectStatus.completed:
        return 'Completado';
      case ProjectStatus.failed:
        return 'Fallido';
      case ProjectStatus.unknown:
        return 'Desconocido';
    }
  }

  static Color _color(ProjectStatus status) {
    switch (status) {
      case ProjectStatus.created:
        return Colors.blueGrey;
      case ProjectStatus.ready:
        return Colors.orange;
      case ProjectStatus.processing:
        return Colors.blue;
      case ProjectStatus.completed:
        return Colors.green;
      case ProjectStatus.failed:
        return Colors.red;
      case ProjectStatus.unknown:
        return Colors.grey;
    }
  }
}
