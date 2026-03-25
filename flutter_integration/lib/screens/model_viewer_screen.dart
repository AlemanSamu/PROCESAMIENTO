import 'package:flutter/material.dart';
import 'package:model_viewer_plus/model_viewer_plus.dart';

class ModelViewerScreen extends StatelessWidget {
  const ModelViewerScreen({
    super.key,
    required this.modelUrl,
    required this.projectName,
  });

  final String modelUrl;
  final String projectName;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Modelo 3D: $projectName')),
      body: Column(
        children: [
          Container(
            width: double.infinity,
            color: Colors.blueGrey.shade50,
            padding: const EdgeInsets.all(12),
            child: Text(
              'Fuente del modelo: $modelUrl',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
          const Divider(height: 1),
          Expanded(
            child: ModelViewer(
              src: modelUrl,
              alt: 'Modelo reconstruido',
              autoRotate: true,
              cameraControls: true,
              disableZoom: false,
            ),
          ),
        ],
      ),
    );
  }
}
