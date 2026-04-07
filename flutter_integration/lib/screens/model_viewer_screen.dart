import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:model_viewer_plus/model_viewer_plus.dart' as mv;
import 'package:webview_flutter/webview_flutter.dart';

import '../models/project_models.dart';
import '../services/local_api_service.dart';

class ModelViewerScreen extends StatefulWidget {
  const ModelViewerScreen({
    super.key,
    required this.apiService,
    required this.project,
  });

  final LocalApiService apiService;
  final ProjectModel project;

  @override
  State<ModelViewerScreen> createState() => _ModelViewerScreenState();
}

class _ModelViewerScreenState extends State<ModelViewerScreen> {
  ModelViewerAsset? _modelAsset;
  Map<String, dynamic> _viewerDiagnostics = const {};
  bool _isPreparingModel = true;
  String? _errorMessage;
  String? _viewerRuntimeError;
  int _viewerEventCount = 0;

  @override
  void initState() {
    super.initState();
    _prepareModelForViewer();
  }

  @override
  void didUpdateWidget(covariant ModelViewerScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.project.id != widget.project.id ||
        oldWidget.project.updatedAt != widget.project.updatedAt ||
        oldWidget.project.modelFilename != widget.project.modelFilename) {
      _prepareModelForViewer();
    }
  }

  Future<void> _prepareModelForViewer() async {
    setState(() {
      _isPreparingModel = true;
      _errorMessage = null;
      _viewerRuntimeError = null;
      _viewerDiagnostics = const {};
      _viewerEventCount = 0;
      _modelAsset = null;
    });

    try {
      final asset = await widget.apiService.downloadModelForViewer(
        project: widget.project,
      );
      if (!mounted) {
        return;
      }
      setState(() {
        _modelAsset = asset;
        _isPreparingModel = false;
      });
      _logDebug(
        'Visor preparado. source=${asset.sourceUri} local=${asset.localPath} type=${asset.detectedType} bytes=${asset.byteCount}',
      );
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _errorMessage = error.toString();
        _isPreparingModel = false;
      });
      _logDebug('No se pudo preparar el visor: $error');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Modelo 3D: ${widget.project.name}')),
      body: Column(
        children: [
          _buildSummaryPanel(context),
          const Divider(height: 1),
          Expanded(child: _buildViewerBody(context)),
          SizedBox(
            height: 220,
            child: _buildDiagnosticsPanel(context),
          ),
        ],
      ),
    );
  }

  Widget _buildSummaryPanel(BuildContext context) {
    final asset = _modelAsset;
    final sourceUrl = asset?.sourceUri.toString() ??
        widget.apiService
            .resolveModelUri(widget.project, cacheBust: true)
            .toString();

    return Container(
      width: double.infinity,
      constraints: const BoxConstraints(maxHeight: 150),
      color: const Color(0xFFF3F6FB),
      padding: const EdgeInsets.all(12),
      child: SelectionArea(
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Backend source URL: $sourceUrl',
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 6),
              Text(
                'Archivo local cargado por el visor: ${asset?.localPath ?? 'pendiente'}',
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 6),
              Text(
                'Tipo detectado: ${asset?.detectedType ?? widget.project.finalModelType ?? 'sin reporte'} | stage=${widget.project.currentStage ?? 'sin reporte'} | metodo=${widget.project.methodUsed ?? widget.project.sparseFallback?['mesh_method'] ?? 'sin reporte'}',
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 6),
              Text(
                'Mesh backend: vertices=${widget.project.meshVertexCount?.toString() ?? 'sin reporte'} faces=${widget.project.meshFaceCount?.toString() ?? 'sin reporte'} fallback=${widget.project.fallbackUsed}',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildViewerBody(BuildContext context) {
    if (_isPreparingModel) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 12),
            Text('Preparando GLB para el visor...'),
          ],
        ),
      );
    }

    if (_errorMessage != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                _errorMessage!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.redAccent),
              ),
              const SizedBox(height: 12),
              FilledButton.icon(
                onPressed: _prepareModelForViewer,
                icon: const Icon(Icons.refresh),
                label: const Text('Reintentar carga del visor'),
              ),
            ],
          ),
        ),
      );
    }

    final asset = _modelAsset;
    if (asset == null) {
      return const Center(
        child: Text('No hay modelo listo para mostrar.'),
      );
    }

    return Padding(
      padding: const EdgeInsets.all(12),
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
          boxShadow: const [
            BoxShadow(
              color: Color(0x19000000),
              blurRadius: 18,
              offset: Offset(0, 8),
            ),
          ],
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(18),
          child: mv.ModelViewer(
            key: ValueKey<String>(asset.localPath),
            src: asset.localUri.toString(),
            id: 'flutter-model-viewer',
            alt: 'Modelo reconstruido',
            loading: mv.Loading.eager,
            reveal: mv.Reveal.auto,
            backgroundColor: const Color(0xFFE9EEF5),
            autoRotate: true,
            autoRotateDelay: 0,
            rotationPerSecond: '18deg',
            cameraControls: true,
            disableZoom: false,
            cameraTarget: 'auto auto auto',
            cameraOrbit: '35deg 70deg auto',
            fieldOfView: '30deg',
            minFieldOfView: '18deg',
            maxFieldOfView: '65deg',
            exposure: 1.15,
            shadowIntensity: 0.55,
            shadowSoftness: 0.35,
            interactionPrompt: mv.InteractionPrompt.none,
            relatedCss: _viewerCss,
            relatedJs: _buildViewerDiagnosticsScript(asset),
            debugLogging: kDebugMode,
            javascriptChannels: {
              mv.JavascriptChannel(
                'ModelViewerDebugChannel',
                onMessageReceived: _handleViewerMessage,
              ),
            },
            onWebViewCreated: (_) {
              _logDebug(
                  'WebView del visor inicializada para ${asset.localPath}');
            },
          ),
        ),
      ),
    );
  }

  Widget _buildDiagnosticsPanel(BuildContext context) {
    final renderMode =
        _viewerDiagnostics['renderMode']?.toString() ?? 'solid_pbr';
    final proxySrc = _viewerDiagnostics['currentSrc']?.toString() ??
        _viewerDiagnostics['srcAttr']?.toString() ??
        'pendiente';

    return Container(
      width: double.infinity,
      color: const Color(0xFFEEF2F7),
      padding: const EdgeInsets.all(12),
      child: SelectionArea(
        child: ListView(
          children: [
            Text(
              'Diagnostico del visor',
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: 8),
            _debugLine('Proxy/model-viewer src', proxySrc),
            _debugLine(
              'Tipo final',
              _modelAsset?.detectedType ??
                  widget.project.finalModelType ??
                  'sin reporte',
            ),
            _debugLine(
                'Render', '$renderMode | wireframe=false | points=false'),
            _debugLine(
              'Bounding box / extents',
              _formatVector(_viewerDiagnostics['extents']),
            ),
            _debugLine(
              'Centro',
              _formatVector(_viewerDiagnostics['center']),
            ),
            _debugLine(
              'Camera orbit inicial',
              _formatOrbit(_viewerDiagnostics['cameraOrbit']),
            ),
            _debugLine(
              'Camera target',
              _formatGeneric(_viewerDiagnostics['cameraTarget']),
            ),
            _debugLine(
              'Field of view',
              _formatGeneric(_viewerDiagnostics['fieldOfView']),
            ),
            _debugLine(
              'Scale',
              _formatGeneric(_viewerDiagnostics['scale']),
            ),
            _debugLine(
              'Material count',
              _formatGeneric(_viewerDiagnostics['materialCount']),
            ),
            _debugLine(
              'Evento JS',
              '${_viewerDiagnostics['phase'] ?? 'sin eventos'} | mensajes=$_viewerEventCount',
            ),
            _debugLine(
              'Archivo final backend',
              widget.project.finalModelPath ?? 'sin reporte',
            ),
            if (_viewerRuntimeError != null)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(
                  'Error runtime del visor: $_viewerRuntimeError',
                  style: const TextStyle(color: Colors.redAccent),
                ),
              ),
            if ((widget.project.meshFaceCount ?? 0) > 0 &&
                (widget.project.meshFaceCount ?? 0) <= 60)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(
                  'La malla tiene pocas caras; el volumen puede verse muy simple aunque el GLB este correcto.',
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.secondary,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _debugLine(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(top: 4),
      child: Text('$label: $value'),
    );
  }

  void _handleViewerMessage(JavaScriptMessage message) {
    try {
      final decoded = jsonDecode(message.message);
      if (decoded is! Map) {
        _logDebug('Mensaje JS no interpretable: ${message.message}');
        return;
      }

      final normalized = decoded.map(
        (key, value) => MapEntry(key.toString(), value),
      );
      _logDebug('ModelViewer JS -> ${jsonEncode(normalized)}');

      if (!mounted) {
        return;
      }

      setState(() {
        _viewerEventCount += 1;
        _viewerDiagnostics = Map<String, dynamic>.from(_viewerDiagnostics)
          ..addAll(normalized);
        if (normalized['type']?.toString() == 'error') {
          _viewerRuntimeError =
              normalized['message']?.toString() ?? 'Error desconocido';
        }
      });
    } catch (error) {
      _logDebug(
          'No se pudo procesar mensaje JS: $error | raw=${message.message}');
    }
  }

  String _formatVector(dynamic value) {
    if (value is Map) {
      final x = _toDouble(value['x']);
      final y = _toDouble(value['y']);
      final z = _toDouble(value['z']);
      if (x != null && y != null && z != null) {
        return '${x.toStringAsFixed(2)}, ${y.toStringAsFixed(2)}, ${z.toStringAsFixed(2)}';
      }
      return value.toString();
    }
    return _formatGeneric(value);
  }

  String _formatOrbit(dynamic value) {
    if (value is Map) {
      final theta = value['theta']?.toString();
      final phi = value['phi']?.toString();
      final radius = value['radius']?.toString();
      if (theta != null || phi != null || radius != null) {
        return [theta, phi, radius]
            .whereType<String>()
            .where((item) => item.isNotEmpty)
            .join(' | ');
      }
      return _formatVector(value);
    }
    return _formatGeneric(value);
  }

  String _formatGeneric(dynamic value) {
    final text = value?.toString().trim();
    if (text == null || text.isEmpty || text == 'null') {
      return 'pendiente';
    }
    return text;
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

  String _buildViewerDiagnosticsScript(ModelViewerAsset asset) {
    final expectedType = jsonEncode(asset.detectedType);
    return '''
(() => {
  const channelName = 'ModelViewerDebugChannel';
  const viewerId = 'flutter-model-viewer';
  const expectedType = $expectedType;

  function post(payload) {
    try {
      if (window[channelName] && typeof window[channelName].postMessage === 'function') {
        window[channelName].postMessage(JSON.stringify(payload));
      }
    } catch (error) {
      console.log('ModelViewer debug channel error', error);
    }
  }

  function toPlainVector(value) {
    if (!value) {
      return null;
    }
    const plain = {};
    if (typeof value.x === 'number') plain.x = Number(value.x);
    if (typeof value.y === 'number') plain.y = Number(value.y);
    if (typeof value.z === 'number') plain.z = Number(value.z);
    if (value.theta !== undefined) plain.theta = String(value.theta);
    if (value.phi !== undefined) plain.phi = String(value.phi);
    if (value.radius !== undefined) plain.radius = String(value.radius);
    try {
      const raw = value.toString ? String(value.toString()) : null;
      if (raw && raw !== '[object Object]') plain.raw = raw;
    } catch (_) {}
    return Object.keys(plain).length ? plain : null;
  }

  function toPlainValue(value) {
    if (value == null) {
      return null;
    }
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      return value;
    }
    const vector = toPlainVector(value);
    if (vector) {
      return vector;
    }
    try {
      return String(value);
    } catch (_) {
      return null;
    }
  }

  function emit(phase, extra) {
    const viewer = document.getElementById(viewerId);
    if (!viewer) {
      post({type: 'error', phase, message: 'viewer_not_found'});
      return;
    }

    post(Object.assign({
      type: 'diagnostics',
      phase,
      expectedType,
      srcAttr: viewer.getAttribute('src'),
      currentSrc: viewer.currentSrc || viewer.src || null,
      cameraTarget: toPlainValue(viewer.cameraTarget || viewer.getAttribute('camera-target')),
      fieldOfView: toPlainValue(viewer.fieldOfView || viewer.getAttribute('field-of-view')),
      scale: toPlainValue(viewer.scale || viewer.getAttribute('scale')),
      extents: viewer.getDimensions ? toPlainVector(viewer.getDimensions()) : null,
      center: viewer.getBoundingBoxCenter ? toPlainVector(viewer.getBoundingBoxCenter()) : null,
      cameraOrbit: viewer.getCameraOrbit ? toPlainVector(viewer.getCameraOrbit()) : null,
      materialCount: viewer.model && Array.isArray(viewer.model.materials) ? viewer.model.materials.length : null,
      renderMode: 'solid_pbr',
      wireframe: false,
      points: false
    }, extra || {}));
  }

  function applyFraming(viewer) {
    try {
      viewer.cameraTarget = 'auto auto auto';
      viewer.cameraOrbit = '35deg 70deg auto';
      viewer.fieldOfView = '30deg';
      if (viewer.updateFraming) {
        viewer.updateFraming();
      }
      if (viewer.jumpCameraToGoal) {
        viewer.jumpCameraToGoal();
      }
    } catch (error) {
      post({type: 'error', phase: 'apply_framing', message: String(error)});
    }
  }

  function setup() {
    const viewer = document.getElementById(viewerId);
    if (!viewer) {
      post({type: 'error', phase: 'setup', message: 'viewer_not_found'});
      return;
    }

    emit('setup');
    viewer.addEventListener('load', () => {
      applyFraming(viewer);
      emit('load');
      setTimeout(() => emit('post_load_250ms'), 250);
      setTimeout(() => emit('post_load_1000ms'), 1000);
    });
    viewer.addEventListener('error', (event) => {
      post({
        type: 'error',
        phase: 'load',
        message: event && event.detail && event.detail.type ? event.detail.type : 'load_error'
      });
    });
  }

  const start = () => {
    if (window.customElements && window.customElements.whenDefined) {
      window.customElements.whenDefined('model-viewer').then(setup).catch((error) => {
        post({type: 'error', phase: 'when_defined', message: String(error)});
      });
      return;
    }
    setup();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start, {once: true});
  } else {
    start();
  }
})();
''';
  }

  void _logDebug(String message) {
    if (kDebugMode) {
      debugPrint('[ModelViewerScreen] $message');
    }
  }
}

const String _viewerCss = '''
body {
  background: linear-gradient(180deg, #f5f7fb 0%, #e3e9f2 100%);
}

model-viewer {
  background:
      radial-gradient(circle at top, rgba(255, 255, 255, 0.95) 0%, rgba(236, 241, 248, 0.98) 52%, rgba(218, 225, 236, 1) 100%);
}
''';
