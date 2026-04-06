from __future__ import annotations

from pathlib import Path

from app.models.schemas import OutputFormat

from .artifacts import PipelineStageResult, ReconstructionPipelineResult, write_json
from .exporter import ModelExporter
from .feature_matcher import FeatureMatcher
from .image_preprocessor import ImagePreprocessor
from .mesh_builder import MeshBuilder
from .point_cloud_builder import PointCloudBuilder
from .pose_estimator import PoseEstimator


class ReconstructionPipeline:
    """Coordina las etapas de reconstruccion 3D del backend local."""

    def __init__(
        self,
        *,
        preprocessor: ImagePreprocessor | None = None,
        feature_matcher: FeatureMatcher | None = None,
        pose_estimator: PoseEstimator | None = None,
        point_cloud_builder: PointCloudBuilder | None = None,
        mesh_builder: MeshBuilder | None = None,
        exporter: ModelExporter | None = None,
    ) -> None:
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.feature_matcher = feature_matcher or FeatureMatcher()
        self.pose_estimator = pose_estimator or PoseEstimator()
        self.point_cloud_builder = point_cloud_builder or PointCloudBuilder()
        self.mesh_builder = mesh_builder or MeshBuilder()
        self.exporter = exporter or ModelExporter()

    def execute(
        self,
        project_id: str,
        images_dir: Path,
        output_dir: Path,
        output_format: OutputFormat,
    ) -> ReconstructionPipelineResult:
        pipeline_dir = output_dir / "pipeline"
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        stage_results: list[PipelineStageResult] = []

        # Etapa 1: validacion, normalizacion y metrica real de imagen.
        preprocessed_images, preprocessing_report = self.preprocessor.run(images_dir, pipeline_dir)
        stage_results.append(preprocessing_report)

        # Etapa 2: extraccion de caracteristicas. Usa OpenCV o pixels reales si estan disponibles
        # y cae a un extractor sintetico solo cuando no puede leer la imagen o no hay soporte.
        features, matches, feature_report = self.feature_matcher.run(preprocessed_images, pipeline_dir)
        stage_results.append(feature_report)

        # Etapa 3: estimacion de pose. Si hay correspondencias reales suficientes, estima una
        # transformacion aproximada; de lo contrario mantiene el modelo sintetico anterior.
        poses, pose_report = self.pose_estimator.run(features, matches, pipeline_dir)
        stage_results.append(pose_report)

        # Etapa 4: nube de puntos. Usa las correspondencias reales cuando existen y si no,
        # conserva la nube sintetica estable para no romper el flujo.
        point_cloud, point_cloud_report = self.point_cloud_builder.run(
            poses,
            pipeline_dir,
            matches=matches,
        )
        stage_results.append(point_cloud_report)

        # Etapa 5: malla. Sigue siendo una aproximacion simple pero ya parte de una nube
        # de puntos mejor sustentada por datos reales cuando estan disponibles.
        mesh, mesh_report = self.mesh_builder.run(point_cloud, pipeline_dir)
        stage_results.append(mesh_report)

        # Etapa 6: exportacion final. El formato real del archivo es valido aunque la
        # geometria provenga de una reconstruccion aproximada o sintetica.
        export_result = self.exporter.export(
            project_id=project_id,
            mesh=mesh,
            output_dir=output_dir,
            output_format=output_format,
            work_dir=pipeline_dir,
        )
        stage_results.append(
            PipelineStageResult(
                name=self.exporter.name,
                status="completed",
                summary=f"Modelo exportado como {output_format.value.upper()}.",
                mode="real",
                artifact_path=export_result.model_path,
                metrics=export_result.to_dict(),
            )
        )

        report_path = write_json(
            pipeline_dir / f"{project_id}_pipeline_report.json",
            {
                "project_id": project_id,
                "output_format": output_format.value,
                "mode_summary": {item.name: item.mode for item in stage_results},
                "stage_results": [item.to_dict() for item in stage_results],
                "image_count": len(preprocessed_images),
                "feature_count": len(features),
                "match_count": len(matches),
                "point_count": len(point_cloud.points),
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "model_path": str(export_result.model_path),
            },
        )

        return ReconstructionPipelineResult(
            project_id=project_id,
            output_format=output_format,
            model_path=export_result.model_path,
            report_path=report_path,
            stage_results=stage_results,
            image_count=len(preprocessed_images),
            feature_count=len(features),
            match_count=len(matches),
            point_count=len(point_cloud.points),
            face_count=len(mesh.faces),
        )
