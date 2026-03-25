from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

from app.core.errors import InvalidProjectStateError
from app.models.schemas import OutputFormat
from app.services.engines.base_engine import ReconstructionEngine
from app.services.engines.colmap_engine import ColmapReconstructionEngine
from app.services.engines.mock_engine import MockReconstructionEngine
from app.services.project_service import ProjectService
from app.services.storage_service import StorageService


class ProcessingService:
    def __init__(
        self,
        project_service: ProjectService,
        storage_service: StorageService,
        settings,
    ) -> None:
        self.project_service = project_service
        self.storage_service = storage_service
        self.settings = settings
        self._engine = self._build_engine()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reconstruction")
        self._jobs: dict[str, Future] = {}
        self._jobs_lock = Lock()

    @property
    def engine_name(self) -> str:
        return self._engine.name

    def _build_engine(self) -> ReconstructionEngine:
        requested_engine = self.settings.processing_engine.lower().strip()
        colmap_engine = ColmapReconstructionEngine(colmap_binary=self.settings.colmap_binary)

        if requested_engine == "colmap" and colmap_engine.is_available() and colmap_engine.is_implemented:
            return colmap_engine

        if requested_engine == "auto" and colmap_engine.is_available() and colmap_engine.is_implemented:
            return colmap_engine

        return MockReconstructionEngine(delay_seconds=self.settings.simulation_delay_seconds)

    def start_processing(self, project_id: str, output_format: OutputFormat) -> str:
        self._ensure_not_running(project_id)
        self.project_service.mark_processing(project_id, output_format)

        future = self._executor.submit(self._run_reconstruction_job, project_id, output_format)
        with self._jobs_lock:
            self._jobs[project_id] = future
        future.add_done_callback(lambda _: self._cleanup_job(project_id))
        return self._engine.name

    def _ensure_not_running(self, project_id: str) -> None:
        with self._jobs_lock:
            existing = self._jobs.get(project_id)
        if existing and not existing.done():
            raise InvalidProjectStateError("El proyecto ya tiene un proceso en ejecucion.")

    def _run_reconstruction_job(self, project_id: str, output_format: OutputFormat) -> None:
        try:
            images_dir = self.storage_service.get_images_dir(project_id)
            output_dir = self.storage_service.get_output_dir(project_id)
            model_path = self._engine.reconstruct(project_id, images_dir, output_dir, output_format)
            self.project_service.mark_completed(project_id, output_format, model_path.name)
        except Exception as exc:
            self.project_service.mark_failed(project_id, str(exc))

    def _cleanup_job(self, project_id: str) -> None:
        with self._jobs_lock:
            self._jobs.pop(project_id, None)
