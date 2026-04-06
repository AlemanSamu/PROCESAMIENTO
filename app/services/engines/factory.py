from __future__ import annotations

from app.services.engines.base_engine import ReconstructionEngine
from app.services.engines.colmap_engine import ColmapReconstructionEngine
from app.services.engines.mock_engine import MockReconstructionEngine


def build_reconstruction_engine(settings) -> ReconstructionEngine:
    requested_engine = str(getattr(settings, 'processing_engine', 'auto')).lower().strip()
    colmap_engine = ColmapReconstructionEngine(
        colmap_binary=str(getattr(settings, 'colmap_binary', 'colmap')),
    )

    if requested_engine == 'colmap' and colmap_engine.is_available() and colmap_engine.is_implemented:
        return colmap_engine

    if requested_engine == 'auto' and colmap_engine.is_available() and colmap_engine.is_implemented:
        return colmap_engine

    return MockReconstructionEngine(
        delay_seconds=int(getattr(settings, 'simulation_delay_seconds', 5)),
    )
