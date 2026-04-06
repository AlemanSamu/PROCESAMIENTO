from __future__ import annotations

import logging

from app.services.engines.base_engine import ReconstructionEngine
from app.services.engines.colmap_engine import ColmapReconstructionEngine
from app.services.engines.mock_engine import MockReconstructionEngine

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _build_mock_engine(settings) -> MockReconstructionEngine:
    return MockReconstructionEngine(
        delay_seconds=int(getattr(settings, "simulation_delay_seconds", 5)),
    )


def _build_colmap_engine(settings) -> ColmapReconstructionEngine:
    configured_path = str(
        getattr(settings, "colmap_path", None)
        or getattr(settings, "colmap_binary", "colmap")
    )
    return ColmapReconstructionEngine(
        colmap_binary=configured_path,
        timeout_seconds=int(getattr(settings, "colmap_timeout_seconds", 1800)),
        use_gpu=bool(getattr(settings, "colmap_use_gpu", False)),
        camera_model=str(getattr(settings, "colmap_camera_model", "SIMPLE_RADIAL")),
        single_camera=bool(getattr(settings, "colmap_single_camera", True)),
    )


def build_reconstruction_engines(settings) -> tuple[ReconstructionEngine, ReconstructionEngine | None]:
    requested_engine = str(getattr(settings, "processing_engine", "colmap")).lower().strip()
    colmap_engine = _build_colmap_engine(settings)
    mock_engine = _build_mock_engine(settings)
    allow_fallback = bool(getattr(settings, "colmap_fallback_to_mock", True))

    if requested_engine not in {"colmap", "auto", "mock"}:
        logger.warning(
            "Unknown processing_engine '%s'. Falling back to preferred COLMAP selection.",
            requested_engine,
        )
        requested_engine = "colmap"

    logger.info(
        "Selecting reconstruction engine. requested=%s configured_colmap_path=%s fallback_to_mock=%s",
        requested_engine,
        colmap_engine.colmap_binary,
        allow_fallback,
    )

    colmap_available = colmap_engine.is_available() and colmap_engine.is_implemented
    if colmap_available:
        logger.info("COLMAP detected successfully. binary=%s", colmap_engine.detected_binary)
    else:
        logger.warning(
            "COLMAP was not detected. configured_colmap_path=%s. MockEngine will be used when needed.",
            colmap_engine.colmap_binary,
        )

    if requested_engine == "mock":
        logger.info("Selected reconstruction engine: mock (explicit configuration).")
        return mock_engine, None

    if colmap_available:
        logger.info("Selected reconstruction engine: colmap")
        return colmap_engine, mock_engine if allow_fallback else None

    logger.warning("Selected reconstruction engine: mock (COLMAP unavailable).")
    return mock_engine, None


def build_reconstruction_engine(settings) -> ReconstructionEngine:
    engine, _ = build_reconstruction_engines(settings)
    return engine