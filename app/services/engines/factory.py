from __future__ import annotations

import logging
import subprocess

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
    gpu_mode = _resolve_gpu_mode(settings)
    use_gpu, gpu_reason = _resolve_gpu_request(gpu_mode, settings)
    logger.info(
        "COLMAP GPU selection resolved. mode=%s use_gpu=%s reason=%s",
        gpu_mode,
        use_gpu,
        gpu_reason,
    )
    return ColmapReconstructionEngine(
        colmap_binary=configured_path,
        timeout_seconds=int(getattr(settings, "colmap_timeout_seconds", 1800)),
        use_gpu=use_gpu,
        profile=str(getattr(settings, "profile", "balanced")),
        enable_dense_stages=bool(getattr(settings, "colmap_enable_dense_stages", True)),
        camera_model=str(getattr(settings, "colmap_camera_model", "SIMPLE_RADIAL")),
        single_camera=bool(getattr(settings, "colmap_single_camera", True)),
        require_dense_reconstruction=bool(getattr(settings, "colmap_require_dense_reconstruction", False)),
        gpu_mode=gpu_mode,
        gpu_probe_reason=gpu_reason,
    )


def _resolve_gpu_mode(settings) -> str:
    raw_mode = str(getattr(settings, "colmap_gpu_mode", "") or "").strip().lower()
    if raw_mode in {"auto", "enabled", "disabled"}:
        return raw_mode
    if bool(getattr(settings, "colmap_use_gpu", False)):
        return "enabled"
    return "auto"


def _resolve_gpu_request(gpu_mode: str, settings) -> tuple[bool, str]:
    if gpu_mode == "enabled":
        return True, "gpu_forced_enabled_by_config"
    if gpu_mode == "disabled":
        return False, "gpu_forced_disabled_by_config"

    timeout_seconds = max(1, int(getattr(settings, "colmap_gpu_probe_timeout_seconds", 3)))
    probe_ok, probe_reason = _probe_nvidia_gpu(timeout_seconds=timeout_seconds)
    if probe_ok:
        return True, probe_reason

    legacy_gpu_flag = bool(getattr(settings, "colmap_use_gpu", False))
    if legacy_gpu_flag:
        return True, "legacy_colmap_use_gpu_true"
    return False, probe_reason


def _probe_nvidia_gpu(*, timeout_seconds: int) -> tuple[bool, str]:
    try:
        probe = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return False, "nvidia_smi_not_found"
    except subprocess.TimeoutExpired:
        return False, "nvidia_smi_timeout"
    except OSError:
        return False, "nvidia_smi_os_error"

    output = f"{probe.stdout}\n{probe.stderr}".lower()
    if probe.returncode != 0:
        return False, f"nvidia_smi_returncode_{probe.returncode}"
    if "gpu" not in output:
        return False, "nvidia_smi_no_gpu_listed"
    return True, "nvidia_smi_detected_gpu"


def build_reconstruction_engines(settings) -> tuple[ReconstructionEngine, ReconstructionEngine | None]:
    requested_engine = str(getattr(settings, "processing_engine", "colmap")).lower().strip()
    colmap_engine = _build_colmap_engine(settings)
    mock_engine = _build_mock_engine(settings)
    allow_fallback = bool(getattr(settings, "colmap_fallback_to_mock", False))
    academic_fallback_enabled = bool(getattr(settings, "primitive_box_fallback_enabled", False))

    if requested_engine not in {"colmap", "auto", "mock"}:
        logger.warning(
            "Unknown processing_engine '%s'. Falling back to preferred COLMAP selection.",
            requested_engine,
        )
        requested_engine = "colmap"

    logger.info(
        "Selecting reconstruction engine. requested=%s configured_colmap_path=%s fallback_to_mock=%s primitive_box_fallback=%s",
        requested_engine,
        colmap_engine.colmap_binary,
        allow_fallback,
        academic_fallback_enabled,
    )

    colmap_available = colmap_engine.is_available() and colmap_engine.is_implemented
    if colmap_available:
        logger.info("COLMAP detected successfully. binary=%s", colmap_engine.detected_binary)
    else:
        logger.warning(
            "COLMAP was not detected. configured_colmap_path=%s.",
            colmap_engine.colmap_binary,
        )

    if requested_engine == "mock":
        logger.info("Selected reconstruction engine: mock (explicit configuration).")
        return mock_engine, None

    if requested_engine == "auto":
        if colmap_available:
            logger.info("Selected reconstruction engine: colmap (auto mode)")
            return colmap_engine, mock_engine if allow_fallback else None

        if academic_fallback_enabled:
            logger.warning(
                "Selected reconstruction engine: colmap (auto mode, COLMAP unavailable, primitive fallback enabled). "
                "The job will fail fast in COLMAP and recover with the academic box fallback."
            )
            return colmap_engine, None

        logger.warning("Selected reconstruction engine: mock (auto mode, COLMAP unavailable and primitive fallback disabled).")
        return mock_engine, None

    if colmap_available:
        logger.info("Selected reconstruction engine: colmap (explicit configuration).")
    else:
        logger.warning(
            "Selected reconstruction engine: colmap (explicit configuration), but COLMAP is unavailable. "
            "Processing will fail instead of falling back to mock."
        )
    return colmap_engine, None


def build_reconstruction_engine(settings) -> ReconstructionEngine:
    engine, _ = build_reconstruction_engines(settings)
    return engine
