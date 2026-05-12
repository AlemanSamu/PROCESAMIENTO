from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.engines.colmap_engine import ColmapReconstructionEngine  # noqa: E402


logging.getLogger("app.services.engines.colmap_engine").setLevel(logging.WARNING)


_COMMANDS_TO_PROBE = (
    "feature_extractor",
    "exhaustive_matcher",
    "mapper",
    "model_converter",
    "image_undistorter",
    "patch_match_stereo",
    "stereo_fusion",
    "poisson_mesher",
)


def _normalize_candidate(raw: str | None) -> str | None:
    if not raw:
        return None
    value = str(raw).strip().strip('"').strip("'").strip()
    return value or None


def _run(command: list[str], *, timeout_seconds: int = 12) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(timeout_seconds, 1),
            check=False,
        )
    except FileNotFoundError:
        return {
            "ok": False,
            "reason": "not_found",
            "command": command,
            "return_code": None,
            "output_tail": "",
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "reason": "timeout",
            "command": command,
            "return_code": None,
            "output_tail": "",
        }
    except OSError as exc:
        return {
            "ok": False,
            "reason": f"os_error: {exc}",
            "command": command,
            "return_code": None,
            "output_tail": "",
        }

    output = f"{completed.stdout}\n{completed.stderr}".strip()
    return {
        "ok": completed.returncode == 0,
        "reason": "ok" if completed.returncode == 0 else f"returncode_{completed.returncode}",
        "command": command,
        "return_code": completed.returncode,
        "output_tail": output[-5000:],
    }


def _probe_commands(binary: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for command_name in _COMMANDS_TO_PROBE:
        probe = _run([binary, command_name, "-h"], timeout_seconds=15)
        results[command_name] = {
            "available": bool(probe.get("ok")),
            "return_code": probe.get("return_code"),
            "reason": probe.get("reason"),
        }
    return results


def _probe_colmap_entry(candidate: str | None, label: str) -> dict[str, Any]:
    normalized = _normalize_candidate(candidate)
    candidate_path = Path(normalized) if normalized else None
    entry: dict[str, Any] = {
        "label": label,
        "configured_value": candidate,
        "normalized_candidate": normalized,
        "path_exists": bool(candidate_path.exists()) if candidate_path is not None and candidate_path.is_absolute() else None,
        "path_is_dir": bool(candidate_path.is_dir()) if candidate_path is not None and candidate_path.is_absolute() else None,
        "path_is_file": bool(candidate_path.is_file()) if candidate_path is not None and candidate_path.is_absolute() else None,
        "path_contains_spaces": bool(" " in normalized) if normalized else False,
        "available": False,
        "detected_binary": None,
        "colmap_help": None,
        "colmap_version": None,
        "commands_available": {},
        "sift_gpu_flags_detected": {
            "SiftExtraction.use_gpu": False,
            "SiftMatching.use_gpu": False,
            "FeatureExtraction.use_gpu": False,
            "FeatureMatching.use_gpu": False,
        },
        "dense_cuda_support_hint": None,
    }
    if not normalized:
        return entry

    engine = ColmapReconstructionEngine(colmap_binary=normalized, timeout_seconds=60)
    detected = engine.detect_binary(force_refresh=True)
    entry["detected_binary"] = detected
    entry["available"] = detected is not None
    if detected is None:
        return entry

    root_help = _run([detected, "-h"], timeout_seconds=15)
    entry["colmap_help"] = {
        "ok": bool(root_help.get("ok")),
        "reason": root_help.get("reason"),
        "return_code": root_help.get("return_code"),
    }
    help_text = str(root_help.get("output_tail") or "")
    lower_help = help_text.lower()
    entry["dense_cuda_support_hint"] = "without_cuda" not in lower_help and "without cuda" not in lower_help
    entry["colmap_version"] = engine.get_colmap_version(detected)
    entry["commands_available"] = _probe_commands(detected)

    feature_help = _run([detected, "feature_extractor", "-h"], timeout_seconds=15)
    matcher_help = _run([detected, "exhaustive_matcher", "-h"], timeout_seconds=15)
    feature_text = str(feature_help.get("output_tail") or "")
    matcher_text = str(matcher_help.get("output_tail") or "")
    entry["sift_gpu_flags_detected"] = {
        "SiftExtraction.use_gpu": "--SiftExtraction.use_gpu" in feature_text,
        "SiftMatching.use_gpu": "--SiftMatching.use_gpu" in matcher_text,
        "FeatureExtraction.use_gpu": "--FeatureExtraction.use_gpu" in feature_text,
        "FeatureMatching.use_gpu": "--FeatureMatching.use_gpu" in matcher_text,
    }
    return entry


def _recommendations(payload: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    colmap_ready = bool(payload.get("ready_for_colmap"))
    gpu = payload.get("gpu_probe") if isinstance(payload.get("gpu_probe"), dict) else {}
    gpu_available = bool(gpu.get("available"))
    rtx_4050 = bool(gpu.get("rtx_4050_detected"))

    if not colmap_ready:
        recommendations.append(
            "Configura LOCAL3D_COLMAP_BINARY con ruta absoluta a COLMAP.bat/colmap.exe y verifica `colmap -h`."
        )
    if colmap_ready and not gpu_available:
        recommendations.append(
            "No se detecto GPU NVIDIA con nvidia-smi. Usa LOCAL3D_PROFILE=conservative o balanced con fallback a CPU."
        )
    if colmap_ready and gpu_available and rtx_4050:
        recommendations.append(
            "RTX 4050 detectada. Usa LOCAL3D_PROFILE=balanced para corridas estables y quality para evidencia final."
        )
    if colmap_ready and gpu_available and not rtx_4050:
        recommendations.append("GPU NVIDIA detectada, pero no se identifico RTX 4050 en la salida de nvidia-smi.")

    recommendations.append("Mantener LOCAL3D_COLMAP_ENABLE_DENSE_STAGES=false reduce riesgo en hardware limitado.")
    return recommendations


def main() -> int:
    env_binary_raw = os.environ.get("LOCAL3D_COLMAP_BINARY") or os.environ.get("LOCAL3D_COLMAP_PATH")
    path_binary = shutil.which("colmap")
    default_binary = "colmap"
    gpu_probe = ColmapReconstructionEngine.detect_nvidia_gpu(timeout_seconds=5)

    env_probe = _probe_colmap_entry(env_binary_raw, "LOCAL3D_COLMAP_BINARY")
    path_probe = _probe_colmap_entry(path_binary or default_binary, "PATH/colmap")

    ready_for_colmap = bool(env_probe.get("available") or path_probe.get("available"))
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "local3d_colmap_binary": env_probe,
        "path_colmap": path_probe,
        "gpu_probe": gpu_probe,
        "profiles": {
            name: ColmapReconstructionEngine.profile_options(name, gpu_available=bool(gpu_probe.get("available")))
            for name in ("conservative", "balanced", "quality")
        },
        "ready_for_colmap": ready_for_colmap,
    }
    payload["recommendations"] = _recommendations(payload)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
