from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prueba el backend local de reconstruccion 3D.")
    parser.add_argument("--input", required=True, type=Path, help="Carpeta con imagenes de entrada.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="URL base del backend.")
    parser.add_argument("--output-format", choices=["glb", "obj"], default="glb", help="Formato final solicitado.")
    parser.add_argument("--api-key", default=os.environ.get("LOCAL3D_API_KEY"), help="API key opcional.")
    parser.add_argument("--project-name", default=None, help="Nombre del proyecto de prueba.")
    parser.add_argument("--timeout-seconds", type=int, default=1800, help="Tiempo maximo de espera.")
    parser.add_argument("--poll-seconds", type=float, default=3.0, help="Intervalo de consulta de estado.")
    parser.add_argument("--download-dir", type=Path, default=Path("tmp_pipeline_downloads"), help="Carpeta de descarga.")
    return parser


def _headers(api_key: str | None = None, *, json_content: bool = False) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if json_content:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _request_json(method: str, url: str, api_key: str | None, payload: dict | None = None) -> dict:
    data = None
    headers = _headers(api_key, json_content=payload is not None)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} en {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se pudo conectar con {url}: {exc}") from exc


def _collect_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError(f"La carpeta de entrada no existe: {input_dir}")
    images = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"No se encontraron imagenes soportadas en: {input_dir}")
    return images


def _build_multipart(files: list[Path]) -> tuple[bytes, str]:
    boundary = f"----Local3DPipeline{uuid.uuid4().hex}"
    body = bytearray()
    for file_path in files:
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="files"; filename="{file_path.name}"\r\n'
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode("utf-8")
        )
        body.extend(file_path.read_bytes())
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _upload_images(base_url: str, project_id: str, images: list[Path], api_key: str | None) -> dict:
    payload, content_type = _build_multipart(images)
    headers = _headers(api_key)
    headers["Content-Type"] = content_type
    request = urllib.request.Request(
        url=f"{base_url}/projects/{project_id}/images",
        data=payload,
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} al subir imagenes: {body}") from exc


def _download_model(base_url: str, model_url: str, project_id: str, output_format: str, api_key: str | None, download_dir: Path) -> Path:
    url = model_url if model_url.startswith("http") else f"{base_url}{model_url}"
    request = urllib.request.Request(url=url, method="GET", headers=_headers(api_key))
    download_dir.mkdir(parents=True, exist_ok=True)
    output_path = download_dir / f"{project_id}_model.{output_format}"
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError("El modelo descargado esta vacio.")
    output_path.write_bytes(payload)
    return output_path


def main() -> int:
    args = _build_parser().parse_args()
    base_url = str(args.base_url).rstrip("/")
    api_key = (args.api_key or "").strip() or None
    images = _collect_images(args.input.resolve())
    project_name = args.project_name or f"pipeline-test-{int(time.time())}"

    print(f"[health] consultando {base_url}/health")
    health = _request_json("GET", f"{base_url}/health", api_key)
    print(f"[health] status={health.get('status')} engine={health.get('engine')} colmap={health.get('colmap')}")

    created = _request_json("POST", f"{base_url}/projects", api_key, {"name": project_name})
    project_id = created["id"]
    print(f"[create] project_id={project_id}")

    uploaded = _upload_images(base_url, project_id, images, api_key)
    print(f"[upload] uploaded={uploaded.get('uploaded_count')} skipped={uploaded.get('skipped_count')} total={uploaded.get('total_images')}")

    started = _request_json(
        "POST",
        f"{base_url}/projects/{project_id}/process",
        api_key,
        {"output_format": args.output_format},
    )
    print(f"[process] engine={started.get('engine')} message={started.get('message')}")

    deadline = time.time() + max(30, int(args.timeout_seconds))
    last_signature = None
    status_payload: dict = {}
    while time.time() < deadline:
        status_payload = _request_json("GET", f"{base_url}/projects/{project_id}/status", api_key)
        signature = (
            status_payload.get("status"),
            status_payload.get("current_stage"),
            status_payload.get("progress"),
            status_payload.get("message"),
        )
        if signature != last_signature:
            progress = status_payload.get("progress")
            progress_text = f"{float(progress) * 100:.1f}%" if isinstance(progress, (int, float)) else "sin porcentaje"
            print(
                f"[status] state={status_payload.get('status')} stage={status_payload.get('current_stage')} "
                f"workflow={(status_payload.get('processing_metadata') or {}).get('workflow_stage')} "
                f"progress={progress_text} fallback={status_payload.get('fallback_used')} message={status_payload.get('message')}"
            )
            last_signature = signature
        if status_payload.get("status") in {"completed", "failed"}:
            break
        time.sleep(max(0.5, float(args.poll_seconds)))
    else:
        raise RuntimeError(f"Timeout esperando el proyecto {project_id}.")

    result = _request_json("GET", f"{base_url}/projects/{project_id}/result", api_key)
    model_url = result.get("model_download_url")
    downloaded_path = None
    if result.get("status") == "completed" and model_url:
        downloaded_path = _download_model(base_url, model_url, project_id, args.output_format, api_key, args.download_dir)

    summary = {
        "project_id": project_id,
        "status": result.get("status"),
        "engine": result.get("engine"),
        "current_stage": result.get("current_stage"),
        "workflow_stage": result.get("workflow_stage"),
        "fallback_used": result.get("fallback_used"),
        "error_message": result.get("error_message"),
        "model_download_url": model_url,
        "downloaded_path": str(downloaded_path) if downloaded_path else None,
        "metrics": result.get("metrics"),
        "artifacts": result.get("artifacts"),
    }
    print("[summary]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
