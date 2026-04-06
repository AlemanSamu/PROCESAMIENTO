from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _request_json(method: str, url: str, payload: dict | None = None, headers: dict | None = None) -> dict:
    request_headers = {"Accept": "application/json", **(headers or {})}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url=url, data=data, method=method, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} en {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se pudo conectar con {url}: {exc}") from exc


def _request_binary(url: str) -> tuple[bytes, dict]:
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} en {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se pudo descargar {url}: {exc}") from exc


def _build_multipart(files: list[Path]) -> tuple[bytes, str]:
    boundary = f"----Local3D{uuid.uuid4().hex}"
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


def _upload_images(base_url: str, project_id: str, images: list[Path]) -> dict:
    body, content_type = _build_multipart(images)
    request = urllib.request.Request(
        url=f"{base_url}/projects/{project_id}/images",
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": content_type,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} al subir imagenes: {body_text}") from exc


def _is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def _discover_images_dir(repo_root: Path) -> Path:
    projects_root = repo_root / "data" / "projects"
    if not projects_root.exists():
        raise RuntimeError(
            "No se encontro `data/projects` para autodetectar imagenes. Usa --images-dir con una carpeta real."
        )

    candidates: list[tuple[int, float, Path]] = []
    for images_dir in projects_root.glob("*/images"):
        if not images_dir.is_dir():
            continue
        files = [path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES]
        readable_files = [path for path in files if _is_readable_image(path)]
        if len(readable_files) < 3:
            continue
        latest_mtime = max(path.stat().st_mtime for path in readable_files)
        candidates.append((len(readable_files), latest_mtime, images_dir))

    if not candidates:
        raise RuntimeError(
            "No hay un dataset local autodetectable con al menos 3 imagenes legibles por COLMAP. Usa --images-dir con una carpeta real de fotos con overlap."
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _collect_images(images_dir: Path, limit: int | None = None) -> list[Path]:
    if not images_dir.exists() or not images_dir.is_dir():
        raise RuntimeError(f"La carpeta de imagenes no existe: {images_dir}")

    candidates = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    )
    valid_images: list[Path] = []
    invalid_images: list[str] = []
    for path in candidates:
        if _is_readable_image(path):
            valid_images.append(path)
        else:
            invalid_images.append(path.name)

    if len(valid_images) < 3:
        raise RuntimeError(
            "Se necesitan al menos 3 imagenes reales y legibles por COLMAP para la validacion end-to-end. "
            f"Invalidas detectadas: {invalid_images or 'ninguna'}."
        )
    if limit is not None:
        valid_images = valid_images[:limit]
    return valid_images


def _wait_for_completion(base_url: str, project_id: str, timeout_seconds: int, poll_interval: float) -> dict:
    deadline = time.time() + timeout_seconds
    last_signature = None

    while time.time() < deadline:
        status_payload = _request_json("GET", f"{base_url}/projects/{project_id}/status")
        signature = (
            status_payload.get("status"),
            status_payload.get("current_stage"),
            status_payload.get("progress"),
            status_payload.get("message"),
        )
        if signature != last_signature:
            progress = status_payload.get("progress")
            progress_text = f"{round(float(progress) * 100, 1)}%" if isinstance(progress, (int, float)) else "sin porcentaje"
            print(
                f"[status] state={status_payload.get('status')} stage={status_payload.get('current_stage')} "
                f"progress={progress_text} message={status_payload.get('message')}"
            )
            last_signature = signature

        if status_payload.get("status") in {"completed", "failed"}:
            return status_payload

        time.sleep(poll_interval)

    raise RuntimeError(f"Timeout esperando la finalizacion del proyecto {project_id}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida el flujo real end-to-end del backend FastAPI + COLMAP.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="URL base del backend FastAPI.")
    parser.add_argument("--images-dir", help="Carpeta con imagenes reales para subir al backend.")
    parser.add_argument(
        "--output-format",
        choices=["obj", "glb"],
        default="obj",
        help="Formato solicitado al endpoint /process. OBJ es el recomendado para validacion real.",
    )
    parser.add_argument("--project-name", help="Nombre del proyecto a crear.")
    parser.add_argument("--timeout-seconds", type=int, default=1800, help="Timeout maximo de espera.")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Segundos entre consultas a /status.")
    parser.add_argument("--max-images", type=int, help="Limita el numero de imagenes subidas para la prueba.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    images_dir = Path(args.images_dir).resolve() if args.images_dir else _discover_images_dir(repo_root)
    images = _collect_images(images_dir, limit=args.max_images)
    project_name = args.project_name or f"colmap-real-e2e-{int(time.time())}"

    print(f"[info] using dataset: {images_dir}")
    print(f"[info] images selected: {len(images)}")

    health = _request_json("GET", f"{args.base_url}/health")
    print(f"[health] {json.dumps(health, ensure_ascii=False)}")
    if health.get("engine") != "colmap":
        raise RuntimeError(
            f"El backend no reporta engine=colmap en /health. Respuesta actual: {health}"
        )

    created = _request_json("POST", f"{args.base_url}/projects", {"name": project_name})
    project_id = created["id"]
    print(f"[create] project_id={project_id} name={created.get('name')}")

    uploaded = _upload_images(args.base_url, project_id, images)
    print(f"[upload] uploaded_count={uploaded.get('uploaded_count')} total_images={uploaded.get('total_images')}")

    started = _request_json(
        "POST",
        f"{args.base_url}/projects/{project_id}/process",
        {"output_format": args.output_format},
    )
    print(f"[process] engine={started.get('engine')} message={started.get('message')}")
    if started.get("engine") != "colmap":
        raise RuntimeError(f"El proceso no arranco con COLMAP: {started}")

    final_status = _wait_for_completion(args.base_url, project_id, args.timeout_seconds, args.poll_interval)
    print(f"[final-status] {json.dumps(final_status, ensure_ascii=False)}")

    if final_status.get("status") != "completed":
        raise RuntimeError(
            f"La reconstruccion no termino en completed. Estado final: {final_status.get('status')} | "
            f"error={final_status.get('error_message')} | metadata={final_status.get('processing_metadata')}"
        )

    processing_metadata = final_status.get("processing_metadata") or {}
    if processing_metadata.get("fallback", {}).get("used"):
        raise RuntimeError(f"La prueba uso fallback a mock y eso invalida la validacion real: {processing_metadata}")
    if (final_status.get("engine") or processing_metadata.get("engine")) != "colmap":
        raise RuntimeError(f"El engine final no es COLMAP: {final_status}")

    model_url = final_status.get("model_download_url")
    if not model_url:
        raise RuntimeError("/status no devolvio model_download_url al completar el proceso.")

    downloaded, headers = _request_binary(f"{args.base_url}{model_url}")
    print(f"[download] bytes={len(downloaded)} content_type={headers.get('Content-Type')}")
    if not downloaded:
        raise RuntimeError("El endpoint /model devolvio un archivo vacio.")

    local_model_path = (
        processing_metadata.get("artifacts", {}).get("model_path")
        or processing_metadata.get("output_path")
    )
    if local_model_path:
        local_model = Path(local_model_path)
        if not local_model.exists():
            raise RuntimeError(f"El archivo de salida reportado por el backend no existe: {local_model}")
        print(f"[artifact] model_path={local_model}")

    metrics = final_status.get("metrics") or processing_metadata.get("metrics") or {}
    print(f"[metrics] {json.dumps(metrics, ensure_ascii=False)}")
    print("VALIDACION E2E REAL EXITOSA")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"VALIDACION E2E REAL FALLIDA: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc