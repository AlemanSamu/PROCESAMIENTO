from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.dependencies import API_KEY_HEADER, get_processing_service, require_api_key
from app.core.errors import register_exception_handlers
from app.core.networking import build_health_network_info
from config import get_settings

settings = get_settings()


def _normalized_api_prefix(raw_prefix: str | None) -> str:
    value = str(raw_prefix or "").strip()
    if not value:
        return ""
    if not value.startswith("/"):
        value = f"/{value}"
    return value.rstrip("/")


def _parse_cors_allowed_origins(raw_value: str | None) -> list[str]:
    raw = str(raw_value or "").strip()
    if not raw:
        return ["*"]
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or ["*"]


api_prefix = _normalized_api_prefix(settings.api_prefix)
cors_allowed_origins = _parse_cors_allowed_origins(settings.cors_allowed_origins)
allow_all_origins = "*" in cors_allowed_origins

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Modulo local para recibir imagenes, procesar reconstruccion 3D y devolver el modelo final.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else cors_allowed_origins,
    allow_credentials=False if allow_all_origins else bool(settings.cors_allow_credentials),
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(api_router, prefix=api_prefix)


@app.get("/health", tags=["system"])
def health_check(
    request: Request,
    _: None = Depends(require_api_key),
) -> dict:
    processing_service = get_processing_service()
    return {
        "status": "ok",
        "engine": processing_service.engine_name,
        "profile": str(getattr(settings, "profile", "balanced")),
        "storage_root": str(settings.storage_root),
        "auth": {
            "required": bool((settings.api_key or "").strip()),
            "header_name": API_KEY_HEADER,
        },
        "colmap": {
            "use_gpu": bool(settings.colmap_use_gpu),
            "gpu_mode": str(getattr(settings, "colmap_gpu_mode", "auto")),
            "gpu_probe_timeout_seconds": int(getattr(settings, "colmap_gpu_probe_timeout_seconds", 3)),
            "enable_dense_stages": bool(getattr(settings, "colmap_enable_dense_stages", True)),
            "require_dense_reconstruction": bool(settings.colmap_require_dense_reconstruction),
        },
        "network": build_health_network_info(request),
    }


if api_prefix:

    @app.get(f"{api_prefix}/health", tags=["system"], include_in_schema=False)
    def health_check_prefixed(
        request: Request,
        _: None = Depends(require_api_key),
    ) -> dict:
        return health_check(request=request)
