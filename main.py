from fastapi import Depends, FastAPI, Request

from app.api.router import api_router
from app.core.dependencies import API_KEY_HEADER, get_processing_service, require_api_key
from app.core.errors import register_exception_handlers
from app.core.networking import build_health_network_info
from config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Modulo local para recibir imagenes, procesar reconstruccion 3D y devolver el modelo final.",
)

register_exception_handlers(app)
app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/health", tags=["system"])
def health_check(
    request: Request,
    _: None = Depends(require_api_key),
) -> dict:
    processing_service = get_processing_service()
    return {
        "status": "ok",
        "engine": processing_service.engine_name,
        "storage_root": str(settings.storage_root),
        "auth": {
            "required": bool((settings.api_key or "").strip()),
            "header_name": API_KEY_HEADER,
        },
        "network": build_health_network_info(request),
    }
