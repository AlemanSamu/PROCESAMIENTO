from fastapi import FastAPI

from app.api.router import api_router
from app.core.dependencies import get_processing_service
from app.core.errors import register_exception_handlers
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
def health_check() -> dict:
    processing_service = get_processing_service()
    return {
        "status": "ok",
        "engine": processing_service.engine_name,
        "storage_root": str(settings.storage_root),
    }
