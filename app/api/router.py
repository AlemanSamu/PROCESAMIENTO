from fastapi import APIRouter

from app.api.routes.projects import router as projects_router

api_router = APIRouter()
api_router.include_router(projects_router)
