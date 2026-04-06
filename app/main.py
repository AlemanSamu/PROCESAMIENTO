"""Punto de entrada ASGI compatible con `uvicorn app.main:app`."""

from main import app

__all__ = ["app"]