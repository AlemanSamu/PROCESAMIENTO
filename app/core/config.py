"""Compatibilidad interna: delega en config.py de raiz."""

from config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
