"""Web dashboard module for battery analysis."""

try:
    from .app import create_app
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    create_app = importlib.import_module("app").create_app

__all__ = ["create_app"]
