"""Simple dashboard package exposing the Dash application factory."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import dash

try:
    from .app import create_app
except ImportError:  # pragma: no cover - fallback for direct execution
    import importlib

    create_app = importlib.import_module("app").create_app
except Exception:  # pragma: no cover - optional dependency may be missing

    def create_app() -> "dash.Dash":  # type: ignore[return-type]
        """Raise informative error if Dash is not installed."""
        raise RuntimeError(
            "Dash is required to use the dashboard. Install with 'pip install dash dash-bootstrap-components'."
        )


__all__ = ["create_app"]
