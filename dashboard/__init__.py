"""Simple dashboard package exposing the Dash application factory."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import dash

try:
    from .app import create_app
except Exception:  # pragma: no cover - optional dependency may be missing
    try:  # pragma: no cover - fallback for direct execution
        import importlib

        create_app = importlib.import_module("app").create_app
    except Exception:  # pragma: no cover - dependencies missing

        def create_app(
            test_role: str | None = None, enable_login: bool = False
        ) -> "dash.Dash":
            """Raise informative error if Dash is not installed."""
            raise RuntimeError(
                "Dash is required to use the dashboard. Install with 'pip install dash dash-bootstrap-components'."
            )


__all__ = ["create_app"]
