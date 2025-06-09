"""Simple dashboard package exposing the Dash application factory."""

try:
    from .app import create_app
except Exception:  # pragma: no cover - optional dependency may be missing
    def create_app() -> "dash.Dash":  # type: ignore[return-type]
        """Raise informative error if Dash is not installed."""
        raise RuntimeError(
            "Dash is required to use the dashboard. Install with 'pip install dash dash-bootstrap-components'."
        )

__all__ = ["create_app"]
