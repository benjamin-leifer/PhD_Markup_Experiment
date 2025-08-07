"""Entry point to run the dashboard with ``python -m dashboard``."""

try:
    from . import create_app
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    create_app = importlib.import_module("app").create_app


def main() -> None:
    """Instantiate the Dash app and run its server."""
    app = create_app()
    app.run_server()


if __name__ == "__main__":  # pragma: no cover - module execution guard
    main()
