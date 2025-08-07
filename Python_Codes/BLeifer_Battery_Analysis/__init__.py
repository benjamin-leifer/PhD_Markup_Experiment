"""Distribution package for the battery_analysis library used in tests."""

try:
    from .battery_analysis import *  # noqa: F401, F403
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    globals().update(importlib.import_module("battery_analysis").__dict__)
