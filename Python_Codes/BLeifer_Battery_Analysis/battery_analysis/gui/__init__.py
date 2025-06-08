"""
GUI module for the battery analysis package.

This module provides a graphical user interface for the battery_analysis package,
allowing users to interact with the package's functionality in a user-friendly way.
"""

from mongoengine import connect

connect(
    db="battery_test_db",   # your database name
    host="localhost",       # or your MongoDB URI
    port=27017              # default MongoDB port
)


def __getattr__(name):
    """Lazily import GUI components to avoid circular dependencies."""
    if name == 'BatteryAnalysisApp':
        from .app import BatteryAnalysisApp as _BatteryAnalysisApp
        return _BatteryAnalysisApp
    if name == 'ScrollableFrame':
        from .scrollable_frame import ScrollableFrame as _ScrollableFrame
        return _ScrollableFrame
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['BatteryAnalysisApp', 'ScrollableFrame']
