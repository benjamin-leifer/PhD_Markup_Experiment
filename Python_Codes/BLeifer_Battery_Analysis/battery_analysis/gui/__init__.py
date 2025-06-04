"""
GUI module for the battery analysis package.

This module provides a graphical user interface for the battery_analysis package,
allowing users to interact with the package's functionality in a user-friendly way.
"""

from .app import BatteryAnalysisApp
from .scrollable_frame import ScrollableFrame

from mongoengine import connect

connect(
    db="battery_test_db",   # your database name
    host="localhost",       # or your MongoDB URI
    port=27017              # default MongoDB port
)


__all__ = ['BatteryAnalysisApp', 'ScrollableFrame']
