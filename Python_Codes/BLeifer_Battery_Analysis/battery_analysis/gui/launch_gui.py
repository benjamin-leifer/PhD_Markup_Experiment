#!/usr/bin/env python
"""
Launcher script for the Battery Test Data Analysis GUI.

This script launches the GUI application for the battery_analysis package.
"""

import os
import sys
import logging

# Ensure that the package root is in the Python path
package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

# Import the GUI application
try:
    from battery_analysis.gui.app import BatteryAnalysisApp
except ImportError as e:
    print(f"Error importing the GUI: {e}")
    print("Make sure you have the required dependencies installed:")
    print("  - tkinter (should be included with Python)")
    print("  - matplotlib")
    print("  - numpy")
    print("  - pandas")
    sys.exit(1)

def main():
    """Launch the GUI application."""
    try:
        app = BatteryAnalysisApp()
        app.mainloop()
    except Exception as e:
        print(f"Error launching the GUI: {e}")
        logging.exception("Error in GUI application")
        sys.exit(1)

if __name__ == "__main__":
    main()
