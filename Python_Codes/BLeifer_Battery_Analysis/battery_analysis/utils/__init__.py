"""
Utility functions for the battery analysis package.
"""

import logging
import pickle
import os
import matplotlib

# Use the Qt backend only if a backend hasn't been configured already and
# the Qt dependencies are available.  This prevents import errors during
# headless test runs where PyQt/PySide are not installed.
if os.environ.get("MPLBACKEND") is None and matplotlib.get_backend().lower() != "agg":
    try:  # pragma: no cover - depends on environment
        matplotlib.use("Qt5Agg")
    except Exception:  # fallback when Qt bindings are missing
        matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:  # pragma: no cover - optional Qt dependencies
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT, FigureManagerQT
    import matplotlib.backends.qt_editor.figureoptions as figureoptions

    class FullToolbar(NavigationToolbar2QT):
        """NavigationToolbar2QT with a Configure button."""

        toolitems = NavigationToolbar2QT.toolitems + (
            (
                "Configure",
                "Edit axis/curve/image properties",
                "qt4_editor_options",
                "configure",
            ),
        )

        def configure(self):
            if figureoptions is not None:
                try:
                    figureoptions.figure_edit(self.canvas.figure)
                except Exception:
                    pass

    FigureManagerQT._toolbar2_class = FullToolbar
except Exception:  # pragma: no cover - Qt not available
    FigureCanvas = None
    NavigationToolbar2QT = None
    FullToolbar = None
    figureoptions = None

try:  # pragma: no cover - depends on environment
    from mongoengine import connect
except Exception:  # pragma: no cover - executed when mongoengine missing
    connect = None
# ---------------------------------------------------------------------------
# Database helper (robust connector)
# ---------------------------------------------------------------------------
from .db import connect_with_fallback as _connect_with_fallback


def connect_to_database(db_name="battery_test_db", host="localhost", port=27017):
    """
    Connect to MongoDB database.

    Args:
        db_name: Name of the database
        host: MongoDB host address
        port: MongoDB port

    Returns:
        bool: True if connection successful, False otherwise
    """
    if connect is None:
        logging.warning("mongoengine not available; skipping DB connection")
        return False
    try:
        # Attempt to connect to the database
        conn = connect(db_name, host=host, port=port)

        # Test connection by accessing server info
        server_info = conn.server_info()

        logging.info(f"Connected to MongoDB {db_name} at {host}:{port}")
        logging.info(f"MongoDB version: {server_info.get('version', 'unknown')}")

        return True
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {str(e)}")
        return False


def debug_connection_status():
    """Print MongoDB connection status for debugging."""
    try:
        from mongoengine.connection import get_connection, get_db

        conn = get_connection()
        if conn:
            print(f"MongoDB connection exists: {conn}")
            db = get_db()
            print(f"Current database: {db.name}")
            return True
        else:
            print("No MongoDB connection exists")
            return False
    except Exception as e:
        print(f"Error checking MongoDB connection: {str(e)}")
        return False


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def get_sample_name(sample):
    """Return the name of ``sample`` regardless of lazy references."""
    try:
        if hasattr(sample, "fetch"):
            sample = sample.fetch()
        return sample.name
    except Exception:
        return getattr(sample, "name", "Unknown")


# Create other utility functions as needed
def get_file_list(directory):
    """
    Get list of supported data files in a directory.

    Args:
        directory: Path to directory to search

    Returns:
        list: List of file paths
    """
    import os
    from battery_analysis.parsers import get_supported_formats

    formats = get_supported_formats()
    files = []

    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in formats:
                files.append(os.path.join(root, filename))

    return files


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def popout_figure(fig):
    """Open a copy of ``fig`` in a standalone matplotlib window.

    Parameters
    ----------
    fig : :class:`matplotlib.figure.Figure`
        Figure embedded in the GUI.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The popped out figure instance. Keeping a reference to this
        object prevents the window from being garbage collected.
    """

    fig_copy = pickle.loads(pickle.dumps(fig))

    # ``Figure.show`` requires a canvas manager registered with pyplot.
    new_canvas = plt.figure().canvas
    plt.close(new_canvas.figure)  # close the empty placeholder figure
    fig_copy.set_canvas(new_canvas)

    import matplotlib._pylab_helpers as pylab_helpers

    pylab_helpers.Gcf.set_active(fig_copy.canvas.manager)
    # ``Figure.show`` is sufficient to display the new window in the
    # context of the running Tk mainloop. Calling ``plt.show`` here would
    # trigger Matplotlib to attempt to manage *all* existing figures and
    # recreate the Tk application after it has been closed, which caused
    # the ``TclError`` when repeatedly popping out plots.  By avoiding
    # ``plt.show`` we only display the requested figure and allow the
    # function to be called multiple times.
    fig_copy.show()

    # Ensure popped out window uses the full toolbar with a Configure button
    try:
        if FullToolbar is not None:
            fig_manager = fig_copy.canvas.manager
            if not isinstance(fig_manager.toolbar, FullToolbar):
                fig_manager.window.removeToolBar(fig_manager.toolbar)
                fig_manager.toolbar = FullToolbar(fig_copy.canvas, fig_manager.window)
                fig_manager.window.addToolBar(fig_manager.toolbar)
    except Exception:
        pass

    return fig_copy
