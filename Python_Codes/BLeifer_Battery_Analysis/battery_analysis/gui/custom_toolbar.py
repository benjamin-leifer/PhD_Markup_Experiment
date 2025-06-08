"""Custom navigation toolbar with figure edit button."""

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from tkinter import messagebox

try:
    from matplotlib.backends.qt_editor import figureoptions
    QT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    QT_AVAILABLE = False


class CustomToolbar(NavigationToolbar2Tk):
    """Navigation toolbar that exposes Matplotlib's figure options."""

    toolitems = NavigationToolbar2Tk.toolitems + (
        ("Edit", "Edit axis/curve/image properties", "subplots", "edit_figure"),
    )

    def edit_figure(self):
        """Open the Qt figure options dialog if available."""
        if not QT_AVAILABLE:
            messagebox.showinfo(
                "Qt Not Available",
                "Install PyQt5 or PySide2 to enable figure editing.",
            )
            return
        try:
            figureoptions.figure_edit(self.canvas.figure)
        except Exception as err:  # pragma: no cover - GUI errors
            messagebox.showerror("Edit Failed", str(err))
