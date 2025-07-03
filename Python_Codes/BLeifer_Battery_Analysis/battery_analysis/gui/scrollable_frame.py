import tkinter as tk
from tkinter import ttk

class ScrollableFrame(ttk.Frame):
    """A frame with vertical and horizontal scrollbars."""

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)

        self.v_scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.h_scrollbar = ttk.Scrollbar(
            self, orient="horizontal", command=self.canvas.xview
        )
        self.canvas.configure(
            yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set
        )

        self.inner = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")

    def _on_inner_configure(self, event):
        """Update scroll region when the inner frame changes size."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if self.inner.winfo_reqwidth() < self.canvas.winfo_width():
            self.canvas.itemconfigure(self.canvas_window, width=self.canvas.winfo_width())

    def _on_canvas_configure(self, event):
        """Expand inner frame to fill available width if smaller."""
        if self.inner.winfo_reqwidth() < event.width:
            self.canvas.itemconfigure(self.canvas_window, width=event.width)
