import os
import sys
import pytest

TESTS_DIR = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.abspath(os.path.join(TESTS_DIR, '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from battery_analysis import models


def test_sample_components_creation():
    anode = models.Sample(name="Anode")
    cathode = models.Sample(name="Cathode")
    cell = models.Sample(name="Cell", anode=anode, cathode=cathode)
    assert hasattr(cell, "anode")
    assert hasattr(cell, "cathode")


def test_record_missing_components():
    import importlib
    import matplotlib
    import types
    import sys
    orig_use = matplotlib.use
    matplotlib.use = lambda *a, **k: None
    sys.modules.setdefault("networkx", types.ModuleType("networkx"))
    DataUploadTab = importlib.import_module("battery_analysis.gui.app").DataUploadTab
    sys.modules.pop("networkx", None)
    matplotlib.use = orig_use

    class DummyApp:
        def __init__(self):
            self.missing_data = []

    dummy = DummyApp()
    tab = DataUploadTab.__new__(DataUploadTab)
    tab.main_app = dummy
    sample = models.Sample(name="Cell")
    test = models.TestResult()
    tab._record_missing_components(sample, test)
    assert dummy.missing_data and dummy.missing_data[0]["missing"] == [
        "anode",
        "cathode",
        "separator",
        "electrolyte",
    ]


def test_missing_data_tab_listings():
    import tkinter as tk
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tk not available")
    from battery_analysis.gui.missing_data_tab import MissingDataTab

    class DummyApp:
        def __init__(self):
            self.missing_data = [{"test_id": "T1", "missing": ["anode"]}]

    app = DummyApp()
    tab = MissingDataTab(root, app)
    tab.refresh_table()
    items = tab.tree.get_children()
    assert len(items) == 1
    vals = tab.tree.item(items[0])["values"]
    assert vals[0] == "T1"
    root.destroy()
