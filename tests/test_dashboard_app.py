"""Integration tests for the dashboard Dash app."""

from pathlib import Path
import sys
import pytest
import dash
import base64

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TAB_LABELS = [
    "Overview",
    "New Material",
    "Data Import",
    "Export",
    "Comparison",
    "Advanced Analysis",
    "Cycle Detail",
    "EIS",
    "Document Status",
    "Missing Data",
    "Trait Filter",
    "Flags",
]


def test_all_tabs_render():
    """The app layout renders and includes all tab labels for an admin user."""
    create_app = pytest.importorskip("dashboard.app").create_app
    app = create_app()
    render = app.callback_map["page-content.children"]["callback"].__wrapped__
    layout = render("admin")
    html_str = str(layout)
    for label in TAB_LABELS:
        assert label in html_str


def test_basic_callbacks(monkeypatch):
    """Each tab's primary callback executes without error."""
    create_app = pytest.importorskip("dashboard.app").create_app
    app = create_app()
    cb = app.callback_map

    toggle = cb["export-modal.is_open"]["callback"].__wrapped__
    assert toggle(1, None, False) is True

    aa_key = next(k for k in cb if "dqdv-options.style" in k)
    toggle_opts = cb[aa_key]["callback"].__wrapped__
    styles = toggle_opts("dqdv")
    assert styles[0] == {"display": "block"}

    update_tests = cb["cd-test.options"]["callback"].__wrapped__

    from battery_analysis.models import Sample, TestResult
    import mongomock
    from mongoengine import connect, disconnect

    disconnect()
    connect("testdb", mongo_client_class=mongomock.MongoClient, alias="default")
    sample = Sample(name="S1").save()
    test = TestResult(sample=sample, tester="Arbin", name="T1").save()
    import dashboard.data_access as data_access

    monkeypatch.setattr(data_access, "get_cell_dataset", lambda _code: None)

    options = update_tests(str(sample.id))
    assert options == [{"label": test.name, "value": str(test.id)}]
    disconnect()

    update_graph = cb["compare-graph.figure"]["callback"].__wrapped__
    fig = update_graph(None, "capacity")
    assert fig.data == ()

    eis_key = next(k for k in cb if "eis-file-div.style" in k)
    toggle_source = cb[eis_key]["callback"].__wrapped__
    file_style, db_style = toggle_source("file")
    assert file_style["display"] == "block" and db_style["display"] == "none"

    render_table = cb["missing-data-table.data"]["callback"].__wrapped__
    rows = render_table([{"test_id": "A", "cell_code": "C_A", "missing": ["cathode"]}])
    assert rows[0]["missing"] == "cathode" and rows[0]["cell_code"] == "C_A"

    update_results = cb["trait-results.children"]["callback"].__wrapped__
    result = update_results(1, None, None, None, None, None, None, None, None, None, [])
    assert result is not None

    db_status = cb["db-status.children"]["callback"].__wrapped__
    status_text = db_status(0)
    assert "Database:" in status_text


def test_upload_progress_message():
    """Progress message shows during file parsing."""
    create_app = pytest.importorskip("dashboard.app").create_app
    app = create_app()
    cb = app.callback_map
    progress_key = next(
        k
        for k, v in cb.items()
        if any(
            i["id"] == "upload-data" and i["property"] == "loading_state"
            for i in v["inputs"]
        )
    )
    show_progress = cb[progress_key]["callback"].__wrapped__
    assert show_progress({"is_loading": True}) == "Parsing file..."
    assert show_progress({"is_loading": False}) is dash.no_update


def test_handle_upload_clears_status(monkeypatch, tmp_path):
    """Upload handler clears progress message on success."""
    create_app = pytest.importorskip("dashboard.app").create_app
    app = create_app()
    cb = app.callback_map
    handle_key = next(k for k in cb if "upload-form.style" in k)
    handle_upload = cb[handle_key]["callback"].__wrapped__
    import dashboard.app as appmod

    monkeypatch.setattr(
        appmod.data_access,
        "store_temp_upload",
        lambda filename, data: str(tmp_path / filename),
    )
    monkeypatch.setattr(appmod, "parse_file", lambda path: ([], {}))
    content = "data:text/plain;base64," + base64.b64encode(b"hi").decode()
    result = handle_upload(content, "x.txt")
    assert result[-1] == ""


def test_missing_data_resolve_flow(monkeypatch):
    """Resolving a row removes it from the missing data dataset."""
    import importlib.util
    import types
    import sys
    from pathlib import Path

    # Load module directly to avoid importing the full dashboard package
    spec = importlib.util.spec_from_file_location(
        "missing_data_tab", Path(ROOT, "dashboard", "missing_data_tab.py")
    )
    missing_data_tab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(missing_data_tab)

    app = dash.Dash(__name__)
    app.layout = missing_data_tab.layout()
    missing_data_tab.register_callbacks(app)
    cb = app.callback_map

    render_table = cb["missing-data-table.data"]["callback"].__wrapped__
    data = [{"test_id": "T1", "cell_code": "C1", "missing": ["anode", "cathode"]}]
    rows = render_table(data)
    assert rows == [
        {
            "test_id": "T1",
            "cell_code": "C1",
            "missing": "anode, cathode",
            "resolve": "[Resolve](#)",
        }
    ]

    open_key = next(
        k
        for k in cb
        if "missing-data-modal.is_open" in k and "missing-data-store" not in k
    )
    open_modal = cb[open_key]["callback"].__wrapped__
    is_open, selected, body = open_modal({"row": 0, "column_id": "resolve"}, rows)
    assert is_open and selected["test_id"] == "T1" and selected["cell_code"] == "C1"
    assert body[0].children == "Cell Code / Filename: C1"

    class _Component:
        def __init__(self, name=None):
            self.name = name

        @classmethod
        def objects(cls, name):
            class Q:
                def first(self_inner):
                    return None

            return Q()

        def save(self):
            pass

    class _Sample:
        def __init__(self):
            self.anode = self.cathode = self.separator = self.electrolyte = None

        def save(self):
            pass

    class _Test:
        def __init__(self, id):
            self.id = id
            self.sample = _Sample()

        @classmethod
        def objects(cls, id):
            class Q:
                def first(self_inner):
                    return _Test(id)

            return Q()

    models = types.SimpleNamespace(
        TestResult=_Test,
        Anode=_Component,
        Cathode=_Component,
        Separator=_Component,
        Electrolyte=_Component,
    )
    fake_ba = types.ModuleType("battery_analysis")
    fake_ba.models = models
    monkeypatch.setitem(sys.modules, "battery_analysis", fake_ba)
    monkeypatch.setitem(sys.modules, "battery_analysis.models", models)

    resolve_key = next(
        k
        for k, v in cb.items()
        if "missing-data-store.data" in k
        and any(i["id"] == "confirm-resolve" for i in v["inputs"])
    )
    resolve = cb[resolve_key]["callback"].__wrapped__
    new_data, modal_open, _body2 = resolve(1, selected, data, ["A1", "C1"])
    assert new_data == [] and modal_open is False


def test_missing_data_suggestions(monkeypatch):
    """Suggested values pre-populate fields and confirming them resolves the row."""
    import importlib.util
    import types
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "missing_data_tab", Path(ROOT, "dashboard", "missing_data_tab.py")
    )
    missing_data_tab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(missing_data_tab)

    monkeypatch.setattr(
        missing_data_tab,
        "_suggest_values",
        lambda _tid, _missing: {"anode": "A1", "cathode": "C1"},
    )

    app = dash.Dash(__name__)
    app.layout = missing_data_tab.layout()
    missing_data_tab.register_callbacks(app)
    cb = app.callback_map

    render_table = cb["missing-data-table.data"]["callback"].__wrapped__
    data = [{"test_id": "T1", "cell_code": "C1", "missing": ["anode", "cathode"]}]
    rows = render_table(data)

    open_key = next(
        k
        for k in cb
        if "missing-data-modal.is_open" in k and "missing-data-store" not in k
    )
    open_modal = cb[open_key]["callback"].__wrapped__
    is_open, selected, body = open_modal({"row": 0, "column_id": "resolve"}, rows)
    assert (
        is_open
        and selected["cell_code"] == "C1"
        and selected["suggestions"]
        == {
            "anode": "A1",
            "cathode": "C1",
        }
    )
    assert body[0].children == "Cell Code / Filename: C1"
    from dash import html

    inputs = [c.children[1] for c in body if isinstance(c, html.Div)]
    assert [i.value for i in inputs] == ["A1", "C1"]

    class _Component:
        def __init__(self, name=None):
            self.name = name

        @classmethod
        def objects(cls, name):
            class Q:
                def first(self_inner):
                    return None

            return Q()

        def save(self):
            pass

    class _Sample:
        def __init__(self):
            self.anode = self.cathode = self.separator = self.electrolyte = None

        def save(self):
            pass

    class _Test:
        def __init__(self, id):
            self.id = id
            self.sample = _Sample()

        @classmethod
        def objects(cls, id):
            class Q:
                def first(self_inner):
                    return _Test(id)

            return Q()

    models = types.SimpleNamespace(
        TestResult=_Test,
        Anode=_Component,
        Cathode=_Component,
        Separator=_Component,
        Electrolyte=_Component,
        Sample=_Sample,
    )
    fake_ba = types.ModuleType("battery_analysis")
    fake_ba.models = models
    monkeypatch.setitem(sys.modules, "battery_analysis", fake_ba)
    monkeypatch.setitem(sys.modules, "battery_analysis.models", models)

    resolve_key = next(
        k
        for k, v in cb.items()
        if "missing-data-store.data" in k
        and any(i["id"] == "confirm-resolve" for i in v["inputs"])
    )
    resolve = cb[resolve_key]["callback"].__wrapped__
    new_data, modal_open, _body2 = resolve(1, selected, data, ["A1", "C1"])
    assert new_data == [] and modal_open is False


def test_missing_data_lazy_load(monkeypatch):
    """_get_missing_data executes only when the tab is opened and filters results."""
    import importlib.util
    from pathlib import Path
    import dash

    spec = importlib.util.spec_from_file_location(
        "missing_data_tab", Path(ROOT, "dashboard", "missing_data_tab.py")
    )
    missing_data_tab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(missing_data_tab)

    records = [
        {"test_id": "1", "cell_code": "C1", "missing": ["anode"]},
        {"test_id": "2", "cell_code": "C2", "missing": []},
    ]

    calls = {"n": 0}

    def wrapped():
        calls["n"] += 1
        return [r for r in records if r["missing"]]

    monkeypatch.setattr(missing_data_tab, "_get_missing_data", wrapped)

    app = dash.Dash(__name__)
    app.layout = missing_data_tab.layout()
    missing_data_tab.register_callbacks(app)

    assert calls["n"] == 0

    cb = app.callback_map
    load_key = next(
        k
        for k, v in cb.items()
        if "missing-data-store.data" in k
        and any(i["id"] == "tabs" for i in v["inputs"])
    )
    load = cb[load_key]["callback"].__wrapped__

    result = load("missing-data", [])
    assert calls["n"] == 1
    assert result == [{"test_id": "1", "cell_code": "C1", "missing": ["anode"]}]

    again = load("missing-data", result)
    assert calls["n"] == 1 and again is dash.no_update


def test_missing_data_falls_back_to_test_name(monkeypatch):
    """Records lacking a cell code use the test's filename instead."""
    import importlib.util
    import types
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "missing_data_tab", Path(ROOT, "dashboard", "missing_data_tab.py")
    )
    missing_data_tab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(missing_data_tab)

    class _Sample:
        anode = cathode = separator = electrolyte = None

    class _Test:
        id = "T1"
        cell_code = ""
        name = "file1.txt"
        sample = _Sample()

    class _Query:
        def only(self, *fields):
            return [_Test()]

    class _TestResult:
        @staticmethod
        def _get_collection():
            class _Coll:
                def create_index(self, _field):
                    pass

            return _Coll()

        @staticmethod
        def objects(__raw__):
            return _Query()

    models = types.SimpleNamespace(TestResult=_TestResult)
    fake_ba = types.ModuleType("battery_analysis")
    fake_ba.models = models
    monkeypatch.setitem(sys.modules, "battery_analysis", fake_ba)
    monkeypatch.setitem(sys.modules, "battery_analysis.models", models)

    records = missing_data_tab._get_missing_data()
    assert records[0]["cell_code"] == "file1.txt"


def test_export_plot_prompts_kaleido(monkeypatch):
    """Missing kaleido triggers a toast notification when exporting plots."""
    import types
    import importlib.util
    import sys
    from pathlib import Path
    import dash
    import plotly.graph_objs as go

    root = Path(__file__).resolve().parents[1]
    pkg = types.ModuleType("dashboard")
    pkg.__path__ = [str(root / "dashboard")]
    sys.modules["dashboard"] = pkg

    spec = importlib.util.spec_from_file_location(
        "dashboard.comparison_tab", root / "dashboard" / "comparison_tab.py"
    )
    comparison_tab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison_tab)

    app = dash.Dash(__name__)
    app.layout = comparison_tab.layout()
    comparison_tab.register_callbacks(app)

    export_key = next(
        k for k in app.callback_map if "compare-export-img-download.data" in k
    )
    export_cb = app.callback_map[export_key]["callback"].__wrapped__

    def boom(self, *args, **kwargs):
        raise ValueError("no kaleido")

    monkeypatch.setattr(go.Figure, "write_image", boom)

    result = export_cb(1, go.Figure().to_dict())
    assert result[0] is dash.no_update
    assert result[1] is True and "kaleido" in result[2].lower()


def test_missing_data_tab_shows_filename_for_empty_cell_code(monkeypatch):
    """A test without a cell code displays its filename in the table."""
    import importlib.util
    import types
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "missing_data_tab", Path(ROOT, "dashboard", "missing_data_tab.py")
    )
    missing_data_tab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(missing_data_tab)

    class _Sample:
        anode = cathode = separator = electrolyte = None

    class _Test:
        id = "T1"
        cell_code = ""
        name = "file1.txt"
        sample = _Sample()

    class _Query:
        def only(self, *fields):
            return [_Test()]

    class _TestResult:
        @staticmethod
        def _get_collection():
            class _Coll:
                def create_index(self, _field):
                    pass

            return _Coll()

        @staticmethod
        def objects(__raw__):
            return _Query()

    models = types.SimpleNamespace(TestResult=_TestResult)
    fake_ba = types.ModuleType("battery_analysis")
    fake_ba.models = models
    monkeypatch.setitem(sys.modules, "battery_analysis", fake_ba)
    monkeypatch.setitem(sys.modules, "battery_analysis.models", models)

    app = dash.Dash(__name__)
    app.layout = missing_data_tab.layout()
    missing_data_tab.register_callbacks(app)

    records = missing_data_tab._get_missing_data()
    render_table = app.callback_map["missing-data-table.data"]["callback"].__wrapped__
    rows = render_table(records)

    assert rows == [
        {
            "test_id": "T1",
            "cell_code": "file1.txt",
            "missing": "anode, cathode, separator, electrolyte",
            "resolve": "[Resolve](#)",
        }
    ]
