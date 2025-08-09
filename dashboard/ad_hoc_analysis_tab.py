"""Ad hoc analysis tab allowing dynamic loading of user scripts."""

from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path

import dash
from dash import Input, Output, State, dcc, html

UPLOAD = "adhoc-upload"
SCRIPT_SELECT = "adhoc-script-select"
OUTPUT = "adhoc-output"
SCRIPTS_DIR = Path(__file__).parent / "adhoc_scripts"


def _script_options():
    """Return dropdown options for available scripts."""
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        {"label": p.stem, "value": p.stem}
        for p in sorted(SCRIPTS_DIR.glob("*.py"))
    ]


def layout() -> html.Div:
    """Return layout for the ad hoc analysis tab."""
    return html.Div(
        [
            dcc.Upload(id=UPLOAD, children=html.Button("Upload Script")),
            dcc.Dropdown(
                id=SCRIPT_SELECT,
                options=_script_options(),
                placeholder="Select a script",
                className="mt-2",
            ),
            html.Div(id=OUTPUT, className="mt-3"),
        ]
    )


def register_callbacks(app: dash.Dash) -> None:
    """Register callbacks for the ad hoc analysis tab."""

    @app.callback(
        Output(SCRIPT_SELECT, "options"),
        Input(UPLOAD, "contents"),
        State(UPLOAD, "filename"),
        prevent_initial_call=True,
    )
    def save_upload(contents, filename):
        if not contents or not filename:
            raise dash.exceptions.PreventUpdate
        data = base64.b64decode(contents.split(",", 1)[1])
        path = SCRIPTS_DIR / filename
        path.write_bytes(data)
        return _script_options()

    @app.callback(Output(OUTPUT, "children"), Input(SCRIPT_SELECT, "value"))
    def render_script(name):
        if not name:
            return html.Div("Select a script to run.")
        try:
            file_path = SCRIPTS_DIR / f"{name}.py"
            spec = importlib.util.spec_from_file_location(name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            assert spec.loader
            spec.loader.exec_module(module)
            if hasattr(module, "layout"):
                return module.layout()
            return html.Div(f"{name} has no layout() function")
        except Exception as err:  # pragma: no cover - simple error handling
            return html.Div(f"Error loading {name}: {err}")


__all__ = ["layout", "register_callbacks"]
