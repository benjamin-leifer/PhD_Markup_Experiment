"""Simple authentication helpers for the Dash dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

import dash
from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc
import bcrypt


class UserRecord(TypedDict):
    password_hash: str
    role: str


# Path to the persistent user store
USERS_FILE = Path(__file__).with_name("users.json")


def load_users(path: str | Path = USERS_FILE) -> dict[str, UserRecord]:
    """Load user definitions from ``path``.

    The JSON file must map usernames to objects containing ``password_hash`` and
    ``role`` fields. Password hashes are stored using ``bcrypt".
    """

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return cast(dict[str, UserRecord], data)


def check_credentials(
    username: str, password: str, users: dict[str, dict[str, str]] | None = None
) -> str | None:
    """Validate ``username`` and ``password`` returning the user's role.

    Parameters
    ----------
    username, password:
        The submitted credentials.
    users:
        Optional pre-loaded user mapping for testing.

    Returns
    -------
    The user's role if authentication succeeds, otherwise ``None``.
    """

    user_map = users or load_users()
    user = user_map.get(username or "")
    if not user:
        return None
    stored = user.get("password_hash", "").encode()
    if bcrypt.checkpw((password or "").encode(), stored):
        return user.get("role")
    return None


def layout() -> html.Div:
    """Return a basic login form."""
    return dbc.Container(
        [
            dbc.Row(dbc.Col(html.H2("Login"), width=12)),
            dbc.Row(
                dbc.Col(
                    dcc.Input(id="login-username", placeholder="Username", type="text"),
                    md=4,
                ),
                className="mt-3",
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Input(
                        id="login-password", placeholder="Password", type="password"
                    ),
                    md=4,
                ),
                className="mt-2",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Button("Login", id="login-button", color="primary"),
                    md=2,
                    className="mt-3",
                )
            ),
            html.Div(id="login-message", className="mt-2"),
        ],
        className="mt-5",
    )


def register_callbacks(app: dash.Dash) -> None:
    """Register callbacks for handling authentication."""

    @app.callback(
        Output("user-role", "data"),
        Output("login-message", "children"),
        Input("login-button", "n_clicks"),
        State("login-username", "value"),
        State("login-password", "value"),
        prevent_initial_call=True,
    )  # type: ignore[misc]
    def handle_login(
        n_clicks: int, username: str | None, password: str | None
    ) -> tuple[object, object]:
        role = check_credentials(username or "", password or "")
        if role:
            return role, ""
        return dash.no_update, dbc.Alert("Invalid credentials", color="danger")
