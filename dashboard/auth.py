"""Simple authentication helpers for the Dash dashboard."""

from __future__ import annotations

import dash
from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc

# In-memory user database for demonstration purposes
USERS: dict[str, dict[str, str]] = {
    "admin": {"password": "admin", "role": "admin"},
    "viewer": {"password": "viewer", "role": "viewer"},
}


def layout() -> html.Div:
    """Return a basic login form."""
    return dbc.Container(
        [
            dbc.Row(dbc.Col(html.H2("Login"), width=12)),
            dbc.Row(
                dbc.Col(
                    dcc.Input(
                        id="login-username", placeholder="Username", type="text"
                    ),
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
    )
    def handle_login(n_clicks, username, password):
        user = USERS.get(username or "")
        if user and user["password"] == (password or ""):
            return user["role"], ""
        return dash.no_update, dbc.Alert("Invalid credentials", color="danger")
