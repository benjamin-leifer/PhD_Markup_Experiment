from typing import Callable, Tuple, cast

import dash
import dash_bootstrap_components as dbc

from dashboard import auth


def _get_login_callback() -> Callable[[int, str, str], Tuple[object, object]]:
    app = dash.Dash(__name__)
    auth.register_callbacks(app)
    key = next(iter(app.callback_map))
    cb = app.callback_map[key]["callback"].__wrapped__
    return cast(Callable[[int, str, str], Tuple[object, object]], cb)


def test_login_success() -> None:
    handle_login = _get_login_callback()
    role, message = handle_login(1, "admin", "admin")
    assert role == "admin"
    assert message == ""


def test_login_failure() -> None:
    handle_login = _get_login_callback()
    role, message = handle_login(1, "admin", "wrong")
    assert role is dash.no_update
    assert isinstance(message, dbc.Alert)
