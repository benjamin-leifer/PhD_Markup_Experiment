from __future__ import annotations

from typing import Any, Callable, Dict, List, TypedDict

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from dashboard.auth import load_users
from battery_analysis.utils.import_directory import import_directory
from battery_analysis.utils.doe_builder import save_plan

app = FastAPI()

security = HTTPBearer()


class _UserRecord(TypedDict, total=False):
    password_hash: str
    role: str
    api_token: str


def _get_role(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    token = credentials.credentials
    users: Dict[str, _UserRecord] = load_users()
    for user in users.values():
        if user.get("api_token") == token:
            return user["role"]
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")


def require_role(*allowed: str) -> Callable[[str], str]:
    def _checker(role: str = Depends(_get_role)) -> str:
        if role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return role

    return _checker


class ImportRequest(BaseModel):  # type: ignore[misc]
    path: str


@app.post("/import")  # type: ignore[misc]
def import_endpoint(
    req: ImportRequest, role: str = Depends(require_role("admin"))
) -> Dict[str, Any]:
    code = import_directory(req.path, dry_run=True)
    return {"status": "ok", "code": code}


@app.get("/tests")  # type: ignore[misc]
def list_tests(role: str = Depends(require_role("admin", "viewer"))) -> Dict[str, Any]:
    tests: List[Dict[str, Any]] = []
    try:
        from battery_analysis.models import TestResult

        for test in TestResult.objects[:50]:
            tests.append({"id": str(test.id), "name": getattr(test, "name", "")})
    except Exception:
        pass
    return {"status": "ok", "tests": tests}


class DoeRequest(BaseModel):  # type: ignore[misc]
    name: str
    factors: Dict[str, List[Any]]


@app.post("/doe-plans")  # type: ignore[misc]
def doe_plans(
    req: DoeRequest, role: str = Depends(require_role("admin"))
) -> Dict[str, Any]:
    plan = save_plan(req.name, req.factors)
    return {"status": "ok", "plan": {"name": plan.name, "matrix": plan.matrix}}
