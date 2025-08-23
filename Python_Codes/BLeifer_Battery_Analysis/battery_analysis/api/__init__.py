from __future__ import annotations

from typing import Any, Callable, Dict, List, TypedDict, cast

from concurrent.futures import ThreadPoolExecutor
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from dashboard.auth import load_users
from battery_analysis.utils.import_directory import import_directory
from battery_analysis.utils.doe_builder import save_plan
from battery_analysis.models import ImportJob, ImportJobSummary
from battery_analysis.utils import file_storage
from fastapi.responses import StreamingResponse

app = FastAPI()

security = HTTPBearer()

# Simple thread pool for background import tasks
executor = ThreadPoolExecutor(max_workers=2)


class _UserRecord(TypedDict, total=False):
    password_hash: str
    role: str
    api_token: str


def _get_role(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    token = credentials.credentials
    users: Dict[str, _UserRecord] = cast(Dict[str, _UserRecord], load_users())
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


@app.get("/raw/{file_id}")  # type: ignore[misc]
def get_raw_file(
    file_id: str, role: str = Depends(require_role("admin", "operator"))
) -> StreamingResponse:
    """Stream the raw file bytes for ``file_id``."""

    data = file_storage.retrieve_raw(file_id)
    return StreamingResponse(iter([data]), media_type="application/octet-stream")


class DoeRequest(BaseModel):  # type: ignore[misc]
    name: str
    factors: Dict[str, List[Any]]


@app.post("/doe-plans")  # type: ignore[misc]
def doe_plans(
    req: DoeRequest, role: str = Depends(require_role("admin"))
) -> Dict[str, Any]:
    plan = save_plan(req.name, req.factors)
    return {"status": "ok", "plan": {"name": plan.name, "matrix": plan.matrix}}


class ImportJobRequest(BaseModel):  # type: ignore[misc]
    path: str


@app.post("/import-jobs")  # type: ignore[misc]
def create_import_job(
    req: ImportJobRequest, role: str = Depends(require_role("admin"))
) -> Dict[str, str]:
    job = ImportJob().save()
    # Run the import asynchronously, resuming the created job
    executor.submit(import_directory, req.path, resume=str(job.id))
    return {"status": "queued", "job_id": str(job.id)}


@app.get("/import-jobs")  # type: ignore[misc]
def list_import_jobs(
    role: str = Depends(require_role("admin", "viewer"))
) -> Dict[str, Any]:
    jobs: List[Dict[str, Any]] = []
    try:
        for job in ImportJob.objects.order_by("-start_time"):
            jobs.append(
                {
                    "id": str(job.id),
                    "start_time": getattr(job.start_time, "isoformat", lambda: None)(),
                    "end_time": getattr(job.end_time, "isoformat", lambda: None)(),
                    "processed_count": getattr(job, "processed_count", 0),
                    "errors": list(getattr(job, "errors", [])),
                }
            )
    except Exception:
        pass
    return {"status": "ok", "jobs": jobs}


@app.get("/import-jobs/{job_id}")  # type: ignore[misc]
def get_import_job(
    job_id: str, role: str = Depends(require_role("admin", "viewer"))
) -> Dict[str, Any]:
    job = ImportJob.objects(id=job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
        )
    return {
        "status": "ok",
        "job": {
            "start_time": getattr(job.start_time, "isoformat", lambda: None)(),
            "processed_count": getattr(job, "processed_count", 0),
            "errors": list(getattr(job, "errors", [])),
        },
    }


@app.get("/import-job-summaries")  # type: ignore[misc]
def list_import_job_summaries(
    role: str = Depends(require_role("admin", "viewer"))
) -> Dict[str, Any]:
    jobs: List[Dict[str, Any]] = []
    try:
        for s in ImportJobSummary.objects.order_by("-start_time"):
            jobs.append(
                {
                    "id": str(s.id),
                    "start_time": getattr(s.start_time, "isoformat", lambda: None)(),
                    "end_time": getattr(s.end_time, "isoformat", lambda: None)(),
                    "created": getattr(s, "created_count", 0),
                    "updated": getattr(s, "updated_count", 0),
                    "skipped": getattr(s, "skipped_count", 0),
                    "status": getattr(s, "status", ""),
                    "errors": list(getattr(s, "errors", [])),
                }
            )
    except Exception:
        pass
    return {"status": "ok", "jobs": jobs}
