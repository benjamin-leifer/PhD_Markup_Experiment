# Battery Analysis API

A lightweight FastAPI application provides programmatic access to common
operations.  Authentication is handled via bearer tokens that map to user
roles defined in `dashboard/users.json`.

## Authentication

Include an `Authorization` header using the token assigned to your user.

```
Authorization: Bearer <token>
```

The repository ships with two example tokens:

| User   | Role   | Token         |
|--------|--------|---------------|
| admin  | admin  | `admin-token` |
| viewer | viewer | `viewer-token`|

## Endpoints

### `GET /tests`

Return a JSON list of recently imported tests.  Requires either `admin` or
`viewer` role.  When no database connection is available an empty list is
returned.

```
GET /tests
Authorization: Bearer viewer-token
```

Response:

```json
{"status": "ok", "tests": []}
```

### `POST /import`

Trigger directory import using :func:`battery_analysis.utils.import_directory`.
Only `admin` users may access this endpoint.

Request body:

```json
{"path": "/data/tests"}
```

The import runs in dry‑run mode for safety and returns the utility's exit
code.

The underlying command-line tool also tracks import jobs. Use ``--status`` to
list existing jobs and ``--resume <JOB_ID>`` to continue an interrupted run.

### `POST /doe-plans`

Create a simple design‑of‑experiments plan by calling
:func:`battery_analysis.utils.doe_builder.save_plan`.  Only `admin` users may
access this endpoint.

Request body:

```json
{
  "name": "example",
  "factors": {"temp": [25, 40], "rate": [1, 2]}
}
```

Response includes the generated plan matrix.
