"""MongoEngine model tracking refactor runs."""

# mypy: ignore-errors

import datetime

from mongoengine import Document, fields


class RefactorJob(Document):
    """Record of a TestResult refactor run."""

    start_time = fields.DateTimeField(default=datetime.datetime.utcnow)
    end_time = fields.DateTimeField()
    filter = fields.DictField(default=dict)
    dry_run = fields.BooleanField(default=False)
    processed = fields.IntField(default=0)
    updated = fields.IntField(default=0)
    errors = fields.ListField(fields.StringField(), default=list)
    status = fields.StringField(default="running")

    meta = {"collection": "refactor_jobs", "indexes": ["start_time"]}
