"""MongoEngine model tracking refactor runs."""

# mypy: ignore-errors

import datetime

from mongoengine import Document, fields


class RefactorJob(Document):
    """Record of a :func:`refactor_tests` run."""

    start_time = fields.DateTimeField(default=datetime.datetime.utcnow)
    end_time = fields.DateTimeField()
    current_test = fields.StringField()
    processed_count = fields.IntField(default=0)
    total_count = fields.IntField(default=0)
    errors = fields.ListField(fields.StringField(), default=list)
    status = fields.StringField(default="running")

    meta = {"collection": "refactor_jobs", "indexes": ["start_time"]}
