import datetime
from mongoengine import Document, fields


class ImportJob(Document):
    """Record of a directory import run."""

    start_time = fields.DateTimeField(default=datetime.datetime.utcnow)
    end_time = fields.DateTimeField()
    files = fields.ListField(fields.DictField(), default=list)
    errors = fields.ListField(fields.StringField(), default=list)
    current_file = fields.StringField()
    processed_count = fields.IntField(default=0)
    total_count = fields.IntField(default=0)

    meta = {"collection": "import_jobs", "indexes": ["start_time"]}


class ImportJobSummary(Document):
    """Lightweight summary of an import job run."""

    start_time = fields.DateTimeField(default=datetime.datetime.utcnow)
    end_time = fields.DateTimeField()
    total_count = fields.IntField(default=0)
    processed_count = fields.IntField(default=0)
    created_count = fields.IntField(default=0)
    updated_count = fields.IntField(default=0)
    skipped_count = fields.IntField(default=0)
    errors = fields.ListField(fields.StringField(), default=list)
    status = fields.StringField(default="running")

    meta = {"collection": "import_job_summaries", "indexes": ["start_time"]}
