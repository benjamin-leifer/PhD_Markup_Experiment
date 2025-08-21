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
