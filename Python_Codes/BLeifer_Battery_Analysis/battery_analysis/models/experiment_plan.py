"""MongoEngine model for storing design-of-experiments plans."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List

from mongoengine import Document, fields


class ExperimentPlan(Document):
    """Persisted experiment design made up of factor combinations."""

    name = fields.StringField(required=True, unique=True)
    factors = fields.DictField(required=True)
    matrix = fields.ListField(fields.DictField(), required=True)
    sample_ids = fields.ListField(fields.ReferenceField("Sample"), default=list)
    created_at = fields.DateTimeField(default=datetime.datetime.utcnow)

    meta = {"collection": "experiment_plans", "indexes": ["name"]}

    @classmethod
    def get_by_name(cls, name: str) -> "ExperimentPlan | None":
        """Return plan with ``name`` or ``None`` if missing."""
        return cls.objects(name=name).first()


__all__ = ["ExperimentPlan"]
