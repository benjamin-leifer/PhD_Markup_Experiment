# battery_analysis/models/stages.py

"""Processing stage models and metadata inheritance helpers.

This module defines document models representing the manufacturing stages of a
battery cathode. Each stage references the previous stage via a ``parent``
field. The :func:`inherit_metadata` helper walks up this chain to merge
``metadata`` dictionaries so that child objects automatically inherit relevant
attributes from their ancestors.
"""

from __future__ import annotations

import datetime
from mongoengine import Document, fields


def inherit_metadata(obj):
    """Recursively merge ``metadata`` dictionaries from ``obj`` and parents.

    Parameters
    ----------
    obj: Any
        The object whose metadata should be resolved. The object must have a
        ``metadata`` attribute and may optionally define a ``parent`` attribute
        pointing to the previous processing stage.

    Returns
    -------
    dict
        A dictionary containing the merged metadata where child values override
        parent values.
    """

    merged: dict[str, object] = {}
    current = obj
    while current is not None:
        data = getattr(current, "metadata", {}) or {}
        merged = {**data, **merged}
        parent = getattr(current, "parent", None)
        if hasattr(parent, "fetch"):
            parent = parent.fetch()
        current = parent
    return merged


class CathodeMaterial(Document):
    """Base material used to create a cathode slurry."""

    name = fields.StringField(required=True, unique=True)
    composition = fields.StringField(required=False)
    manufacturer = fields.StringField(required=False)
    metadata = fields.DictField(default=dict)
    parent = fields.ReferenceField("self", required=False)

    created_at = fields.DateTimeField(default=datetime.datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.datetime.utcnow)

    meta = {"collection": "cathode_materials"}

    def clean(self):  # pragma: no cover - simple assignment
        self.updated_at = datetime.datetime.utcnow()
        self.metadata = inherit_metadata(self)
        super().clean()

    @classmethod
    def from_parent(cls, parent: CathodeMaterial | None = None, **kwargs):
        obj = cls(parent=parent, **kwargs)
        obj.metadata = inherit_metadata(obj)
        return obj


class Slurry(Document):
    """Slurry derived from a :class:`CathodeMaterial`."""

    parent = fields.ReferenceField("CathodeMaterial", required=True)
    solids_content = fields.FloatField(required=False)
    mixing_time = fields.FloatField(required=False)
    metadata = fields.DictField(default=dict)

    created_at = fields.DateTimeField(default=datetime.datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.datetime.utcnow)

    meta = {"collection": "slurries"}

    def clean(self):  # pragma: no cover - simple assignment
        self.updated_at = datetime.datetime.utcnow()
        self.metadata = inherit_metadata(self)
        super().clean()

    @classmethod
    def from_parent(cls, parent: CathodeMaterial, **kwargs):
        obj = cls(parent=parent, **kwargs)
        obj.metadata = inherit_metadata(obj)
        return obj


class Electrode(Document):
    """Electrode produced from a :class:`Slurry`."""

    parent = fields.ReferenceField("Slurry", required=True)
    loading = fields.FloatField(required=False)
    thickness = fields.FloatField(required=False)
    metadata = fields.DictField(default=dict)

    created_at = fields.DateTimeField(default=datetime.datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.datetime.utcnow)

    meta = {"collection": "electrodes"}

    def clean(self):  # pragma: no cover - simple assignment
        self.updated_at = datetime.datetime.utcnow()
        self.metadata = inherit_metadata(self)
        super().clean()

    @classmethod
    def from_parent(cls, parent: Slurry, **kwargs):
        obj = cls(parent=parent, **kwargs)
        obj.metadata = inherit_metadata(obj)
        return obj


class Cell(Document):
    """Final cell composed using an :class:`Electrode`."""

    parent = fields.ReferenceField("Electrode", required=True)
    format = fields.StringField(required=False)
    nominal_capacity = fields.FloatField(required=False)
    metadata = fields.DictField(default=dict)

    created_at = fields.DateTimeField(default=datetime.datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.datetime.utcnow)

    meta = {"collection": "cells"}

    def clean(self):  # pragma: no cover - simple assignment
        self.updated_at = datetime.datetime.utcnow()
        self.metadata = inherit_metadata(self)
        super().clean()

    @classmethod
    def from_parent(cls, parent: Electrode, **kwargs):
        obj = cls(parent=parent, **kwargs)
        obj.metadata = inherit_metadata(obj)
        return obj
