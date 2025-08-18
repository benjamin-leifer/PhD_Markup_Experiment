# battery_analysis/models/sample.py

import datetime
from mongoengine import Document, fields, ReferenceField


class Sample(Document):
    """Represents a physical sample or cell.

    Use :meth:`add_note` to append timestamped notes for tracking::

        sample.add_note("checked performance", author="researcher")
    """

    name = fields.StringField(required=True, unique=True)
    chemistry = fields.StringField(required=False)
    manufacturer = fields.StringField(required=False)
    form_factor = fields.StringField(required=False)
    nominal_capacity = fields.FloatField(
        required=False, help_text="Nominal capacity in mAh"
    )
    parent = fields.ReferenceField("self", required=False)
    anode = fields.LazyReferenceField("self", required=False)
    cathode = fields.LazyReferenceField("self", required=False)
    separator = fields.LazyReferenceField("self", required=False)
    electrolyte = fields.LazyReferenceField("self", required=False)
    # Remove CASCADE to break circular dependency
    tests = fields.ListField(fields.LazyReferenceField("TestResult"))
    default_dataset = ReferenceField("CellDataset")

    created_at = fields.DateTimeField(default=datetime.datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.datetime.utcnow)
    tags = fields.ListField(fields.StringField(), required=False)
    description = fields.StringField(required=False)
    created_by = fields.StringField(required=False)
    last_modified_by = fields.StringField(required=False)
    notes_log = fields.ListField(
        fields.DictField(),
        default=list,
        help_text="List of timestamped note entries",
    )

    avg_initial_capacity = fields.FloatField(required=False)
    avg_final_capacity = fields.FloatField(required=False)
    avg_capacity_retention = fields.FloatField(required=False)
    avg_coulombic_eff = fields.FloatField(required=False)
    avg_energy_efficiency = fields.FloatField(required=False)
    median_internal_resistance = fields.FloatField(required=False)

    meta = {"collection": "samples", "indexes": ["name"]}

    def add_note(self, text: str, author: str | None = None) -> None:
        """Append a note entry to :attr:`notes_log` and persist the change."""

        self.notes_log.append(
            {
                "text": text,
                "author": author,
                "timestamp": datetime.datetime.utcnow(),
            }
        )
        self.save()

    @classmethod
    def get_by_name(cls, name: str):
        """Return the sample with the given name or ``None`` if not found."""
        return cls.objects(name=name).first()

    @classmethod
    def get_or_create(cls, name: str, **attrs) -> "Sample":
        """Retrieve a sample by name or create and save a new one."""
        sample = cls.get_by_name(name)
        if sample is None:
            sample = cls(name=name, **attrs)
            sample.save()
        return sample

    def clean(self):
        self.updated_at = datetime.datetime.utcnow()
        super().clean()

    def __str__(self):
        try:
            test_count = len(self.tests)
        except Exception:
            test_count = "?"
        return f"<Sample {self.name}, {test_count} tests>"
