# battery_analysis/models/sample.py

import datetime
from typing import TYPE_CHECKING, List

import numpy as np
from mongoengine import Document, ValidationError, fields
from mongoengine.errors import NotUniqueError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .testresult import TestResult


class Sample(Document):
    """Represents a physical sample or cell.

    Use :meth:`add_note` to append timestamped notes for tracking::

        sample.add_note("checked performance", author="researcher")

    Use :meth:`get_or_create` to fetch an existing sample or create it if
    missing::

        sample = Sample.get_or_create("Cell-42")

    Link an image to the sample using :func:`ingest_image_file`::

        ingest_image_file("microscope.png", sample=sample)
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
    images = fields.ListField(fields.LazyReferenceField("RawDataFile"))
    default_dataset = fields.ReferenceField("CellDataset")

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

    meta = {"collection": "samples", "indexes": ["name", "tags"]}

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
        """Atomically fetch or create a sample by name.

        MongoDB performs the lookup and insert as a single upsert so concurrent
        import workers do not race on the unique ``name`` constraint. Any
        ``attrs`` are applied only when the sample is first inserted.
        """

        now = datetime.datetime.utcnow()
        update = {
            "set_on_insert__name": name,
            "set_on_insert__created_at": now,
            "set_on_insert__updated_at": now,
        }
        for key, value in attrs.items():
            update[f"set_on_insert__{key}"] = value

        try:
            sample = cls.objects(name=name).modify(upsert=True, new=True, **update)
        except NotUniqueError:
            sample = cls.objects(name=name).get()
        if sample is None:  # pragma: no cover - defensive fallback
            sample = cls.objects(name=name).get()
        return sample

    def clean(self):
        self.updated_at = datetime.datetime.utcnow()
        self.validate_components()
        super().clean()

    def validate_components(self) -> None:
        """Ensure component references are saved and not self."""
        components = {
            "parent": self.parent,
            "anode": self.anode,
            "cathode": self.cathode,
            "separator": self.separator,
            "electrolyte": self.electrolyte,
        }

        for name, ref in components.items():
            if ref is None:
                continue
            ref_id = getattr(ref, "id", None)
            # Self-reference check (requires referenced doc to be saved)
            if self.id is not None and ref_id == self.id:
                raise ValidationError(f"{name} cannot reference self")
            if ref_id is None:
                raise ValidationError(
                    f"{name} reference must point to a saved Sample"
                )

    def __str__(self):
        try:
            test_count = len(self.tests)
        except Exception:
            test_count = "?"
        return f"<Sample {self.name}, {test_count} tests>"

    # ------------------------------------------------------------------
    # Metric aggregation
    # ------------------------------------------------------------------
    def recompute_metrics(self) -> None:
        """Recalculate aggregated metrics from current ``TestResult`` rows.

        The aggregation queries ``TestResult`` documents by ``sample`` instead of
        trusting :attr:`tests` to already be in sync. The stored ``tests`` list is
        then refreshed in one update so import workers can safely add results
        without racing on in-memory list mutation.
        """

        fetched_tests: List["TestResult"]
        if self.id is not None:
            from .testresult import TestResult

            fetched_tests = list(TestResult.objects(sample=self.id))
        else:
            fetched_tests = []
            for ref in self.tests:
                try:
                    test = ref.fetch() if hasattr(ref, "fetch") else ref
                except Exception:  # pragma: no cover - defensive against bad refs
                    continue
                fetched_tests.append(test)

        init_caps = [
            t.initial_capacity
            for t in fetched_tests
            if getattr(t, "initial_capacity", None) is not None
        ]
        final_caps = [
            t.final_capacity
            for t in fetched_tests
            if getattr(t, "final_capacity", None) is not None
        ]
        retentions = [
            t.capacity_retention
            for t in fetched_tests
            if getattr(t, "capacity_retention", None) is not None
        ]
        coul_eff = [
            t.avg_coulombic_eff
            for t in fetched_tests
            if getattr(t, "avg_coulombic_eff", None) is not None
        ]
        energy_eff = [
            t.avg_energy_efficiency
            for t in fetched_tests
            if getattr(t, "avg_energy_efficiency", None) is not None
        ]
        resistances = [
            t.median_internal_resistance
            for t in fetched_tests
            if hasattr(t, "median_internal_resistance")
            and t.median_internal_resistance is not None
        ]

        metrics = {
            "avg_initial_capacity": float(np.mean(init_caps)) if init_caps else None,
            "avg_final_capacity": float(np.mean(final_caps)) if final_caps else None,
            "avg_capacity_retention": float(np.mean(retentions)) if retentions else None,
            "avg_coulombic_eff": float(np.mean(coul_eff)) if coul_eff else None,
            "avg_energy_efficiency": float(np.mean(energy_eff)) if energy_eff else None,
            "median_internal_resistance": (
                float(np.median(resistances)) if resistances else None
            ),
        }

        test_ids = [test.id for test in fetched_tests if getattr(test, "id", None) is not None]
        if self.id is not None:
            self.update(set__tests=test_ids, **{f"set__{key}": value for key, value in metrics.items()})
            self.reload()
        else:  # pragma: no cover - unsaved instance fallback
            self.tests = fetched_tests
            for key, value in metrics.items():
                setattr(self, key, value)

