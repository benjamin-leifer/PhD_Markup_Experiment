# battery_analysis/models/test_result.py

import datetime
import re
from mongoengine import Document, fields, ValidationError

from battery_analysis.utils.gridfs_conversion import data_to_gridfs, gridfs_to_data
from battery_analysis.utils.db import ensure_connection
from .stages import inherit_metadata

try:
    from .cycle_summary import CycleSummary
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    CycleSummary = importlib.import_module("cycle_summary").CycleSummary


class TestResult(Document):
    # Remove CASCADE to break circular dependency
    sample = fields.LazyReferenceField("Sample", required=True)
    parent = fields.ReferenceField("Cell", required=False)
    tester = fields.StringField(
        required=True,
        choices=["Arbin", "BioLogic", "Maccor", "Neware", "Other"],
        help_text="Origin of the data",
    )
    test_type = fields.StringField(
        choices=["Cycling", "EIS", "CV", "GITT", "HPPC", "Other"],
        default="Cycling",
        help_text="Type of electrochemical test",
    )
    name = fields.StringField(required=False, help_text="Test name or file identifier")
    cell_code = fields.StringField(
        required=False, help_text="Identifier derived from name"
    )
    file_path = fields.StringField(required=False, help_text="Original data file path")
    date = fields.DateTimeField(required=False, default=datetime.datetime.utcnow)

    temperature = fields.FloatField(required=False, help_text="Test temperature in °C")
    upper_cutoff_voltage = fields.FloatField(required=False)
    lower_cutoff_voltage = fields.FloatField(required=False)
    charge_rate = fields.FloatField(required=False)
    discharge_rate = fields.FloatField(required=False)

    cycles = fields.ListField(fields.EmbeddedDocumentField(CycleSummary))

    cycle_count = fields.IntField(required=False)
    initial_capacity = fields.FloatField(required=False)
    final_capacity = fields.FloatField(required=False)
    capacity_retention = fields.FloatField(required=False)
    avg_coulombic_eff = fields.FloatField(required=False)
    avg_energy_efficiency = fields.FloatField(required=False)
    median_internal_resistance = fields.FloatField(required=False)

    # New fields for automated protocol detection
    last_cycle_complete = fields.BooleanField(required=False)
    c_rates = fields.ListField(fields.FloatField(), required=False)
    protocol = fields.ReferenceField("TestProtocol", required=False)

    validated = fields.BooleanField(default=False)
    notes = fields.StringField(required=False)
    custom_data = fields.DictField(required=False)
    metadata = fields.DictField(default=dict)
    created_by = fields.StringField(required=False)
    last_modified_by = fields.StringField(required=False)
    notes_log = fields.ListField(
        fields.DictField(),
        default=list,
        help_text="List of timestamped note entries",
    )

    meta = {
        "collection": "test_results",
        "indexes": [
            "sample",
            "name",
            "cell_code",
            "date",
            "sample.anode",
            "sample.cathode",
            "sample.separator",
            "sample.electrolyte",
        ],
    }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, *args, **kwargs):  # type: ignore[override]
        """Persist the test result and refresh sample aggregates."""
        result = super().save(*args, **kwargs)
        try:
            sample = (
                self.sample.fetch() if hasattr(self.sample, "fetch") else self.sample
            )
            if sample is not None:
                existing_ids = [getattr(ref, "id", None) for ref in sample.tests]
                if self.id not in existing_ids:
                    sample.tests.append(self)
                sample.recompute_metrics()
        except Exception:  # pragma: no cover - best-effort update
            pass
        return result

    def clean(self):
        """Custom validation and automatic protocol assignment."""
        # Merge metadata from parent hierarchy
        self.metadata = inherit_metadata(self)

        if not self.cell_code and self.name:
            match = re.search(r"(CN\d+)", self.name)
            if match:
                self.cell_code = match.group(1)
        if self.cycle_count is not None:
            cycles_len = len(self.cycles or [])
            if cycles_len != self.cycle_count:
                raise ValidationError(
                    "cycle_count must match the number of items in cycles"
                )

        # ------------------------------------------------------------------
        # Protocol handling
        # ------------------------------------------------------------------
        try:  # Lazy import to avoid circular dependencies when running as script
            from .test_protocol import TestProtocol
        except Exception:  # pragma: no cover - defensive
            import importlib

            TestProtocol = importlib.import_module("test_protocol").TestProtocol

        # Validate referenced protocol exists
        if self.protocol is not None:
            proto_id = getattr(self.protocol, "id", None)
            exists = False
            if proto_id is not None:
                exists = TestProtocol.objects(id=proto_id).first() is not None
            if not exists:
                raise ValidationError(
                    "protocol reference must point to a saved TestProtocol"
                )

        # Auto-assign protocol if c_rates match a known pattern and no protocol provided
        if self.protocol is None and self.c_rates:
            try:
                from battery_analysis.analysis.protocol_detection import (
                    summarize_protocol,
                )

                summary = summarize_protocol(self.c_rates)
                proto = TestProtocol.objects(summary=summary).first()
                if proto:
                    self.protocol = proto
            except Exception:  # pragma: no cover - summarization is optional
                pass

    def full_clean(self) -> None:
        """Run MongoEngine validation, including :meth:`clean`."""
        self.validate(clean=True)

    def __str__(self):
        try:
            sample_name = self.sample.fetch().name if self.sample else "Unknown"
        except Exception:
            sample_name = "Unfetched"
        return f"<TestResult {self.name} ({self.tester}) for Sample {sample_name}>"

    # ------------------------------------------------------------------
    # Detailed cycle data helpers
    # ------------------------------------------------------------------
    def save_cycle_detail(self, cycle_index, charge_data, discharge_data):
        """Save detailed data for a cycle to GridFS.

        Parameters
        ----------
        cycle_index: int
            Index of the cycle being stored.
        charge_data: dict
            Dictionary containing charge segment data arrays.
        discharge_data: dict
            Dictionary containing discharge segment data arrays.
        """

        if not ensure_connection():  # pragma: no cover - requires DB
            return None

        existing = CycleDetailData.objects(
            test_result=self.id, cycle_index=cycle_index
        ).first()
        if existing:
            existing.delete()

        detail = CycleDetailData(test_result=self, cycle_index=cycle_index)
        if charge_data:
            data_to_gridfs(
                detail.charge_data,
                charge_data,
                f"charge_data_cycle_{cycle_index}.pkl",
            )
        if discharge_data:
            data_to_gridfs(
                detail.discharge_data,
                discharge_data,
                f"discharge_data_cycle_{cycle_index}.pkl",
            )
        detail.save()

        # Mark the cycle as having detailed data
        for cyc in self.cycles or []:
            if getattr(cyc, "cycle_index", None) == cycle_index:
                setattr(cyc, "has_detailed_data", True)
                break
        self.save()
        return detail

    def get_cycle_detail(self, cycle_index=None):
        """Retrieve detailed cycle data from GridFS.

        Parameters
        ----------
        cycle_index: int | None
            If provided, only that cycle is returned; otherwise a dictionary of
            all available cycles is returned.

        Returns
        -------
        dict
            Detailed data with ``charge`` and ``discharge`` keys.
        """

        if not ensure_connection():  # pragma: no cover - requires DB
            return {}

        if cycle_index is not None:
            detail = CycleDetailData.objects(
                test_result=self.id, cycle_index=cycle_index
            ).first()
            if not detail:
                return {}
            result = {}
            if detail.charge_data:
                result["charge"] = gridfs_to_data(detail.charge_data)
            if detail.discharge_data:
                result["discharge"] = gridfs_to_data(detail.discharge_data)
            return result

        # No cycle index specified: return all cycles
        data = {}
        for cyc in self.cycles or []:
            idx = getattr(cyc, "cycle_index", None)
            if idx is None:
                continue
            cycle_data = self.get_cycle_detail(idx)
            if cycle_data:
                data[idx] = cycle_data
        return data

    @classmethod
    def from_parent(cls, parent, sample, **kwargs):
        obj = cls(parent=parent, sample=sample, **kwargs)
        obj.metadata = inherit_metadata(obj)
        return obj


# In battery_analysis/models/test_result.py
class CycleDetailData(Document):
    """Document to store detailed cycle data using GridFS."""

    test_result = fields.ReferenceField("TestResult", required=True)
    cycle_index = fields.IntField(required=True)

    # Store data files
    charge_data = fields.FileField()
    discharge_data = fields.FileField()

    meta = {
        "collection": "cycle_detail_data",
        "indexes": [{"fields": ["test_result", "cycle_index"], "unique": True}],
    }
