# battery_analysis/models/test_result.py

import datetime
from mongoengine import Document, fields
from .cycle_summary import CycleSummary


class TestResult(Document):
    # Remove CASCADE to break circular dependency
    sample = fields.LazyReferenceField("Sample", required=True)
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

    # New fields for automated protocol detection
    last_cycle_complete = fields.BooleanField(required=False)
    c_rates = fields.ListField(fields.FloatField(), required=False)
    protocol = fields.ReferenceField("TestProtocol", required=False)

    validated = fields.BooleanField(default=False)
    notes = fields.StringField(required=False)
    custom_data = fields.DictField(required=False)
    created_by = fields.StringField(required=False)
    last_modified_by = fields.StringField(required=False)
    notes_log = fields.ListField(
        fields.DictField(),
        default=list,
        help_text="List of timestamped note entries",
    )

    meta = {"collection": "test_results"}

    def __str__(self):
        try:
            sample_name = self.sample.fetch().name if self.sample else "Unknown"
        except Exception:
            sample_name = "Unfetched"
        return f"<TestResult {self.name} ({self.tester}) for Sample {sample_name}>"


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
