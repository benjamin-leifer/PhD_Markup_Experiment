"""
MongoDB document models for battery test data analysis.

This module defines the MongoEngine document classes for Sample and TestResult,
as well as embedded documents like CycleSummary.
"""

import datetime
from mongoengine import Document, EmbeddedDocument, fields, CASCADE


class CycleSummary(EmbeddedDocument):
    """Embedded document to store summary of a single cycle in a battery test."""

    cycle_index = fields.IntField(required=True)
    charge_capacity = fields.FloatField(
        required=True,
        help_text="Charge capacity in mAh for this cycle"
    )
    discharge_capacity = fields.FloatField(
        required=True,
        help_text="Discharge capacity in mAh for this cycle"
    )
    coulombic_efficiency = fields.FloatField(
        required=True,
        help_text="Coulombic efficiency (discharge/charge) for this cycle"
    )
    # Additional per-cycle metrics can be added here (e.g., energy, resistance, etc.)
    charge_energy = fields.FloatField(
        required=False,
        help_text="Charge energy in Wh for this cycle"
    )
    discharge_energy = fields.FloatField(
        required=False,
        help_text="Discharge energy in Wh for this cycle"
    )
    energy_efficiency = fields.FloatField(
        required=False,
        help_text="Energy efficiency (discharge_energy/charge_energy) for this cycle"
    )
    internal_resistance = fields.FloatField(
        required=False,
        help_text="Internal resistance in Ohms"
    )

    def __str__(self):
        return f"Cycle {self.cycle_index}: {self.discharge_capacity:.3f} mAh, CE: {self.coulombic_efficiency:.3f}"


class Sample(Document):
    """A battery sample or cell, which may have one or more test results."""

    name = fields.StringField(required=True, unique=True)
    chemistry = fields.StringField(required=False)  # e.g., "Li-ion NMC", etc.
    manufacturer = fields.StringField(required=False)
    form_factor = fields.StringField(required=False)  # e.g., "18650", "pouch", etc.
    nominal_capacity = fields.FloatField(required=False, help_text="Nominal capacity in mAh")
    parent = fields.ReferenceField('Sample', required=False)  # link to a parent sample (e.g., batch or material)
    tests = fields.ListField(fields.ReferenceField('TestResult', reverse_delete_rule=CASCADE))

    # Metadata fields
    created_at = fields.DateTimeField(default=datetime.datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.datetime.utcnow)
    tags = fields.ListField(fields.StringField(), required=False)
    description = fields.StringField(required=False)

    # Inferred / aggregated properties from tests:
    avg_initial_capacity = fields.FloatField(required=False)
    avg_final_capacity = fields.FloatField(required=False)
    avg_capacity_retention = fields.FloatField(required=False)
    avg_coulombic_eff = fields.FloatField(required=False)
    avg_energy_efficiency = fields.FloatField(required=False)
    median_internal_resistance = fields.FloatField(required=False)

    meta = {'collection': 'samples'}  # Specify collection name in MongoDB

    def __str__(self):
        return f"<Sample {self.name}>"

    def clean(self):
        """Update the updated_at timestamp before saving."""
        self.updated_at = datetime.datetime.utcnow()
        super().clean()


class TestResult(Document):
    """An electrochemical test result (e.g., a battery cycling test) associated with a Sample."""

    sample = fields.ReferenceField(Sample, required=True, reverse_delete_rule=CASCADE)
    tester = fields.StringField(
        required=True,
        choices=['Arbin', 'BioLogic', 'Maccor', 'Neware', 'Other'],
        help_text="Origin of the data"
    )
    test_type = fields.StringField(
        choices=['Cycling', 'EIS', 'CV', 'GITT', 'HPPC', 'Other'],
        default='Cycling',
        help_text="Type of electrochemical test"
    )
    name = fields.StringField(required=False, help_text="Test name or file identifier")
    file_path = fields.StringField(required=False, help_text="Original data file path")
    date = fields.DateTimeField(required=False, default=datetime.datetime.utcnow)

    # Test parameters
    temperature = fields.FloatField(required=False, help_text="Test temperature in °C")
    upper_cutoff_voltage = fields.FloatField(required=False, help_text="Upper cutoff voltage in V")
    lower_cutoff_voltage = fields.FloatField(required=False, help_text="Lower cutoff voltage in V")
    charge_rate = fields.FloatField(required=False, help_text="Charge rate (C-rate)")
    discharge_rate = fields.FloatField(required=False, help_text="Discharge rate (C-rate)")

    cycles = fields.ListField(fields.EmbeddedDocumentField(CycleSummary))

    # Summary metrics for the entire test:
    cycle_count = fields.IntField(required=False, help_text="Total number of cycles in the test")
    initial_capacity = fields.FloatField(required=False, help_text="Discharge capacity of first cycle (mAh)")
    final_capacity = fields.FloatField(required=False, help_text="Discharge capacity of last cycle (mAh)")
    capacity_retention = fields.FloatField(required=False, help_text="Fraction of capacity retained (final/initial)")
    avg_coulombic_eff = fields.FloatField(required=False, help_text="Average Coulombic Efficiency over all cycles")
    avg_energy_efficiency = fields.FloatField(required=False, help_text="Average Energy Efficiency over all cycles")

    # Flags for data validation and quality
    validated = fields.BooleanField(default=False, help_text="Whether the test data has been validated")
    notes = fields.StringField(required=False, help_text="Notes or comments about the test")

    # Custom data field for storing additional test-specific data
    custom_data = fields.DictField(required=False, help_text="Additional test-specific data")

    meta = {'collection': 'test_results'}

    def __str__(self):
        return f"<TestResult {self.name} ({self.tester}) for Sample {self.sample.name}>"