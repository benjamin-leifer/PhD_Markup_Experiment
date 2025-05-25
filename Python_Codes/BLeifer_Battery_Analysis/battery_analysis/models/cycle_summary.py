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
    # Additional per-cycle metrics (existing fields)
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

    # NEW: Add fields to store detailed cycle data as arrays
    voltage_charge = fields.ListField(fields.FloatField(), required=False)
    current_charge = fields.ListField(fields.FloatField(), required=False)
    capacity_charge = fields.ListField(fields.FloatField(), required=False)
    time_charge = fields.ListField(fields.FloatField(), required=False)

    voltage_discharge = fields.ListField(fields.FloatField(), required=False)
    current_discharge = fields.ListField(fields.FloatField(), required=False)
    capacity_discharge = fields.ListField(fields.FloatField(), required=False)
    time_discharge = fields.ListField(fields.FloatField(), required=False)

    # Methods to add detailed data
    def add_charge_point(self, voltage, current, capacity, time):
        """Add a data point to the charge segment."""
        self.voltage_charge.append(float(voltage))
        self.current_charge.append(float(current))
        self.capacity_charge.append(float(capacity))
        self.time_charge.append(float(time))

    def add_discharge_point(self, voltage, current, capacity, time):
        """Add a data point to the discharge segment."""
        self.voltage_discharge.append(float(voltage))
        self.current_discharge.append(float(current))
        self.capacity_discharge.append(float(capacity))
        self.time_discharge.append(float(time))

    def __str__(self):
        return f"Cycle {self.cycle_index}: {self.discharge_capacity:.3f} mAh, CE: {self.coulombic_efficiency:.3f}"

