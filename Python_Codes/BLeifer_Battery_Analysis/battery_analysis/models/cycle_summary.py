"""Model definitions for per-cycle summary data."""

from mongoengine import DynamicEmbeddedDocument, fields


class CycleSummary(DynamicEmbeddedDocument):
    """Embedded document storing summary metrics for a cycle.

    Detailed time-series data (voltage, current, capacity, etc.) is stored
    separately in :class:`CycleDetailData` via GridFS. This document therefore
    keeps only lightweight aggregate metrics.
    """

    cycle_index = fields.IntField(required=True)
    charge_capacity = fields.FloatField(
        required=True, help_text="Charge capacity in mAh for this cycle"
    )
    discharge_capacity = fields.FloatField(
        required=True, help_text="Discharge capacity in mAh for this cycle"
    )
    coulombic_efficiency = fields.FloatField(
        required=True,
        help_text="Coulombic efficiency (discharge/charge) for this cycle",
    )

    # Additional per-cycle metrics
    charge_energy = fields.FloatField(
        required=False, help_text="Charge energy in Wh for this cycle"
    )
    discharge_energy = fields.FloatField(
        required=False, help_text="Discharge energy in Wh for this cycle"
    )
    energy_efficiency = fields.FloatField(
        required=False,
        help_text="Energy efficiency (discharge_energy/charge_energy) for this cycle",
    )
    internal_resistance = fields.FloatField(
        required=False, help_text="Internal resistance in Ohms"
    )

    # Flag noting whether detailed time-series data exists externally
    has_detailed_data = fields.BooleanField(required=False, default=False)

    def __str__(self) -> str:  # pragma: no cover - simple representation
        return (
            f"Cycle {self.cycle_index}: {self.discharge_capacity:.3f} mAh, "
            f"CE: {self.coulombic_efficiency:.3f}"
        )
