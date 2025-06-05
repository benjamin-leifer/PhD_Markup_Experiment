from __future__ import annotations

from typing import Optional, List

try:  # pragma: no cover - optional import depending on environment
    from battery_analysis.models import Sample, CycleSummary, TestResult
except Exception:  # Fallback lightweight dataclasses if battery_analysis not installed
    from dataclasses import dataclass, field

    @dataclass
    class CycleSummary:
        cycle_index: int
        charge_capacity: float
        discharge_capacity: float
        coulombic_efficiency: float

    @dataclass
    class TestResult:
        cycles: List[CycleSummary] = field(default_factory=list)

    @dataclass
    class Sample:
        name: str
        tests: List[TestResult] = field(default_factory=list)
        avg_final_capacity: Optional[float] = None
        median_internal_resistance: Optional[float] = None


def _get_attr(obj: object, attr: str) -> Optional[float]:
    """Helper to fetch attribute possibly stored in ``custom_data``."""
    val = getattr(obj, attr, None)
    if val is None and hasattr(obj, "custom_data"):
        val = obj.custom_data.get(attr)
    return val


def normalize_capacity(sample: Sample, by: str = "area") -> Optional[float]:
    """Return capacity normalized by area if possible."""
    cap = getattr(sample, "avg_final_capacity", None)
    if cap is None:
        cap = getattr(sample, "avg_initial_capacity", None)
    if cap is None:
        return None

    if by == "area":
        area = _get_attr(sample, "area")
        if not area:
            return None
        return cap / area
    return None


def normalize_impedance(sample: Sample, by: str = "thickness") -> Optional[float]:
    """Return impedance normalized by thickness if data is available."""
    resistance = getattr(sample, "median_internal_resistance", None)
    if resistance is None:
        return None

    if by == "thickness":
        thickness = _get_attr(sample, "thickness")
        if not thickness:
            return None
        return resistance * thickness
    return None


def coulombic_efficiency_percent(sample: Sample, first_cycles: int = 5) -> Optional[float]:
    """Return average coulombic efficiency over first ``first_cycles`` as percent."""
    if not hasattr(sample, "tests"):
        return None

    ce_values: List[float] = []
    for t in sample.tests:
        cycles = getattr(t, "cycles", [])
        for c in cycles[:first_cycles]:
            ce = getattr(c, "coulombic_efficiency", None)
            if ce is not None:
                ce_values.append(float(ce))
    if not ce_values:
        return None
    return sum(ce_values) / len(ce_values) * 100
