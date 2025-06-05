import sys
import os

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from normalization_utils import (
    normalize_capacity,
    normalize_impedance,
    coulombic_efficiency_percent,
)
from dataclasses import dataclass, field


@dataclass
class CycleSummary:
    cycle_index: int
    charge_capacity: float
    discharge_capacity: float
    coulombic_efficiency: float


@dataclass
class TestResult:
    cycles: list[CycleSummary] = field(default_factory=list)


@dataclass
class Sample:
    name: str
    tests: list[TestResult] = field(default_factory=list)
    avg_final_capacity: float | None = None
    median_internal_resistance: float | None = None


def test_capacity_normalization():
    sample = Sample(name="S1")
    sample.avg_final_capacity = 2.0
    sample.area = 2.0
    assert normalize_capacity(sample) == 1.0


def test_impedance_normalization():
    sample = Sample(name="S1")
    sample.median_internal_resistance = 0.5
    sample.thickness = 0.1
    assert normalize_impedance(sample) == 0.05


def test_coulombic_efficiency_percent():
    sample = Sample(name="S1")
    cycle = CycleSummary(1, 1.0, 0.9, 0.9)
    test = TestResult(cycles=[cycle])
    sample.tests.append(test)
    assert abs(coulombic_efficiency_percent(sample, first_cycles=1) - 90.0) < 1e-6
