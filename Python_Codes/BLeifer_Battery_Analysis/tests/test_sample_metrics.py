import os
import sys
import types

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

# Avoid executing battery_analysis.__init__ with heavy deps
BA_PATH = os.path.abspath(os.path.join(TEST_ROOT, "battery_analysis"))
pkg = types.ModuleType("battery_analysis")
pkg.__path__ = [BA_PATH]
sys.modules.setdefault("battery_analysis", pkg)

from battery_analysis.models.sample import Sample as SampleDoc  # noqa: E402
import pytest  # noqa: E402


class DummyTest:
    def __init__(self, init, final, retention, ce, energy, resistance):
        self.initial_capacity = init
        self.final_capacity = final
        self.capacity_retention = retention
        self.avg_coulombic_eff = ce
        self.avg_energy_efficiency = energy
        self.median_internal_resistance = resistance


class DummySample:
    def __init__(self, name):
        self.name = name
        self.tests = []
        self.avg_initial_capacity = None
        self.avg_final_capacity = None
        self.avg_capacity_retention = None
        self.avg_coulombic_eff = None
        self.avg_energy_efficiency = None
        self.median_internal_resistance = None

    def save(self, *args, **kwargs):
        pass


DummySample.recompute_metrics = SampleDoc.recompute_metrics


def test_recompute_metrics_aggregation():
    sample = DummySample("S1")
    t1 = DummyTest(1.0, 0.8, 0.8, 0.9, 0.85, 0.1)
    t2 = DummyTest(1.2, 1.0, 1.0 / 1.2, 0.92, 0.88, 0.2)
    sample.tests = [t1, t2]
    sample.recompute_metrics()
    assert sample.avg_initial_capacity == pytest.approx(1.1)
    assert sample.avg_final_capacity == pytest.approx(0.9)
    expected_retention = (0.8 + (1.0 / 1.2)) / 2
    assert sample.avg_capacity_retention == pytest.approx(expected_retention)
    assert sample.avg_coulombic_eff == pytest.approx((0.9 + 0.92) / 2)
    assert sample.avg_energy_efficiency == pytest.approx((0.85 + 0.88) / 2)
    assert sample.median_internal_resistance == pytest.approx(0.15)


def test_recompute_metrics_no_tests():
    sample = DummySample("Empty")
    sample.recompute_metrics()
    assert sample.avg_initial_capacity is None
    assert sample.avg_final_capacity is None
    assert sample.avg_capacity_retention is None
    assert sample.avg_coulombic_eff is None
    assert sample.avg_energy_efficiency is None
    assert sample.median_internal_resistance is None
