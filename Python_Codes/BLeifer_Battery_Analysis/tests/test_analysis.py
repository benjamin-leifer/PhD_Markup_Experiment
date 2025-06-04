# tests/test_analysis.py

import os
import sys

# Add the package root to sys.path so ``battery_analysis`` can be imported
TESTS_DIR = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.abspath(os.path.join(TESTS_DIR, '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from battery_analysis import analysis
from mongoengine import connect, disconnect
import mongomock

def test_compute_metrics():
    # Create a dummy cycles summary list for testing
    cycles = [
        {"cycle_index": 1, "charge_capacity": 1.0, "discharge_capacity": 0.95, "coulombic_efficiency": 0.95},
        {"cycle_index": 2, "charge_capacity": 1.0, "discharge_capacity": 0.90, "coulombic_efficiency": 0.90}
    ]
    metrics = analysis.compute_metrics(cycles)
    # There are 2 cycles
    assert metrics["cycle_count"] == 2
    # Initial capacity should be 0.95 (from cycle 1), final 0.90 (cycle 2)
    assert abs(metrics["initial_capacity"] - 0.95) < 1e-6
    assert abs(metrics["final_capacity"] - 0.90) < 1e-6
    # Retention = 0.90/0.95
    expected_retention = 0.90 / 0.95
    assert abs(metrics["capacity_retention"] - expected_retention) < 1e-6
    # Average CE = (0.95 + 0.90)/2
    assert abs(metrics["avg_coulombic_eff"] - 0.925) < 1e-6

def test_inferred_property_propagation():
    # Here we simulate using the model layer (without actual DB for simplicity)
    # Create a dummy sample and test results in memory
    from battery_analysis.models import TestResult

    # Use an in-memory MongoDB so TestResult.objects does not fail
    connect("testdb", mongo_client_class=mongomock.MongoClient)

    # Temporarily disable the objects manager to avoid DB lookups
    original_manager = getattr(TestResult, "objects", None)
    if hasattr(TestResult, "objects"):
        delattr(TestResult, "objects")

    class DummySample:
        def __init__(self):
            self.tests = []
            self.parent = None
            self.avg_initial_capacity = None
            self.avg_final_capacity = None
            self.avg_capacity_retention = None
            self.avg_coulombic_eff = None

    sample = DummySample()
    # Normally, sample.save() and connecting to DB is needed, but for logic test we skip DB ops.
    # Create dummy TestResult objects (not saved to DB) and assign to sample
    class DummyTest:  # Simulate minimal interface of TestResult for this test
        def __init__(self, init_cap, final_cap):
            self.initial_capacity = init_cap
            self.final_capacity = final_cap
            self.capacity_retention = final_cap/init_cap if init_cap else 0
            self.avg_coulombic_eff = 0.0
    # Add two dummy tests with known values
    t1 = DummyTest(1.0, 0.8)   # 80% retention
    t2 = DummyTest(1.2, 1.0)   # ~83.3% retention
    # Monkey-patch sample.tests to behave like ReferenceField list for analysis.update_sample_properties
    sample.tests.append(t1)
    sample.tests.append(t2)
    # Compute inferred properties
    analysis.update_sample_properties(sample, save=False)

    # Restore objects manager and clean up connection
    if original_manager is not None:
        TestResult.objects = original_manager
    
    disconnect()
    # Now check that sample averages are correct:
    # avg_initial_capacity = (1.0 + 1.2) / 2 = 1.1
    assert abs(sample.avg_initial_capacity - 1.1) < 1e-6
    # avg_final_capacity = (0.8 + 1.0) / 2 = 0.9
    assert abs(sample.avg_final_capacity - 0.9) < 1e-6
    # avg_capacity_retention = (0.8/1.0 + 1.0/1.2) / 2 = (0.8 + 0.8333)/2 ~ 0.8167
    expected_avg_ret = ((0.8) + (1.0/1.2)) / 2
    assert abs(sample.avg_capacity_retention - expected_avg_ret) < 1e-6


def test_get_cycle_data_include_raw():
    """Ensure raw cycle arrays are returned when include_raw=True."""
    from battery_analysis import models

    # Dummy cycle with detailed arrays
    class DummyCycle:
        def __init__(self):
            self.cycle_index = 1
            self.charge_capacity = 1.0
            self.discharge_capacity = 0.9
            self.coulombic_efficiency = 0.9
            self.voltage_charge = [3.6, 3.7]
            self.current_charge = [0.1, 0.1]
            self.capacity_charge = [0.0, 0.5]
            self.time_charge = [0.0, 1.0]
            self.voltage_discharge = [3.7, 3.6]
            self.current_discharge = [-0.1, -0.1]
            self.capacity_discharge = [0.5, 0.0]
            self.time_discharge = [1.0, 2.0]

    class DummySample:
        def __init__(self):
            self.name = "S1"

    class DummyTest:
        def __init__(self):
            self.id = "T1"
            self.sample = DummySample()
            self.name = "Test"
            self.tester = "Arbin"
            self.cycle_count = 1
            self.initial_capacity = 1.0
            self.final_capacity = 0.9
            self.capacity_retention = 0.9
            self.avg_coulombic_eff = 0.9
            self.cycles = [DummyCycle()]

    dummy_test = DummyTest()

    # Patch the objects manager so get_cycle_data can find our dummy test
    class DummyQuery:
        def __init__(self, obj):
            self.obj = obj

        def first(self):
            return self.obj

    class DummyManager:
        def __init__(self, obj):
            self.obj = obj

        def __call__(self, id):
            return DummyQuery(self.obj)

    original_manager = getattr(models.TestResult, "objects", None)
    models.TestResult.objects = DummyManager(dummy_test)

    try:
        data = analysis.get_cycle_data("T1", include_raw=True)
    finally:
        # Restore manager
        if original_manager is not None:
            models.TestResult.objects = original_manager
        else:
            delattr(models.TestResult, "objects")

    assert data["cycles"][0]["cycle_index"] == 1
    assert "raw" in data["cycles"][0]
    assert data["cycles"][0]["raw"]["charge"]["voltage"] == [3.6, 3.7]
    assert data["cycles"][0]["raw"]["discharge"]["current"] == [-0.1, -0.1]
