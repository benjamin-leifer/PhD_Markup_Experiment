from battery_analysis.models import Sample, TestResult
import time


def test_timestamps_are_set_and_updated():
    sample = Sample(name="S1")
    test = TestResult(sample=sample, tester="Arbin")
    assert test.created_at is not None
    assert test.updated_at is not None

    first_created = test.created_at
    first_updated = test.updated_at

    time.sleep(0.01)
    test.clean()
    assert test.created_at == first_created
    assert test.updated_at > first_updated
