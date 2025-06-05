import sys
import os

TESTS_DIR = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from battery_analysis.outlier_analysis import detect_outliers


class DummySample:
    def __init__(self, sid, val):
        self.id = sid
        self.name = sid
        self.capacity_retention = val


def test_detect_outliers_identifies_value():
    samples = [
        DummySample("S1", 0.8),
        DummySample("S2", 0.82),
        DummySample("S3", 0.81),
        DummySample("S4", 0.79),
        DummySample("S5", 1.5),
    ]
    outliers, fig = detect_outliers(samples)
    fig.clf()
    assert outliers == ["S5"]
