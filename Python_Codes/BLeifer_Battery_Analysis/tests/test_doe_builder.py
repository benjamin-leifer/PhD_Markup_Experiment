import os
import sys

TESTS_DIR = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from battery_analysis.utils.doe_builder import generate_combinations, save_plan


def test_generate_combinations_cartesian_product():
    factors = {"A": [1, 2], "B": ["x"]}
    combos = generate_combinations(factors)
    assert combos == [
        {"A": 1, "B": "x"},
        {"A": 2, "B": "x"},
    ]


def test_save_plan_builds_matrix():
    factors = {"Temp": [25, 30], "Rate": [0.5]}
    plan = save_plan("demo", factors)
    assert plan.name == "demo"
    assert len(plan.matrix) == 2
    assert plan.factors == factors
