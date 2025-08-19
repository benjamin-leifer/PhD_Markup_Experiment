import os
import sys
import pytest

TESTS_DIR = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

doe_builder = pytest.importorskip("battery_analysis.utils.doe_builder")  # noqa: E402

generate_combinations = doe_builder.generate_combinations
save_plan = doe_builder.save_plan
main = doe_builder.main


def test_generate_combinations_cartesian_product() -> None:
    factors = {"A": [1, 2], "B": ["x"]}
    combos = generate_combinations(factors)
    assert combos == [
        {"A": 1, "B": "x"},
        {"A": 2, "B": "x"},
    ]


def test_save_plan_builds_matrix() -> None:
    factors = {"Temp": [25, 30], "Rate": [0.5]}
    plan = save_plan("demo", factors)
    assert plan.name == "demo"
    assert len(plan.matrix) == 2
    assert plan.factors == factors


def test_status_reports_remaining(
    capsys: pytest.CaptureFixture[str],
) -> None:
    factors = {"A": [1], "B": [2]}
    plan = save_plan("status_demo", factors)
    # Should list the only combination as remaining
    main(["--status", "status_demo"])
    captured = capsys.readouterr().out
    assert "Remaining combinations: 1" in captured
    assert "{'A': 1, 'B': 2}" in captured

    # Once marked with a test entry, remaining should drop to zero
    plan.matrix[0].setdefault("tests", []).append({"id": "1", "timestamp": "now"})
    main(["--status", "status_demo"])
    captured = capsys.readouterr().out
    assert "Remaining combinations: 0" in captured
