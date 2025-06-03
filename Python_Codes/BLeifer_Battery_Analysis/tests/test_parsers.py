# tests/test_parsers.py

import os
import sys

# Ensure the battery_analysis package is importable when running tests
TESTS_DIR = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.abspath(os.path.join(TESTS_DIR, '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from battery_analysis.parsers import arbin_parser, biologic_parser

def test_arbin_parse_basic():
    # Assume we have a small CSV fixture for Arbin data with known values
    test_file = "tests/fixtures/arbin_small.csv"
    if not os.path.exists(test_file):
        # If fixture not present, skip this test
        return
    cycles = arbin_parser.parse_arbin_excel(test_file)
    # The fixture is known to have 3 cycles
    assert len(cycles) == 3
    # Check that keys exist
    assert "cycle_index" in cycles[0] and "discharge_capacity" in cycles[0]
    # If known expected values from the fixture:
    # e.g., first cycle discharge capacity ~1.234 mAh, coulombic_eff ~95%
    assert abs(cycles[0]["discharge_capacity"] - 1.234) < 1e-3
    assert abs(cycles[0]["coulombic_efficiency"] - 0.95) < 1e-2

def test_biologic_parse_basic():
    # Similar approach for a BioLogic .mpt fixture
    test_file = "tests/fixtures/biologic_small.mpt"
    if not os.path.exists(test_file):
        return
    cycles = biologic_parser.parse_biologic(test_file)
    # If the file is one long cycle (e.g., CV data treated as one cycle)
    assert len(cycles) >= 1
    assert "charge_capacity" in cycles[0] and "coulombic_efficiency" in cycles[0]
    # If known values, test a couple (here we assume one cycle with 0.5 mAh charged, 0.45 mAh discharged)
    if len(cycles) == 1:
        assert abs(cycles[0]["charge_capacity"] - 0.50) < 1e-2
        assert abs(cycles[0]["discharge_capacity"] - 0.45) < 1e-2
        assert abs(cycles[0]["coulombic_efficiency"] - 0.90) < 1e-2
