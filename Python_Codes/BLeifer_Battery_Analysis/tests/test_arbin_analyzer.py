import os
import sys
import pandas as pd

TESTS_DIR = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.abspath(os.path.join(TESTS_DIR, '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from battery_analysis.arbin_analyzer import split_cycles_by_rate


def make_df(n_cycles: int) -> pd.DataFrame:
    rows = []
    for c in range(1, n_cycles + 1):
        for step in ['charge', 'discharge']:
            rows.append({
                'cycle': c,
                'step_type': step,
                'current_A': 0.1,
                'capacity_mAh': c * 0.1,
                'voltage_V': 3.7,
                'timestamp': pd.Timestamp('2023-01-01')
            })
    return pd.DataFrame(rows)


def test_split_cycles_basic():
    df = make_df(60)
    segs = split_cycles_by_rate(df)
    assert set(segs['formation']['cycle'].unique()) == {1}
    assert set(segs['rate_C10']['cycle'].unique()) == {2, 3, 4}
    assert set(segs['rate_2C']['cycle'].unique()) == {17, 18, 19}
    assert segs['long_term']['cycle'].min() == 20
    assert set(segs['cap_check_50']['cycle'].unique()) == {50}


def test_split_cycles_short_file():
    df = make_df(5)
    segs = split_cycles_by_rate(df)
    assert segs['long_term'].empty
    assert set(segs['formation']['cycle'].unique()) == {1}

