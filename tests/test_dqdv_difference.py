from pathlib import Path
import importlib.util
import sys

import numpy as np
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
MODULE_DIR = PACKAGE_ROOT / "battery_analysis"
sys.path.insert(0, str(MODULE_DIR))
ADV_PATH = MODULE_DIR / "advanced_analysis.py"
spec = importlib.util.spec_from_file_location("advanced_analysis", ADV_PATH)
aa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aa)
compute_dqdv_difference = aa.compute_dqdv_difference


def test_compute_dqdv_difference():
    cap1 = np.array([0, 1, 2, 3, 4, 5])
    volt1 = np.array([3.0, 3.2, 3.4, 3.6, 3.8, 4.0])
    cap2 = np.array([0, 1, 2, 3, 4, 5])
    volt2 = np.array([3.0, 3.3, 3.6, 3.9, 4.2, 4.5])

    result, fig = compute_dqdv_difference(cap1, volt1, cap2, volt2, smooth=False)
    diff = np.array(result["difference"]["dqdv"])

    assert np.allclose(diff, -1.6666666666666667)
    assert isinstance(fig, Figure)
