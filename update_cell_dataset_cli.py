"""Update CellDataset documents when new TestResult data are added."""

import os
import sys

PACKAGE_ROOT = os.path.join(
    os.path.dirname(__file__), "Python_Codes", "BLeifer_Battery_Analysis"
)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from battery_analysis.models import TestResult  # noqa: E402
from battery_analysis.utils.db import ensure_connection  # noqa: E402
from battery_analysis.utils.cell_dataset_builder import (  # noqa: E402
    update_cell_dataset,
)


def main(cell_code: str | None = None) -> None:
    """Refresh cell datasets for all or one cell code."""
    if not ensure_connection():
        print("Could not connect to database.")
        return

    if cell_code:
        update_cell_dataset(cell_code)
        print(f"Updated dataset for {cell_code}")
        return

    codes = TestResult.objects.distinct("cell_code")
    for code in codes:
        if code:
            update_cell_dataset(code)
    print("Updated datasets for all cell codes")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
