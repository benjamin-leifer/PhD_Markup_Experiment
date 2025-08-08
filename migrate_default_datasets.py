"""Assign default CellDataset to Sample documents.

This migration iterates over all samples and ensures their
``default_dataset`` field references the dataset built from their tests.
"""

import os
import sys

PACKAGE_ROOT = os.path.join(
    os.path.dirname(__file__), "Python_Codes", "BLeifer_Battery_Analysis"
)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from battery_analysis.models import Sample, TestResult  # noqa: E402
from battery_analysis.utils.db import ensure_connection  # noqa: E402
from battery_analysis.utils.cell_dataset_builder import update_cell_dataset  # noqa: E402


def main() -> None:
    if not ensure_connection():
        print("Could not connect to database.")
        return

    updated = 0
    for sample in Sample.objects():
        test = TestResult.objects(sample=sample.id).first()
        if not test or not test.cell_code:
            continue
        update_cell_dataset(test.cell_code)
        updated += 1
    print(f"Assigned default datasets for {updated} samples")


if __name__ == "__main__":
    main()
