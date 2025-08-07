"""Populate ``cell_code`` for existing TestResult documents."""

import os
import re
import sys

PACKAGE_ROOT = os.path.join(
    os.path.dirname(__file__), "Python_Codes", "BLeifer_Battery_Analysis"
)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from battery_analysis.models import TestResult  # noqa: E402
from battery_analysis.utils.db import ensure_connection  # noqa: E402

PATTERN = re.compile(r"CN\d+")


def main() -> None:
    if not ensure_connection():
        print("Could not connect to database.")
        return
    updated = 0
    for test in TestResult.objects():
        if test.cell_code:
            continue
        match = PATTERN.search(test.name or "")
        if match:
            test.update(set__cell_code=match.group(0))
            updated += 1
    print(f"Updated {updated} test results")


if __name__ == "__main__":
    main()
