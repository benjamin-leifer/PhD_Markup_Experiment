"""Update :class:`CellDataset` documents when new data are added."""

import argparse
import logging
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Refresh cell datasets or report available cell codes."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--cell",
        metavar="CODE",
        help="Refresh dataset for the specified cell code",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Refresh datasets for all cell codes",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Print the number of distinct cell codes and exit",
    )
    args = parser.parse_args()

    if not ensure_connection():
        logger.error("Could not connect to database.")
        return

    codes = [code for code in TestResult.objects.distinct("cell_code") if code]

    if args.count:
        logger.info(str(len(codes)))
        return

    if args.cell:
        update_cell_dataset(args.cell)
        logger.info("Updated dataset for %s", args.cell)
        return

    if args.all:
        for code in codes:
            update_cell_dataset(code)
        logger.info("Updated datasets for all cell codes")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
