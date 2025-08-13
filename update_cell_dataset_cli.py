"""Update :class:`CellDataset` documents when new data are added.

For each cell code a corresponding :class:`~battery_analysis.models.Sample`
is ensured to exist using :func:`Sample.get_or_create` before refreshing the
dataset. Run ``python update_cell_dataset_cli.py --help`` for usage details.
"""

import argparse
import logging
import os
import sys

PACKAGE_ROOT = os.path.join(
    os.path.dirname(__file__), "Python_Codes", "BLeifer_Battery_Analysis"
)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from battery_analysis.models import Sample, TestResult  # noqa: E402
from battery_analysis.utils.cell_dataset_builder import (  # noqa: E402
    update_cell_dataset,
)
from Mongodb_implementation import get_client  # noqa: E402
from mongoengine import connect  # noqa: E402


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

    logging.basicConfig(level=logging.INFO)

    client = get_client()
    db_name = os.getenv("BATTERY_DB_NAME", "battery_test_db")
    try:
        uri = getattr(client, "_configured_uri", None)
        if uri:
            connect(
                db_name,
                host=uri,
                alias="default",
                serverSelectionTimeoutMS=2000,
            )
        else:
            connect(
                db_name,
                host=getattr(client, "_configured_host", "localhost"),
                port=getattr(client, "_configured_port", 27017),
                alias="default",
                serverSelectionTimeoutMS=2000,
            )
    except Exception:
        logging.error("Could not connect to database.")
        return

    codes = [code for code in TestResult.objects.distinct("cell_code") if code]

    if args.count:
        logging.info("%d", len(codes))
        return

    if args.cell:
        Sample.get_or_create(args.cell)  # Ensure the sample exists
        update_cell_dataset(args.cell)
        logging.info("Updated dataset for %s", args.cell)
        return

    if args.all:
        for code in codes:
            Sample.get_or_create(code)  # Ensure the sample exists
            update_cell_dataset(code)
        logging.info("Updated datasets for all cell codes")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
