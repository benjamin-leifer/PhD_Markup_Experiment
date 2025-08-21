"""Update :class:`CellDataset` documents when new data are added.

For each cell code a corresponding :class:`~battery_analysis.models.Sample`
is ensured to exist using :func:`Sample.get_or_create` before refreshing the
dataset. Run ``python update_cell_dataset_cli.py --help`` for usage details.
"""

import argparse
import os
import sys

PACKAGE_ROOT = os.path.join(
    os.path.dirname(__file__), "Python_Codes", "BLeifer_Battery_Analysis"
)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from battery_analysis.models import Sample, TestResult, CellDataset  # noqa: E402
from battery_analysis.utils.cell_dataset_builder import (  # noqa: E402
    update_cell_dataset,
    rollback,
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
    group.add_argument(
        "--list",
        metavar="CODE",
        help="List available dataset versions for the specified cell code",
    )
    group.add_argument(
        "--rollback",
        nargs=2,
        metavar=("CODE", "VERSION"),
        help="Rollback dataset for CODE to VERSION",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Print the number of distinct cell codes and exit",
    )
    args = parser.parse_args()

    from battery_analysis.utils.logging import get_logger

    logger = get_logger(__name__)

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
        logger.error("Could not connect to database.")
        return

    codes = [code for code in TestResult.objects.distinct("cell_code") if code]

    if args.count:
        logger.info("%d", len(codes))
        return

    if args.cell:
        Sample.get_or_create(args.cell)  # Ensure the sample exists
        update_cell_dataset(args.cell)
        logger.info("Updated dataset for %s", args.cell)
        return

    if args.all:
        for code in codes:
            Sample.get_or_create(code)  # Ensure the sample exists
            update_cell_dataset(code)
        logger.info("Updated datasets for all cell codes")
        return

    if args.list:
        datasets = CellDataset.objects(cell_code=args.list).order_by("version")
        for ds in datasets:
            logging.info("version %d: %s", ds.version, ds.id)
        return

    if args.rollback:
        code, ver = args.rollback[0], int(args.rollback[1])
        Sample.get_or_create(code)
        ds = rollback(code, ver)
        if ds:
            logging.info("Rolled back %s to version %d", code, ver)
        else:
            logging.error("Version %d not found for %s", ver, code)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
