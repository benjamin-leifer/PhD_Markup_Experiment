#!/usr/bin/env python
"""Minimal usage example for the ``battery_analysis`` package.

This script demonstrates how to:
1. Parse a simple data file
2. Create a sample and attach the parsed test
3. Run a basic analysis function

The example connects to an in-memory MongoDB instance using ``mongomock`` so it
can run without a real MongoDB server.
"""

import logging
from pathlib import Path

from mongoengine import connect
import mongomock

# Configure logging before importing the package so our settings take effect
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

# Import package pieces after setting up logging
from battery_analysis import models, analysis
from battery_analysis.parsers import parse_file


def main() -> None:
    """Run the minimal example."""
    # Connect to an in-memory MongoDB instance
    connect(
        "example_db",
        host="mongodb://localhost",
        mongo_client_class=mongomock.MongoClient,
    )

    # Create a tiny dummy CSV file to parse
    dummy_path = Path("example_dummy.csv")
    dummy_path.write_text("cycle,data\n1,42\n")

    # Parse the file. The default parser returns a small cycles list
    cycles, metadata = parse_file(str(dummy_path))

    # Create a sample and attach the parsed test result
    sample = models.Sample(name="Demo_Sample", chemistry="NMC")
    sample.save()

    test = analysis.create_test_result(
        sample,
        cycles,
        tester=metadata.get("tester", "Other"),
        metadata=metadata,
    )

    # Run a simple analysis function
    metrics = analysis.compute_metrics(cycles)
    logging.info("Computed metrics: %s", metrics)
    logging.info("Created test '%s' for sample '%s'", test.name, sample.name)


if __name__ == "__main__":
    main()
