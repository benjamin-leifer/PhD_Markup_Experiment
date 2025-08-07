import os
import sys

# Ensure package root on path
TESTS_DIR = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from mongoengine import connect, disconnect  # noqa: E402
import mongomock  # noqa: E402

from battery_analysis import models  # noqa: E402
from battery_analysis.utils import data_update  # noqa: E402


def test_sequential_file_names_match():
    """Sequential files with suffixes like _Wb_1 should map to the same test."""
    disconnect()
    connect("testdb", mongo_client_class=mongomock.MongoClient)
    try:
        sample = models.Sample(name="S1")
        sample.save()
        test = models.TestResult(
            sample=sample.id,
            tester="Arbin",
            name="BL-LL-EU02_RT_Rate_Test_Channel_30",
            file_path="/tmp/BL-LL-EU02_RT_Rate_Test_Channel_30_Wb_1.xlsx",
            cycles=[],
        )
        test.save()
        sample.update(push__tests=test.id)
        sample.reload()

        metadata = {"name": "BL-LL-EU02_RT_Rate_Test_Channel_30", "tester": "Arbin"}
        identifiers = data_update.extract_test_identifiers(
            "/tmp/BL-LL-EU02_RT_Rate_Test_Channel_30_Wb_2.xlsx",
            [],
            metadata,
        )
        matches = data_update.find_matching_tests(identifiers, sample.id)
        assert matches and matches[0].id == test.id
    finally:
        disconnect()
