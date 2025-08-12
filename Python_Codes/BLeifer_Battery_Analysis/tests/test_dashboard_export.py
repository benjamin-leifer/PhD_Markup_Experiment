import os
import sys
import csv

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from dashboard import data_access  # noqa: E402


def test_running_tests_csv():
    csv_str = data_access.get_running_tests_csv()
    reader = csv.DictReader(csv_str.splitlines())
    assert reader.fieldnames == ["cell_id", "test_type", "timestamp"]


def test_upcoming_tests_pdf_bytes():
    pdf_bytes = data_access.get_upcoming_tests_pdf()
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
