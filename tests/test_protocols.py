from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mongomock  # noqa: E402
import pytest  # noqa: E402
from mongoengine import ValidationError, connect, disconnect  # noqa: E402

from battery_analysis import models  # noqa: E402
from battery_analysis.seed_protocols import (  # noqa: E402
    STANDARD_PROTOCOLS,
    seed_standard_protocols,
    summarize_protocol,
)


def setup_db():
    disconnect()
    connect("testdb", mongo_client_class=mongomock.MongoClient)


def teardown_db():
    disconnect()


def test_protocol_auto_assignment():
    setup_db()
    try:
        sample = models.Sample(name="S1")
        sample.save()
        c_rates = [1.0, 1.0, 1.0]
        summary = summarize_protocol(c_rates)
        proto = models.TestProtocol(
            name="Standard 1C", summary=summary, c_rates=c_rates
        )
        proto.save()
        test = models.TestResult(
            sample=sample.id, tester="Arbin", c_rates=c_rates, cycles=[]
        )
        test.save()
        test.reload()
        assert test.protocol.id == proto.id
    finally:
        teardown_db()


def test_protocol_validation():
    setup_db()
    try:
        sample = models.Sample(name="S1")
        sample.save()
        fake_proto = models.TestProtocol(name="Fake", summary="Fake", c_rates=[])
        test = models.TestResult(
            sample=sample.id, tester="Arbin", protocol=fake_proto, cycles=[]
        )
        with pytest.raises(ValidationError):
            test.save()
    finally:
        teardown_db()


def test_seed_standard_protocols():
    setup_db()
    try:
        seed_standard_protocols()
        count = models.TestProtocol.objects.count()
        assert count == len(STANDARD_PROTOCOLS)
    finally:
        teardown_db()
