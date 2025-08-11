import sys
from pathlib import Path

import mongomock
import pytest
from mongoengine import connect, disconnect

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import battery_analysis.advanced_analysis as advanced_analysis  # noqa: E402
import battery_analysis.models as models  # noqa: E402
from battery_analysis.gui.advanced_analysis_tab import AdvancedAnalysisTab  # noqa: E402
from types import SimpleNamespace


def _create_test_result():
    sample = models.Sample(name="S").save()
    cycles = [
        models.CycleSummary(
            cycle_index=i,
            charge_capacity=100.0,
            discharge_capacity=100 - i,
            coulombic_efficiency=1.0,
        )
        for i in range(10)
    ]
    test = models.TestResult(
        sample=sample,
        tester="Arbin",
        cycles=cycles,
        cycle_count=len(cycles),
    ).save()
    return test


def test_custom_eol_and_models():
    disconnect()
    connect("testdb", mongo_client_class=mongomock.MongoClient, alias="default")
    try:
        test = _create_test_result()
        result = advanced_analysis.capacity_fade_analysis(
            str(test.id), eol_percent=70, models=["linear"]
        )
        assert set(result["fade_models"].keys()) == {"linear"}
        assert result["predicted_eol_cycle"] == pytest.approx(30)
        assert result["eol_percent"] == 70

        # Verify GUI text reflects custom EOL percentage without using Tk
        class DummyText:
            def __init__(self):
                self.content = ""

            def config(self, **kwargs):
                pass

            def delete(self, *args, **kwargs):
                self.content = ""

            def insert(self, index, text, tag=None):
                self.content += text

            def tag_configure(self, *args, **kwargs):
                pass

        tab = AdvancedAnalysisTab.__new__(AdvancedAnalysisTab)
        tab.results_text = DummyText()
        tab.current_sample = SimpleNamespace(name="S")
        tab.current_test = SimpleNamespace(name="T")
        tab.current_fade_analysis = result
        AdvancedAnalysisTab.update_fade_results_text(tab)
        assert "70%" in tab.results_text.content
    finally:
        disconnect()

