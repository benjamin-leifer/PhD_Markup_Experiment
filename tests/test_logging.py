import sys
import types
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis"
for p in (ROOT, PACKAGE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Stub minimal battery_analysis package to avoid heavy imports
battery_pkg = types.ModuleType("battery_analysis")
utils_pkg = types.ModuleType("battery_analysis.utils")
sys.modules.setdefault("battery_analysis", battery_pkg)
sys.modules.setdefault("battery_analysis.utils", utils_pkg)

spec = importlib.util.spec_from_file_location(
    "battery_analysis.utils.logging",
    PACKAGE_ROOT / "battery_analysis" / "utils" / "logging.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
sys.modules["battery_analysis.utils.logging"] = module

get_logger = module.get_logger

from dashboard import logs_tab


def test_file_logger_writes(tmp_path: Path) -> None:
    log_file = tmp_path / "test.log"
    logger = get_logger("test_logger", log_file=log_file)
    logger.info("hello world")
    for h in logger.handlers:
        h.flush()
    assert "hello world" in log_file.read_text()


def test_logs_tab_reads_file(tmp_path: Path, monkeypatch) -> None:
    log_file = tmp_path / "app.log"
    log_file.write_text("line1\nline2\n")
    monkeypatch.setattr(logs_tab, "LOG_FILE", log_file)
    content = logs_tab._tail(log_file)
    assert "line2" in content
    layout = logs_tab.layout()
    ids = {c.id for c in layout.children if hasattr(c, "id")}
    assert "logs-interval" in ids
