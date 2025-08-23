import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "Python_Codes" / "BLeifer_Battery_Analysis" / "battery_analysis"

package_stub = types.ModuleType("battery_analysis")
package_stub.__path__ = [str(PACKAGE_DIR)]  # type: ignore[attr-defined]
sys.modules["battery_analysis"] = package_stub

utils_stub = types.ModuleType("battery_analysis.utils")
utils_stub.__path__ = [str(PACKAGE_DIR / "utils")]  # type: ignore[attr-defined]
sys.modules["battery_analysis.utils"] = utils_stub

# Ensure the lightweight dataclass implementations are used by forcing the
# models module to think ``mongoengine`` is unavailable.
sys.modules.setdefault("mongoengine", types.ModuleType("mongoengine"))
