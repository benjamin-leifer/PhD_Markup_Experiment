"""
Battery Test Data Analysis Package

A modular and extensible Python package for electrochemical (battery) test data
management, supporting data ingestion, analysis, and reporting for multiple file formats.

Main components:
- Data Models: MongoEngine document classes for samples and test results
- Data Parsers: Parsers for Arbin and BioLogic file formats
- Analysis: Functions to compute metrics and propagate properties
- Advanced Analysis: Advanced electrochemical techniques like dQ/dV, capacity fade modeling, and anomaly detection
- EIS: Electrochemical Impedance Spectroscopy analysis and modeling
- Reporting: PDF report generation for test results

Basic Usage:
    from battery_analysis import utils, models, analysis
    from battery_analysis.parsers import parse_file

    # Connect to MongoDB
    utils.connect_to_database('battery_db')

    # Create a sample
    sample = models.Sample(name="Sample_123", chemistry="NMC")
    sample.save()

    # Parse a test file
    cycles, metadata = parse_file('path/to/test_file.csv')

    # Create test result and attach to sample
    test = analysis.create_test_result(sample, cycles, tester="Arbin", metadata=metadata)

    # Generate a report
    from battery_analysis import report
    report.generate_report(test, filename="test_report.pdf")

    # Perform advanced analysis
    from battery_analysis import advanced_analysis
    fade_analysis = advanced_analysis.capacity_fade_analysis(test.id)
    anomalies = advanced_analysis.detect_anomalies(test.id)

    # Analyze EIS data
    from battery_analysis import eis
    eis_data = eis.import_eis_data('eis_file.csv', sample_id=sample.id)
    eis.plot_nyquist(eis_data, filename="nyquist.png")
"""

# This module exposes the main API for the ``battery_analysis`` package and
# performs optional imports. The heavy use of dynamic imports means we provide
# explicit typing information for variables that may be set to ``None`` when
# optional dependencies are unavailable.

from types import ModuleType
from typing import Any, Callable, cast
import warnings

# Public objects that may be populated dynamically.
advanced_analysis: ModuleType | None
plots: ModuleType | None
eis: ModuleType | None
outlier_analysis: ModuleType | None

# Package version
__version__ = "0.1.0"

# Import main components for easier access
try:
    from . import models
    from . import analysis
    from . import report
    from . import utils
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    models = importlib.import_module("models")
    analysis = importlib.import_module("analysis")
    report = importlib.import_module("report")
    utils = importlib.import_module("utils")

MISSING_ADVANCED_PACKAGES: list[str] = []


def _check_missing_advanced_packages() -> list[str]:
    """Return a list of optional packages required for advanced analysis that
    are not installed."""

    missing: list[str] = []
    try:  # pragma: no cover - runtime check
        import scipy  # noqa: F401
    except Exception:  # pragma: no cover - dependency may be absent
        missing.append("scipy")

    try:  # pragma: no cover - runtime check
        import sklearn  # noqa: F401
    except Exception:  # pragma: no cover - dependency may be absent
        missing.append("scikit-learn")

    return missing


try:  # Some optional dependencies like scipy may be missing
    from . import advanced_analysis
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    advanced_analysis = importlib.import_module("advanced_analysis")
except Exception as exc:  # pragma: no cover - optional import
    advanced_analysis = None
    MISSING_ADVANCED_PACKAGES = _check_missing_advanced_packages()
    if MISSING_ADVANCED_PACKAGES:
        warnings.warn(
            "Advanced analysis disabled. Missing packages: "
            + ", ".join(MISSING_ADVANCED_PACKAGES)
        )
    else:
        warnings.warn(f"Advanced analysis disabled due to error: {exc}")

try:
    from . import plots
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    plots = importlib.import_module("plots")
except Exception:  # pragma: no cover - optional import
    plots = None

try:
    from . import eis
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    eis = importlib.import_module("eis")
except Exception:  # pragma: no cover - optional import
    eis = None

try:
    from . import outlier_analysis
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    outlier_analysis = importlib.import_module("outlier_analysis")
except Exception:  # pragma: no cover - optional import
    outlier_analysis = None

# Make parsers accessible
try:
    from .parsers import parse_file as _parse_file
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    _parse_file = importlib.import_module("parsers").parse_file

parse_file = cast(Callable[..., Any], _parse_file)
