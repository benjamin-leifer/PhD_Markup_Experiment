"""Model definitions for the battery_analysis package.

This module attempts to import the real MongoEngine based models. If
MongoEngine is not installed (for example when running the lightweight unit
tests in this repository), simple dataclass based stand-ins are provided so
that the rest of the package can be imported without requiring MongoDB or
MongoEngine.
"""

from __future__ import annotations

# Try to import the real MongoEngine models
try:  # pragma: no cover - behaviour depends on environment
    from mongoengine import Document, EmbeddedDocument  # type: ignore
    from mongoengine import fields, CASCADE  # type: ignore

    from .cycle_summary import CycleSummary  # type: ignore
    from .sample import Sample  # type: ignore
    from .testresult import TestResult, CycleDetailData  # type: ignore
    from .raw_file import RawDataFile  # type: ignore

    __all__ = [
        "Sample",
        "TestResult",
        "CycleSummary",
        "RawDataFile",
        "CycleDetailData",
    ]
except Exception:  # pragma: no cover - executed when mongoengine is missing
    # Provide very small dataclass implementations used in tests
    from dataclasses import dataclass, field as dc_field

    @dataclass
    class CycleSummary:  # type: ignore
        cycle_index: int
        charge_capacity: float
        discharge_capacity: float
        coulombic_efficiency: float

    @dataclass
    class TestResult:  # type: ignore
        initial_capacity: float = 0.0
        final_capacity: float = 0.0
        capacity_retention: float = 0.0
        avg_coulombic_eff: float = 0.0
        created_by: str | None = None
        last_modified_by: str | None = None
        notes_log: list = dc_field(default_factory=list)

    @dataclass
    class Sample:  # type: ignore
        name: str
        tests: list = dc_field(default_factory=list)
        avg_initial_capacity: float | None = None
        avg_final_capacity: float | None = None
        avg_capacity_retention: float | None = None
        avg_coulombic_eff: float | None = None
        avg_energy_efficiency: float | None = None
        median_internal_resistance: float | None = None
        parent: "Sample | None" = None
        anode: "Sample | None" = None
        cathode: "Sample | None" = None
        separator: "Sample | None" = None
        electrolyte: "Sample | None" = None
        created_by: str | None = None
        last_modified_by: str | None = None
        notes_log: list = dc_field(default_factory=list)

    class RawDataFile:  # type: ignore
        pass

    class CycleDetailData:  # type: ignore
        pass

    __all__ = [
        "Sample",
        "TestResult",
        "CycleSummary",
        "RawDataFile",
        "CycleDetailData",
    ]
