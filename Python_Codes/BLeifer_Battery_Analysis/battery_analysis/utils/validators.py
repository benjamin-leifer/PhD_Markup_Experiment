# mypy: ignore-errors
"""Pydantic models for validating parser outputs."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Extra


class TestMetadataModel(BaseModel):
    """Schema for required test metadata fields."""

    tester: str
    name: str
    date: datetime

    class Config:
        extra = Extra.allow


class CycleSummaryModel(BaseModel):
    """Schema for per-cycle summary data returned by parsers."""

    cycle_index: int
    charge_capacity: float
    discharge_capacity: float
    coulombic_efficiency: float
    charge_energy: float | None = None
    discharge_energy: float | None = None
    energy_efficiency: float | None = None
    internal_resistance: float | None = None
    has_detailed_data: bool | None = None

    class Config:
        extra = Extra.allow


__all__ = ["TestMetadataModel", "CycleSummaryModel"]
