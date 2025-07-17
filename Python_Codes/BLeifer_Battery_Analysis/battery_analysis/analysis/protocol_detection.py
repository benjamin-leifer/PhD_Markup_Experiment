"""Utilities to infer cycling protocol details from parsed data."""
from __future__ import annotations

from typing import List, Sequence, Optional

import numpy as np

from battery_analysis.models import TestProtocol, TestResult, Sample


def is_last_cycle_complete(cycles: Sequence[dict]) -> bool:
    """Return True if the final cycle appears complete."""
    if not cycles:
        return False
    last = cycles[-1]
    return (last.get("charge_capacity", 0) > 0) and (last.get("discharge_capacity", 0) > 0)


def calculate_cycle_crates(cycles: Sequence[dict], nominal_capacity: Optional[float]) -> List[float]:
    """Estimate C-rate of each cycle using peak current if available."""
    if nominal_capacity is None or nominal_capacity == 0:
        return []
    rates: List[float] = []
    for cycle in cycles:
        currents = []
        if "current_charge" in cycle and cycle["current_charge"]:
            currents.extend(np.abs(cycle["current_charge"]))
        if "current_discharge" in cycle and cycle["current_discharge"]:
            currents.extend(np.abs(cycle["current_discharge"]))
        if currents:
            rate = float(max(currents)) / float(nominal_capacity)
            rates.append(rate)
    return rates


def summarize_protocol(c_rates: Sequence[float]) -> str:
    """Summarize a list of C-rates into a compact protocol string."""
    if not c_rates:
        return "Unknown"
    rounded = [round(r, 3) for r in c_rates]
    summary_parts = []
    current_rate = rounded[0]
    count = 1
    for rate in rounded[1:]:
        if np.isclose(rate, current_rate):
            count += 1
        else:
            summary_parts.append(f"{count}x@{current_rate}C")
            current_rate = rate
            count = 1
    summary_parts.append(f"{count}x@{current_rate}C")
    return "-".join(summary_parts)


def get_or_create_protocol(summary: str, *, prompt: bool = False) -> Optional[TestProtocol]:
    """Find an existing protocol by summary or create a new one."""
    proto = TestProtocol.objects(summary=summary).first()  # type: ignore[attr-defined]
    if proto:
        return proto
    if not prompt:
        return None
    name = input(f"Enter name for new protocol '{summary}': ").strip() or summary
    proto = TestProtocol(name=name, summary=summary)
    proto.save()
    return proto


def detect_and_update_test_protocol(test: TestResult, cycles: Sequence[dict], *, prompt: bool = False) -> None:
    """Populate protocol-related fields on ``test`` from cycle data."""
    test.last_cycle_complete = is_last_cycle_complete(cycles)
    nominal_capacity = None
    if isinstance(test.sample, Sample) or hasattr(test.sample, "fetch"):
        try:
            sample = test.sample.fetch() if hasattr(test.sample, "fetch") else test.sample
            nominal_capacity = getattr(sample, "nominal_capacity", None)
        except Exception:
            nominal_capacity = None
    test.c_rates = calculate_cycle_crates(cycles, nominal_capacity)
    summary = summarize_protocol(test.c_rates)
    proto = get_or_create_protocol(summary, prompt=prompt)
    if proto:
        test.protocol = proto

