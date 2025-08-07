"""CLI to seed standard test protocols into MongoDB."""

from __future__ import annotations

import argparse
from typing import Dict, List

from typing import Sequence

from mongoengine import connect

try:  # Allow running as package or script
    from .models import TestProtocol  # type: ignore
except Exception:  # pragma: no cover - fallback when imported as script
    from models import TestProtocol  # type: ignore


def summarize_protocol(c_rates: Sequence[float]) -> str:
    """Summarize a list of C-rates into a compact protocol string."""
    if not c_rates:
        return "Unknown"
    rounded = [round(r, 3) for r in c_rates]
    summary_parts = []
    current_rate = rounded[0]
    count = 1
    for rate in rounded[1:]:
        if abs(rate - current_rate) < 1e-9:
            count += 1
        else:
            summary_parts.append(f"{count}x@{current_rate}C")
            current_rate = rate
            count = 1
    summary_parts.append(f"{count}x@{current_rate}C")
    return "-".join(summary_parts)


# Commonly used protocol definitions. Values are lists of C-rates.
STANDARD_PROTOCOLS: Dict[str, List[float]] = {
    "Formation": [0.1, 0.1, 0.1],
    "Standard 1C": [1.0, 1.0, 1.0],
}


def seed_standard_protocols() -> None:
    """Insert :data:`STANDARD_PROTOCOLS` into the database if missing."""
    for name, c_rates in STANDARD_PROTOCOLS.items():
        summary = summarize_protocol(c_rates)
        proto = TestProtocol.objects(name=name).first()
        if not proto:
            TestProtocol(name=name, summary=summary, c_rates=c_rates).save()


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Seed standard test protocols")
    parser.add_argument("--db", default="battery", help="MongoDB database name")
    parser.add_argument(
        "--host", default="mongodb://localhost", help="MongoDB connection URI"
    )
    args = parser.parse_args(argv)
    connect(args.db, host=args.host)
    seed_standard_protocols()
    print("Standard protocols seeded.")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
