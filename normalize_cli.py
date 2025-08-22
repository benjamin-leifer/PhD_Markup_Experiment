from __future__ import annotations

"""Command line interface for normalizing cell metrics."""

import argparse
from typing import Optional

import normalization_utils


def fetch_sample(cell_code: str) -> Optional[normalization_utils.Sample]:
    """Return the sample object for ``cell_code`` if available."""
    try:  # pragma: no cover - depends on battery_analysis being installed
        from battery_analysis import models

        return models.Sample.objects(cell_code=cell_code).first()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - gracefully handle missing package/DB
        return None


def format_metrics(sample: normalization_utils.Sample) -> str:
    """Return formatted normalized capacity and impedance for ``sample``."""
    cap = normalization_utils.normalize_capacity(sample)
    imp = normalization_utils.normalize_impedance(sample)
    parts = []
    if cap is not None:
        parts.append(f"capacity={cap:.3f}")
    if imp is not None:
        parts.append(f"impedance={imp:.3f}")
    return " ".join(parts)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Print normalized capacity and impedance for given cell codes",
    )
    parser.add_argument("cell_codes", nargs="+", help="Cell codes to evaluate")
    args = parser.parse_args(argv)

    for code in args.cell_codes:
        sample = fetch_sample(code)
        if not sample:
            print(f"{code}: sample not found")
            continue
        metrics = format_metrics(sample)
        print(f"{code} {metrics}".strip())


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
