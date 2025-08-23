from __future__ import annotations

"""Command line interface for sample similarity suggestions.

This utility exposes a ``suggest`` command which prints a list of samples
similar to a reference sample.  The heavy lifting is handled by
:func:`similarity_suggestions.suggest_similar_samples` which gracefully
handles missing dependencies or database connectivity and simply returns an
empty list in such cases.
"""

import argparse
from typing import List, Dict

from similarity_suggestions import suggest_similar_samples


def _format_suggestion(s: Dict[str, str]) -> str:
    """Return a human readable string for ``s``."""

    parts = [s.get("sample_id", ""), s.get("score", "")]
    diff = s.get("differences", "")
    if diff:
        parts.append(diff)
    return " ".join(part for part in parts if part)


def _cmd_suggest(sample_id: str, count: int) -> None:
    """Print suggestions for ``sample_id``."""

    suggestions: List[Dict[str, str]] = suggest_similar_samples(sample_id, count)
    if not suggestions:
        print("No similar samples found")
        return
    for s in suggestions:
        print(_format_suggestion(s))


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_suggest = sub.add_parser("suggest", help="suggest samples similar to the given id")
    p_suggest.add_argument("sample_id", help="reference sample identifier")
    p_suggest.add_argument(
        "-n", "--count", type=int, default=5, help="number of suggestions to show"
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    """Entry point for the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "suggest":
        _cmd_suggest(args.sample_id, args.count)
    else:  # pragma: no cover - argparse enforces command choices
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
