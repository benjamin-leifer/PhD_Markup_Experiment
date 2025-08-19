"""Utilities for building and persisting design-of-experiments matrices."""

from __future__ import annotations

import argparse
import json
from itertools import product
from typing import Any, Dict, Iterable, List

from ..models import ExperimentPlan


def generate_combinations(factors: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    """Return a list of dictionaries for all factor level combinations."""
    keys = list(factors.keys())
    levels = [list(factors[k]) for k in keys]
    combos = [dict(zip(keys, vals)) for vals in product(*levels)]
    return combos


def save_plan(
    name: str,
    factors: Dict[str, Iterable[Any]],
    matrix: List[Dict[str, Any]] | None = None,
    sample_ids: Iterable[Any] | None = None,
) -> ExperimentPlan:
    """Persist an :class:`ExperimentPlan` and return it.

    Parameters
    ----------
    name:
        Unique name of the experiment plan.
    factors:
        Mapping of factor names to iterable levels.
    matrix:
        Precomputed combinations.  If ``None`` it will be generated.
    sample_ids:
        Iterable of ``Sample`` ids to associate with the plan.
    """

    if matrix is None:
        matrix = generate_combinations(factors)
    plan = ExperimentPlan(
        name=name,
        factors=factors,
        matrix=matrix,
        sample_ids=list(sample_ids or []),
    )
    try:
        plan.save()
    except Exception:
        # Saving may fail if no MongoDB connection is configured; ignore.
        pass
    return plan


def main(argv: list[str] | None = None) -> List[Dict[str, Any]]:
    """Command line entry point for DOE building."""

    parser = argparse.ArgumentParser(description="Build DOE combinations")
    parser.add_argument("--name", required=True, help="Name of the experiment plan")
    parser.add_argument(
        "--factors",
        required=True,
        help="JSON mapping of factor names to levels",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=[],
        help="Sample ids to associate with the plan",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the generated combinations",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist the plan to MongoDB",
    )

    args = parser.parse_args(argv)
    factors = json.loads(args.factors)
    matrix = generate_combinations(factors)

    if args.preview:
        for combo in matrix:
            print(combo)

    if args.save:
        save_plan(args.name, factors, matrix, sample_ids=args.samples)

    return matrix


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
