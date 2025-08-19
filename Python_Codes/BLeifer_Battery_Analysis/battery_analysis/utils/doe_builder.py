"""Utilities for building and persisting design-of-experiments matrices."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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


def export_csv(plan: ExperimentPlan, file_path: str | Path) -> Path:
    """Write the plan matrix to ``file_path`` as CSV and return the path."""

    path = Path(file_path)
    if not plan.matrix:
        path.write_text("")
        return path
    fieldnames = sorted(plan.matrix[0].keys())
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in plan.matrix:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def export_pdf(plan: ExperimentPlan, file_path: str | Path) -> Path:
    """Generate a simple PDF summary of ``plan`` and return the path."""

    path = Path(file_path)
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    y = height - 72
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, f"Experiment Plan: {plan.name}")
    y -= 24
    headers = sorted(plan.matrix[0].keys()) if plan.matrix else []
    if headers:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(72, y, ", ".join(headers))
        y -= 20
        c.setFont("Helvetica", 12)
        for row in plan.matrix:
            line = ", ".join(str(row.get(h, "")) for h in headers)
            c.drawString(72, y, line)
            y -= 15
            if y < 72:
                c.showPage()
                y = height - 72
    c.save()
    return path


def remaining_combinations(plan: ExperimentPlan) -> List[Dict[str, Any]]:
    """Return matrix rows that do not yet have associated tests."""

    return [entry for entry in plan.matrix if not entry.get("tests")]


def status_report(plan_id: str) -> List[Dict[str, Any]]:
    """Print and return remaining combinations for ``plan_id``.

    Parameters
    ----------
    plan_id:
        Identifier or name of the :class:`ExperimentPlan`.
    """

    plan = None
    try:  # pragma: no cover - requires database
        plan = ExperimentPlan.objects(id=plan_id).first()
    except Exception:
        try:
            plan = ExperimentPlan.objects(name=plan_id).first()
        except Exception:
            plan = ExperimentPlan.get_by_name(plan_id)

    if not plan:
        print(f"No experiment plan found with id or name '{plan_id}'")
        return []

    remaining = remaining_combinations(plan)
    print(f"Experiment Plan: {plan.name}")
    print(f"Total combinations: {len(plan.matrix)}")
    print(f"Remaining combinations: {len(remaining)}")
    for combo in remaining:
        print(combo)
    return remaining


def main(argv: list[str] | None = None) -> List[Dict[str, Any]]:
    """Command line entry point for DOE building."""

    parser = argparse.ArgumentParser(description="Build DOE combinations")
    parser.add_argument("--name", help="Name of the experiment plan")
    parser.add_argument(
        "--factors",
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
    parser.add_argument(
        "--csv",
        help="Path to write a CSV summary of the plan",
    )
    parser.add_argument(
        "--pdf",
        help="Path to write a PDF summary of the plan",
    )
    parser.add_argument("--status", help="Show status for an existing plan")

    args = parser.parse_args(argv)
    if args.status:
        return status_report(args.status)

    if not args.name or not args.factors:
        parser.error("--name and --factors required unless --status is provided")

    factors = json.loads(args.factors)
    matrix = generate_combinations(factors)

    if args.preview:
        for combo in matrix:
            print(combo)

    plan = None
    if args.save or args.csv or args.pdf:
        plan = save_plan(args.name, factors, matrix, sample_ids=args.samples)
    if args.csv and plan is not None:
        export_csv(plan, args.csv)
    if args.pdf and plan is not None:
        export_pdf(plan, args.pdf)

    return matrix


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
