"""Utilities for building and persisting design-of-experiments matrices."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, cast

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from ..models import ExperimentPlan
from .logging import get_logger

logger = get_logger(__name__)


def load_from_csv(
    path: str | Path,
) -> Tuple[Dict[str, List[Any]], List[Dict[str, Any]]]:
    """Return factors and matrix parsed from a CSV file."""

    csv_path = Path(path)
    matrix: List[Dict[str, Any]] = []
    with csv_path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            matrix.append(dict(row))
        headers = reader.fieldnames or []
    factors: Dict[str, List[Any]] = {
        h: sorted({row.get(h) for row in matrix}) for h in headers
    }
    return factors, matrix


def load_from_json(
    path: str | Path,
) -> Tuple[Dict[str, List[Any]], List[Dict[str, Any]]]:
    """Return factors and matrix parsed from a JSON file."""

    json_path = Path(path)
    data = json.loads(json_path.read_text())
    if isinstance(data, dict) and "matrix" in data:
        matrix = list(data["matrix"])
        factors = data.get("factors")
    else:
        matrix = list(data)
        factors = None
    if not factors:
        headers: set[str] = set()
        for row in matrix:
            headers.update(row.keys())
        factors = {h: sorted({row.get(h) for row in matrix}) for h in headers}
    return factors, matrix


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

    existing: ExperimentPlan | None = None
    try:  # pragma: no cover - requires database
        existing = ExperimentPlan.get_by_name(name)
    except Exception:
        existing = None

    if existing:
        plan = existing
        # merge matrices while avoiding duplicates
        existing_rows = list(plan.matrix)
        for row in matrix:
            if row not in existing_rows:
                existing_rows.append(row)
        plan.matrix = existing_rows
        # recompute factors from merged matrix
        headers: set[str] = set()
        for row in plan.matrix:
            headers.update(row.keys())
        plan.factors = {h: sorted({r.get(h) for r in plan.matrix}) for h in headers}
        if sample_ids:
            plan.sample_ids = list(set(plan.sample_ids).union(sample_ids))
    else:
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


def export_html(plan: ExperimentPlan, file_path: str | Path) -> Path:
    """Write ``plan`` to ``file_path`` as an interactive HTML table."""

    path = Path(file_path)
    df = pd.DataFrame(plan.matrix or [])
    table_html = df.to_html(index=False, table_id="doe-table", classes="display")
    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<link rel=\"stylesheet\" href=\"https://cdn.datatables.net/1.13.5/css/jquery.dataTables.min.css\" />
<script src=\"https://code.jquery.com/jquery-3.6.0.min.js\"></script>
<script src=\"https://cdn.datatables.net/1.13.5/js/jquery.dataTables.min.js\"></script>
</head>
<body>
{table_html}
<script>
$(document).ready(function() {{
    $('#doe-table').DataTable();
}});
</script>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")
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
        logger.info("No experiment plan found with id or name '%s'", plan_id)
        return []

    remaining = remaining_combinations(plan)
    logger.info("Experiment Plan: %s", plan.name)
    logger.info("Total combinations: %s", len(plan.matrix))
    logger.info("Remaining combinations: %s", len(remaining))
    for combo in remaining:
        logger.info("%s", combo)
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
        "--input",
        help="Path to CSV or JSON file with factor matrix",
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
    parser.add_argument(
        "--html",
        nargs="?",
        const="",
        help="Write an interactive HTML summary (default docs/doe_plans/<name>.html)",
    )
    parser.add_argument("--status", help="Show status for an existing plan")

    args = parser.parse_args(argv)
    if args.status:
        return status_report(args.status)

    if not args.name:
        parser.error("--name required")

    factors: Dict[str, Iterable[Any]]
    matrix: List[Dict[str, Any]]
    if args.input:
        input_path = Path(args.input)
        if input_path.suffix.lower() == ".csv":
            factors_dict, matrix = load_from_csv(input_path)
            factors = cast(Dict[str, Iterable[Any]], factors_dict)
        else:
            factors_dict, matrix = load_from_json(input_path)
            factors = cast(Dict[str, Iterable[Any]], factors_dict)
    else:
        if not args.factors:
            parser.error("--factors or --input required unless --status is provided")
        factors = cast(Dict[str, Iterable[Any]], json.loads(args.factors))
        matrix = generate_combinations(factors)

    if args.preview:
        logger.info("Generated combinations:")
        for combo in matrix:
            logger.info("%s", combo)

    plan = None
    if args.save or args.csv or args.pdf or args.html is not None:
        plan = save_plan(args.name, factors, matrix, sample_ids=args.samples)
    if args.csv and plan is not None:
        export_csv(plan, args.csv)
    if args.pdf and plan is not None:
        export_pdf(plan, args.pdf)
    if args.html is not None and plan is not None:
        output = (
            Path(args.html)
            if args.html
            else Path("docs") / "doe_plans" / f"{plan.name}.html"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        export_html(plan, output)

    return matrix


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
