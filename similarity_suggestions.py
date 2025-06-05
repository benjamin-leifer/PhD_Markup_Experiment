"""Similarity suggestions for battery samples.

This module provides :func:`suggest_similar_samples`, which queries the
``battery_analysis`` MongoDB models to locate samples with comparable
electrolyte composition and cell architecture traits.  The computation uses a
Jaccard similarity measure on tokenized trait vectors and adds simple bonuses
for matching anode/cathode and proximity in test type or test date.

The function gracefully degrades to returning an empty list if the database is
not accessible.
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Tuple


def _jaccard(a: set[str], b: set[str]) -> float:
    """Return Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _extract_traits(sample) -> Tuple[str | None, str | None, List[str], List[str], datetime.datetime | None]:
    """Extract basic traits from ``sample``.

    Returns a tuple ``(anode, cathode, electrolyte_tokens, test_types, date)``.
    """
    anode = None
    cathode = None
    electrolyte_tokens: List[str] = []
    test_types: List[str] = []
    dates: List[datetime.datetime] = []

    name = getattr(sample, "name", "")
    if "-" in name and "|" in name.split("-")[0]:
        prefix, rest = name.split("-", 1)
        parts = [p.strip() for p in prefix.split("|")]
        if len(parts) >= 2:
            anode, cathode = parts[0], parts[1]
        electrolyte_tokens = [t.lower() for t in rest.replace("elyte", "").split() if t]

    # Parse chemistry string if anode/cathode still missing
    try:
        from battery_analysis.pybamm_models import parse_chemistry_string

        chemistry = getattr(sample, "chemistry", "")
        info = parse_chemistry_string(chemistry) if chemistry else {}
        anode = anode or info.get("anode_material")
        cathode = cathode or info.get("cathode_material")
    except Exception:  # pragma: no cover - optional dependency
        pass

    tests = getattr(sample, "tests", []) or []
    for t in tests:
        try:
            obj = t.fetch() if hasattr(t, "fetch") else t
        except Exception:  # pragma: no cover - lazy ref without DB
            obj = None
        if obj is None:
            continue
        test_types.append(getattr(obj, "test_type", ""))
        dt = getattr(obj, "date", None)
        if isinstance(dt, datetime.datetime):
            dates.append(dt)

    date = min(dates) if dates else None
    return anode, cathode, electrolyte_tokens, test_types, date


def _difference_summary(
    ref: Tuple[str | None, str | None, List[str], List[str], datetime.datetime | None],
    other: Tuple[str | None, str | None, List[str], List[str], datetime.datetime | None],
) -> str:
    """Return a short description of how ``other`` differs from ``ref``."""
    r_anode, r_cathode, r_elyte, r_types, r_date = ref
    o_anode, o_cathode, o_elyte, o_types, o_date = other

    diffs = []
    if r_anode != o_anode:
        diffs.append(f"anode={o_anode}")
    if r_cathode != o_cathode:
        diffs.append(f"cathode={o_cathode}")
    if r_elyte != o_elyte:
        diffs.append("elyte=" + " ".join(o_elyte))
    if set(r_types) != set(o_types):
        diffs.append("test_type=" + ",".join(filter(None, o_types)))
    if r_date and o_date and r_date.date() != o_date.date():
        diffs.append("date=" + o_date.strftime("%Y-%m-%d"))
    return "; ".join(diffs)


def suggest_similar_samples(sample_id: str, N: int = 5) -> List[Dict[str, str]]:
    """Return up to ``N`` samples most similar to ``sample_id``.

    Each result dictionary has ``sample_id``, ``score`` and ``differences`` keys.
    If the required database models cannot be accessed the function returns an
    empty list.
    """

    try:
        from battery_analysis import models
    except Exception:  # pragma: no cover - models not importable
        return []

    try:
        ref = models.Sample.objects(id=sample_id).first()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - DB not reachable
        return []

    if ref is None:
        return []

    ref_traits = _extract_traits(ref)
    ref_tokens = set(ref_traits[2]) | {t.lower() for t in ref_traits[3]}

    suggestions = []
    try:
        candidates = models.Sample.objects(id__ne=sample_id)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - DB not reachable
        return []

    for cand in candidates:
        traits = _extract_traits(cand)
        tokens = set(traits[2]) | {t.lower() for t in traits[3]}
        score = _jaccard(ref_tokens, tokens)

        if traits[0] and ref_traits[0] and traits[0].lower() == ref_traits[0].lower():
            score += 1.0
        if traits[1] and ref_traits[1] and traits[1].lower() == ref_traits[1].lower():
            score += 1.0
        if traits[4] and ref_traits[4]:
            days = abs((traits[4] - ref_traits[4]).days)
            score += 1.0 / (1 + days)
        if set(traits[3]) & set(ref_traits[3]):
            score += 0.5

        suggestions.append((score, cand, traits))

    suggestions.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, cand, traits in suggestions[:N]:
        diff = _difference_summary(ref_traits, traits)
        results.append({
            "sample_id": str(cand.id),
            "score": f"{score:.3f}",
            "differences": diff,
        })

    return results
