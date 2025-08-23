import json
from io import BytesIO
from typing import List, Any, Sequence

import pandas as pd


def _sample_to_record(sample: Any, include_metadata: bool) -> dict:
    """Convert a sample dict or object to a flat record."""
    if isinstance(sample, dict):
        obj = sample.get("sample_obj")
        name = sample.get("name") or getattr(obj, "name", None)
        date = (
            sample.get("date")
            or sample.get("created_at")
            or getattr(obj, "created_at", None)
        )
        chemistry = sample.get("chemistry") or getattr(obj, "chemistry", None)
        manufacturer = sample.get("manufacturer") or getattr(obj, "manufacturer", None)
        capacity = sample.get("capacity") or getattr(obj, "avg_final_capacity", None)
        resistance = sample.get("resistance") or getattr(obj, "median_internal_resistance", None)
        ce = sample.get("ce") or getattr(obj, "avg_coulombic_eff", None)
        tags = sample.get("tags") or getattr(obj, "tags", None)
    else:
        obj = sample
        name = getattr(obj, "name", None)
        date = getattr(obj, "created_at", None)
        chemistry = getattr(obj, "chemistry", None)
        manufacturer = getattr(obj, "manufacturer", None)
        capacity = getattr(obj, "avg_final_capacity", None)
        resistance = getattr(obj, "median_internal_resistance", None)
        ce = getattr(obj, "avg_coulombic_eff", None)
        tags = getattr(obj, "tags", None)

    record = {"test_id": name}
    if date is not None:
        if hasattr(date, "isoformat"):
            record["date"] = date.isoformat()
        else:
            record["date"] = date
    if include_metadata:
        record.update({"chemistry": chemistry, "manufacturer": manufacturer})
    record.update({"capacity": capacity, "resistance": resistance, "ce": ce})
    if tags:
        record["tags"] = ", ".join(tags)
    return record


def export_filtered_results(samples: Sequence[Any], format: str = "csv", include_metadata: bool = True) -> Any:
    """Export filtered sample data in the requested format.

    Parameters
    ----------
    samples:
        Sequence of sample dicts or objects.
    format:
        Output format - ``"csv"``, ``"excel"`` or ``"json"``.
    include_metadata:
        Include trait metadata like chemistry and manufacturer.
    """
    records = [_sample_to_record(s, include_metadata) for s in samples]
    df = pd.DataFrame(records)

    if format == "csv":
        return df.to_csv(index=False)
    if format == "excel":
        try:  # Ensure optional dependency is present
            import openpyxl  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment specific
            raise RuntimeError("Excel export requires openpyxl") from exc

        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        return buffer.getvalue()
    if format == "json":
        return df.to_json(orient="records")
    raise ValueError(f"Unsupported format: {format}")
