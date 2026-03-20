from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from battery_analysis.utils.db import ensure_connection

FIXED_EPOCH = dt.datetime(1970, 1, 1)
METADATA_DATETIME_KEYS = (
    "created_at",
    "updated_at",
    "date",
    "start_time",
    "timestamp",
    "test_date",
)
OPERATOR_KEYS = ("operator", "created_by", "last_modified_by", "user", "username")
DEVICE_KEYS = ("acquisition_device", "device", "instrument", "tester")
SAMPLE_KEYS = ("sample_name", "sample_code", "sample")


@dataclass
class MigrationCounters:
    scanned: int = 0
    matched: int = 0
    changed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "scanned": self.scanned,
            "matched": self.matched,
            "changed": self.changed,
        }


@dataclass
class MigrationSummary:
    dry_run: bool
    test_results: MigrationCounters
    raw_data_files: MigrationCounters

    def as_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "test_results": self.test_results.as_dict(),
            "raw_data_files": self.raw_data_files.as_dict(),
        }


def _fetch_reference(value: Any) -> Any:
    try:
        if hasattr(value, "fetch"):
            return value.fetch()
    except Exception:
        return value
    return value


def _normalize_datetime(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is not None:
        return value.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return value


def _coerce_datetime(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return _normalize_datetime(value)
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time.min)
    generation_time = getattr(value, "generation_time", None)
    if isinstance(generation_time, dt.datetime):
        return _normalize_datetime(generation_time)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return _normalize_datetime(dt.datetime.fromisoformat(text))
        except ValueError:
            return None
    return None


def _first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _safe_metadata(obj: Any) -> dict[str, Any]:
    metadata = getattr(obj, "metadata", {}) or {}
    return dict(metadata)


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_operator_from_path(path: str | None) -> str | None:
    if not path:
        return None
    parts = [part for part in Path(path).parts if part not in (os.sep, "")]
    for index, part in enumerate(parts[:-1]):
        lowered = part.lower()
        if lowered in {"operator", "operators", "user", "users"} and index + 1 < len(parts):
            return _stringify(parts[index + 1])
        match = re.match(r"operator[-_ ]?([A-Za-z0-9.]+)$", part, re.IGNORECASE)
        if match:
            return _stringify(match.group(1))
    return None


def _extract_sample_name_from_path(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path).parent.name
    return _stringify(candidate)


def _coerce_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = re.split(r"[,;]", value)
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return []

    seen: set[str] = set()
    normalized: list[str] = []
    for item in items:
        tag = _stringify(item)
        if tag and tag not in seen:
            seen.add(tag)
            normalized.append(tag)
    return normalized


def _merge_tags(*tag_groups: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in tag_groups:
        for tag in _coerce_tags(group):
            if tag not in seen:
                seen.add(tag)
                merged.append(tag)
    return merged


def deterministic_timestamp_for_test_result(
    test_result: Any, linked_raw_files: Iterable[Any] | None = None
) -> dt.datetime:
    metadata = _safe_metadata(test_result)
    raw_files = list(linked_raw_files or [])
    candidates = [
        getattr(test_result, "created_at", None),
        getattr(test_result, "updated_at", None),
        getattr(test_result, "date", None),
    ]
    candidates.extend(metadata.get(key) for key in METADATA_DATETIME_KEYS)
    candidates.extend(getattr(raw_file, "upload_date", None) for raw_file in raw_files)
    candidates.append(getattr(test_result, "id", None))

    for candidate in candidates:
        parsed = _coerce_datetime(candidate)
        if parsed is not None:
            return parsed
    return FIXED_EPOCH


def derive_testresult_timestamp_updates(
    test_result: Any, linked_raw_files: Iterable[Any] | None = None
) -> dict[str, dt.datetime]:
    created_at = getattr(test_result, "created_at", None)
    updated_at = getattr(test_result, "updated_at", None)
    if created_at is not None and updated_at is not None:
        return {}

    timestamp = deterministic_timestamp_for_test_result(test_result, linked_raw_files)
    updates: dict[str, dt.datetime] = {}
    if created_at is None or _coerce_datetime(created_at) is None:
        updates["created_at"] = timestamp
    if updated_at is None or _coerce_datetime(updated_at) is None:
        updates["updated_at"] = timestamp
    return updates


def _resolve_sample_from_name(
    sample_name: str | None, sample_lookup: Callable[[str], Any | None] | None
) -> Any | None:
    if not sample_name or sample_lookup is None:
        return None
    try:
        return sample_lookup(sample_name)
    except Exception:
        return None


def derive_raw_data_file_updates(
    raw_file: Any,
    sample_lookup: Callable[[str], Any | None] | None = None,
) -> dict[str, Any]:
    metadata = _safe_metadata(raw_file)
    test_result = _fetch_reference(getattr(raw_file, "test_result", None))
    linked_sample = _fetch_reference(getattr(raw_file, "sample", None))
    test_sample = _fetch_reference(getattr(test_result, "sample", None)) if test_result else None

    path_candidates = [
        getattr(raw_file, "source_path", None),
        metadata.get("file_path"),
        metadata.get("source_path"),
        getattr(test_result, "file_path", None) if test_result else None,
    ]
    source_path = next((path for path in path_candidates if _stringify(path)), None)

    sample_from_metadata = _resolve_sample_from_name(
        _stringify(_first_present(metadata, SAMPLE_KEYS)), sample_lookup
    )
    sample_from_path = _resolve_sample_from_name(
        _extract_sample_name_from_path(source_path), sample_lookup
    )
    resolved_sample = linked_sample or test_sample or sample_from_metadata or sample_from_path

    operator = (
        _stringify(getattr(raw_file, "operator", None))
        or _stringify(_first_present(metadata, OPERATOR_KEYS))
        or _stringify(getattr(test_result, "created_by", None) if test_result else None)
        or _stringify(getattr(test_result, "last_modified_by", None) if test_result else None)
        or _extract_operator_from_path(source_path)
    )

    acquisition_device = (
        _stringify(getattr(raw_file, "acquisition_device", None))
        or _stringify(_first_present(metadata, DEVICE_KEYS))
        or _stringify(getattr(test_result, "tester", None) if test_result else None)
        or _stringify(getattr(raw_file, "file_type", None))
    )

    merged_tags = _merge_tags(
        getattr(raw_file, "tags", None),
        metadata.get("tags"),
        getattr(test_result, "tags", None) if test_result else None,
    )

    derived_metadata = dict(metadata)
    test_id = getattr(test_result, "id", None) if test_result else None
    sample_name = getattr(resolved_sample, "name", None) if resolved_sample else None
    defaults = {
        "filename": _stringify(getattr(raw_file, "filename", None)),
        "file_path": _stringify(source_path),
        "source_path": _stringify(getattr(raw_file, "source_path", None)),
        "test_result_id": _stringify(test_id),
        "test_name": _stringify(getattr(test_result, "name", None) if test_result else None),
        "tester": _stringify(getattr(test_result, "tester", None) if test_result else None),
        "sample_name": _stringify(sample_name),
        "sample_code": _stringify(sample_name),
        "operator": operator,
        "acquisition_device": acquisition_device,
    }
    for key, value in defaults.items():
        if value and key not in derived_metadata:
            derived_metadata[key] = value
    if merged_tags and "tags" not in derived_metadata:
        derived_metadata["tags"] = list(merged_tags)

    updates: dict[str, Any] = {}
    if linked_sample is None and resolved_sample is not None:
        updates["sample"] = resolved_sample
    if not getattr(raw_file, "operator", None) and operator:
        updates["operator"] = operator
    if not getattr(raw_file, "acquisition_device", None) and acquisition_device:
        updates["acquisition_device"] = acquisition_device
    current_tags = _coerce_tags(getattr(raw_file, "tags", None))
    if merged_tags and merged_tags != current_tags:
        updates["tags"] = merged_tags
    if derived_metadata != metadata:
        updates["metadata"] = derived_metadata
    return updates


def _save_document(document: Any) -> None:
    save = getattr(document, "save", None)
    if callable(save):
        save()


def migrate_test_results(
    test_results: Iterable[Any],
    *,
    raw_files_for_test: Callable[[Any], Iterable[Any]] | None = None,
    dry_run: bool = False,
) -> MigrationCounters:
    counters = MigrationCounters()
    for test_result in test_results:
        counters.scanned += 1
        linked_raw_files = list(raw_files_for_test(test_result) if raw_files_for_test else [])
        updates = derive_testresult_timestamp_updates(test_result, linked_raw_files)
        if not updates:
            continue
        counters.matched += 1
        if dry_run:
            counters.changed += 1
            continue
        for key, value in updates.items():
            setattr(test_result, key, value)
        _save_document(test_result)
        counters.changed += 1
    return counters


def migrate_raw_data_files(
    raw_data_files: Iterable[Any],
    *,
    sample_lookup: Callable[[str], Any | None] | None = None,
    dry_run: bool = False,
) -> MigrationCounters:
    counters = MigrationCounters()
    for raw_file in raw_data_files:
        counters.scanned += 1
        updates = derive_raw_data_file_updates(raw_file, sample_lookup=sample_lookup)
        if not updates:
            continue
        counters.matched += 1
        if dry_run:
            counters.changed += 1
            continue
        for key, value in updates.items():
            setattr(raw_file, key, value)
        _save_document(raw_file)
        counters.changed += 1
    return counters


def _sample_lookup(name: str) -> Any | None:
    from battery_analysis.models import Sample

    try:
        objects = getattr(Sample, "objects", None)
        if callable(objects):
            return Sample.objects(name=name).first()
    except Exception:
        return None
    return None


def run_migration(dry_run: bool = False) -> MigrationSummary:
    from mongoengine.queryset.visitor import Q

    from battery_analysis.models import RawDataFile, TestResult

    if not ensure_connection():
        raise RuntimeError(
            "Could not establish a MongoDB connection. Configure the database "
            "environment first or set USE_MONGO_MOCK for local verification."
        )

    test_results = list(TestResult.objects(Q(created_at=None) | Q(updated_at=None)))
    raw_data_files = list(RawDataFile.objects())

    def raw_files_for_test(test_result: Any) -> Iterable[Any]:
        return RawDataFile.objects(test_result=test_result)

    return MigrationSummary(
        dry_run=dry_run,
        test_results=migrate_test_results(
            test_results,
            raw_files_for_test=raw_files_for_test,
            dry_run=dry_run,
        ),
        raw_data_files=migrate_raw_data_files(
            raw_data_files,
            sample_lookup=_sample_lookup,
            dry_run=dry_run,
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill missing TestResult timestamps and derivable RawDataFile metadata."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many documents would change without saving updates.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the migration summary as JSON for scripting/verification.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = run_migration(dry_run=args.dry_run)
    if args.json:
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True, default=str))
    else:
        mode = "dry-run" if args.dry_run else "apply"
        print(f"Metadata backfill ({mode})")
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
