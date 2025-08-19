# mypy: ignore-errors
"""Utilities for handling data updates and file processing."""

import os
import logging
import re
import hashlib
import datetime
from mongoengine.queryset.visitor import Q

# Update imports to avoid circular dependencies
from battery_analysis import models, parsers
from battery_analysis.utils.db import ensure_connection

# Import directly from the analysis package
from battery_analysis.analysis import (
    compute_metrics,
    update_sample_properties,
    create_test_result,
    summarize_detailed_cycles,
)


def _normalize_identifier(name: str | None) -> str | None:
    """Return a normalized identifier with sequential suffixes removed.

    Some tester exports generate multiple files for a single test where the
    file names differ only by a trailing indicator such as ``_Wb_1``.  Stripping
    this pattern allows us to match related files and append their data rather
    than treating each as a separate test.

    Args:
        name: Raw file or test name.

    Returns:
        The name with common sequential suffixes removed, or ``None`` if no
        name was provided.
    """

    if not name:
        return None

    base = os.path.splitext(name)[0]
    # Remove patterns like "_Wb_1" or "_wb_2" at the end of the name
    base = re.sub(r"_wb_\d+$", "", base, flags=re.IGNORECASE)
    return base


def extract_test_identifiers(file_path, parsed_data, metadata):
    """
    Extract identifying information from a test file.

    Args:
        file_path: Path to the test file
        parsed_data: Parsed data from the file
        metadata: Metadata from the file

    Returns:
        dict: Dictionary of identifiers
    """
    file_name = os.path.basename(file_path)
    identifiers = {
        "file_name": file_name,
        "tester": metadata.get("tester", None),
        "test_name": metadata.get("name", None),
        # Normalized versions for matching sequential files
        "base_file_name": _normalize_identifier(file_name),
    }

    if identifiers["test_name"]:
        identifiers["base_test_name"] = _normalize_identifier(identifiers["test_name"])

    # Add more identifying information as available
    if "date" in metadata:
        identifiers["date"] = metadata["date"]

    if "test_id" in metadata:
        identifiers["test_id"] = metadata["test_id"]

    if "file_hash" in metadata:
        identifiers["file_hash"] = metadata["file_hash"]

    return identifiers


def find_matching_tests(identifiers, sample_id):
    """
    Find tests that match the given identifiers.

    Args:
        identifiers: Dictionary of test identifiers
        sample_id: ID of the sample to search within

    Returns:
        list: List of matching TestResult objects
    """
    sample = models.Sample.objects(id=sample_id).first()
    if not sample:
        return []

    query = models.TestResult.objects(sample=sample_id)
    conditions = []

    for field in ["test_id", "file_hash", "base_test_name", "base_file_name"]:
        value = identifiers.get(field)
        if value:
            conditions.append(Q(**{field: value}))

    if not conditions:
        return []

    combined = conditions[0]
    for cond in conditions[1:]:
        combined |= cond

    return list(query.filter(combined))


def update_test_data(existing_test, new_cycles, metadata, strategy="append"):
    """
    Update an existing test with new data.

    Args:
        existing_test: Existing TestResult object
        new_cycles: New cycle data
        metadata: New metadata
        strategy: Update strategy ('append' or 'replace')

    Returns:
        TestResult: Updated TestResult object
    """
    if strategy == "replace":
        # Replace all cycles
        existing_test.cycles = []

    # Get existing cycle indices
    existing_indices = set(c.cycle_index for c in existing_test.cycles)

    # Add new cycles that don't exist yet
    for cycle in new_cycles:
        if strategy == "replace" or cycle["cycle_index"] not in existing_indices:
            # Create and add the cycle
            cycle_doc = models.CycleSummary(
                cycle_index=cycle["cycle_index"],
                charge_capacity=cycle["charge_capacity"],
                discharge_capacity=cycle["discharge_capacity"],
                coulombic_efficiency=cycle["coulombic_efficiency"],
            )

            # Add optional fields if present
            if "charge_energy" in cycle and cycle["charge_energy"] is not None:
                cycle_doc.charge_energy = cycle["charge_energy"]

            if "discharge_energy" in cycle and cycle["discharge_energy"] is not None:
                cycle_doc.discharge_energy = cycle["discharge_energy"]

            if "energy_efficiency" in cycle and cycle["energy_efficiency"] is not None:
                cycle_doc.energy_efficiency = cycle["energy_efficiency"]

            if (
                "internal_resistance" in cycle
                and cycle["internal_resistance"] is not None
            ):
                cycle_doc.internal_resistance = cycle["internal_resistance"]

            existing_test.cycles.append(cycle_doc)

    # Update metadata if provided
    if metadata:
        # Extract detailed_cycles if present, then remove from metadata
        detailed_cycles = None
        if "detailed_cycles" in metadata:
            detailed_cycles = metadata.pop("detailed_cycles")

        # Update any existing fields
        for key, value in metadata.items():
            if hasattr(existing_test, key):
                setattr(existing_test, key, value)

        # Update custom_data
        if (
            not hasattr(existing_test, "custom_data")
            or existing_test.custom_data is None
        ):
            existing_test.custom_data = {}

        # Add any metadata not already captured
        for key, value in metadata.items():
            if not hasattr(existing_test, key):
                existing_test.custom_data[key] = value

    # Recompute metrics
    metrics = compute_metrics(existing_test.cycles)
    for key, value in metrics.items():
        if hasattr(existing_test, key):
            setattr(existing_test, key, value)

    # Save changes
    existing_test.save()

    # Store detailed cycle data if available
    if detailed_cycles:
        try:
            from battery_analysis.utils.detailed_data_manager import (
                store_detailed_cycle_data,
            )

            store_detailed_cycle_data(str(existing_test.id), detailed_cycles)
            logging.info(
                f"Stored {len(detailed_cycles)} detailed cycles for test {existing_test.id}"
            )
        except Exception as e:
            logging.error(f"Error storing detailed cycle data: {e}")

    # Update the sample
    update_sample_properties(existing_test.sample.fetch())

    return existing_test


def _match_experiment_plans(sample, test_result):
    """Update experiment plan entries with new test information.

    Parameters
    ----------
    sample:
        The :class:`Sample` associated with the test.
    test_result:
        The newly created or updated :class:`TestResult` instance.
    """

    metadata = getattr(sample, "metadata", {}) or {}
    if not metadata:
        return

    try:  # pragma: no cover - requires database
        plans = list(models.ExperimentPlan.objects(sample_ids=sample.id))
    except Exception:
        # Fallback for offline dataclass implementation
        plans = []
        registry = getattr(models.ExperimentPlan, "_registry", {})
        for plan in registry.values():
            if sample in getattr(plan, "sample_ids", []):
                plans.append(plan)

    for plan in plans:
        factor_keys = getattr(plan, "factors", {}).keys()
        updated = False
        for entry in plan.matrix:
            if all(entry.get(k) == metadata.get(k) for k in factor_keys):
                entry.setdefault("tests", []).append(
                    {
                        "id": str(getattr(test_result, "id", "")),
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                    }
                )
                updated = True
        if updated:
            try:  # pragma: no cover - ignore failures without DB
                plan.save()
            except Exception:
                pass


def process_file_with_update(file_path, sample):
    """
    Process a file with automatic update detection.

    Args:
        file_path: Path to the file
        sample: Sample object

    Returns:
        tuple: (TestResult, was_update) - test result and whether it was an update
    """
    # Parse the file
    parsed_data, metadata = parsers.parse_file(file_path)
    logging.info(f"File processed as tester type: {metadata.get('tester', 'Unknown')}")

    # Compute file hash for matching when possible
    file_hash = None
    try:
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
    except Exception as exc:  # pragma: no cover - best effort
        logging.warning(f"Could not compute hash for {file_path}: {exc}")
    if metadata is None:
        metadata = {}
    metadata["file_hash"] = file_hash

    # Extract detailed_cycles if present in metadata
    detailed_cycles = None
    if metadata and "detailed_cycles" in metadata:
        detailed_cycles = metadata.pop("detailed_cycles")
        logging.info(f"Extracted {len(detailed_cycles)} detailed cycle datasets")

    # If parser did not return cycle summaries, build them from detailed data
    if (
        (not parsed_data or not isinstance(parsed_data, list))
        or (parsed_data and "discharge_capacity" not in parsed_data[0])
    ) and detailed_cycles:
        logging.info("Building cycle summaries from detailed data")
        parsed_data = summarize_detailed_cycles(detailed_cycles)

    # Add file path to metadata
    metadata["file_path"] = file_path

    # Extract identifiers and ensure metadata carries normalized versions
    identifiers = extract_test_identifiers(file_path, parsed_data, metadata)
    for key in ["base_file_name", "base_test_name", "test_id", "file_hash"]:
        if identifiers.get(key):
            metadata[key] = identifiers[key]

    # Look for matching tests
    matching_tests = find_matching_tests(identifiers, sample.id)

    if matching_tests:
        # Update the existing test
        existing_test = matching_tests[0]
        updated_test = update_test_data(
            existing_test, parsed_data, metadata, strategy="append"
        )

        # Store detailed cycle data if available
        if detailed_cycles:
            try:
                from battery_analysis.utils.detailed_data_manager import (
                    store_detailed_cycle_data,
                )

                store_detailed_cycle_data(str(updated_test.id), detailed_cycles)
                logging.info(
                    f"Stored {len(detailed_cycles)} detailed cycles for test {updated_test.id}"
                )
            except Exception as e:
                logging.error(f"Error storing detailed cycle data: {e}")

        _match_experiment_plans(sample, updated_test)
        return updated_test, True
    else:
        # Create a new test
        test_result = create_test_result(
            sample=sample,
            cycles_summary=parsed_data,
            tester=metadata.get("tester", "Unknown"),
            metadata=metadata,
        )

        # Store detailed cycle data if available
        if detailed_cycles:
            try:
                from battery_analysis.utils.detailed_data_manager import (
                    store_detailed_cycle_data,
                )

                store_detailed_cycle_data(str(test_result.id), detailed_cycles)
                logging.info(
                    f"Stored {len(detailed_cycles)} detailed cycles for test {test_result.id}"
                )
            except Exception as e:
                logging.error(f"Error storing detailed cycle data: {e}")

        _match_experiment_plans(sample, test_result)
        return test_result, False


def backfill_cycle_summaries(test_ids=None):
    """Populate missing ``CycleSummary`` documents for existing tests.

    Parameters
    ----------
    test_ids : iterable | None
        Optional list of specific TestResult IDs to process. If ``None``, all
        tests missing cycle summaries will be processed.

    Returns
    -------
    int
        Number of tests updated.
    """

    import pickle
    from battery_analysis.utils.detailed_data_manager import get_detailed_cycle_data

    if not ensure_connection():
        logging.error("Database connection not available")
        return 0

    if test_ids:
        tests = models.TestResult.objects(id__in=test_ids)
    else:
        tests = models.TestResult.objects(cycles__size=0)

    updated = 0
    for test in tests:
        detailed = get_detailed_cycle_data(str(test.id), include_incomplete=True)
        if not detailed:
            # If no cycle indices recorded on the test, gather directly
            detailed = {}
            for detail in models.CycleDetailData.objects(test_result=test.id):
                try:
                    charge = (
                        pickle.loads(detail.charge_data.read())
                        if detail.charge_data
                        else {}
                    )
                    discharge = (
                        pickle.loads(detail.discharge_data.read())
                        if detail.discharge_data
                        else {}
                    )
                    detailed[detail.cycle_index] = {
                        "charge": charge,
                        "discharge": discharge,
                    }
                except Exception:
                    continue
        if not detailed:
            continue
        summaries = summarize_detailed_cycles(detailed)
        if not summaries:
            continue
        test.cycles = [models.CycleSummary(**c) for c in summaries]
        metrics = compute_metrics(summaries)
        for key, value in metrics.items():
            if hasattr(test, key):
                setattr(test, key, value)
        test.save()
        updated += 1

    return updated
