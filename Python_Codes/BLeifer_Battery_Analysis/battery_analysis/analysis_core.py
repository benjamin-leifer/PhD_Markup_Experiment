"""
Analysis module for battery test data.

This module provides functions to compute metrics from cycle data, create and update
test results, and propagate properties across related samples.
"""

import numpy as np
import logging
import datetime

try:
    from . import models, utils
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    models = importlib.import_module("models")
    utils = importlib.import_module("utils")
import pandas as pd


def create_test_result(sample, cycles_summary, tester, metadata=None):
    """
    Create a TestResult document from parsed cycle data and attach it to a Sample.

    Args:
        sample: Sample object to which the test result will be attached
        cycles_summary: List of cycle summary dictionaries from the parser
        tester: String indicating the tester type (e.g., 'Arbin', 'BioLogic')
        metadata: Optional metadata from the file

    Returns:
        TestResult: The created and saved TestResult object
    """
    # Set default metadata
    if metadata is None:
        metadata = {}

    # Compute overall metrics from cycle summary
    metrics = compute_metrics(cycles_summary)

    # Create CycleSummary embedded documents list from cycles_summary dicts
    cycle_docs = []
    for cycle in cycles_summary:
        cycle_doc = models.CycleSummary(
            cycle_index=cycle.get("cycle_index"),
            charge_capacity=cycle.get("charge_capacity", 0.0),
            discharge_capacity=cycle.get("discharge_capacity", 0.0),
            coulombic_efficiency=cycle.get("coulombic_efficiency", 0.0),
        )

        # Add optional fields if present
        if "charge_energy" in cycle and cycle["charge_energy"] is not None:
            cycle_doc.charge_energy = cycle["charge_energy"]

        if "discharge_energy" in cycle and cycle["discharge_energy"] is not None:
            cycle_doc.discharge_energy = cycle["discharge_energy"]

        if "energy_efficiency" in cycle and cycle["energy_efficiency"] is not None:
            cycle_doc.energy_efficiency = cycle["energy_efficiency"]

        if "internal_resistance" in cycle and cycle["internal_resistance"] is not None:
            cycle_doc.internal_resistance = cycle["internal_resistance"]

        cycle_docs.append(cycle_doc)

    # Get name from metadata or generate a default one
    name = metadata.get(
        "name", metadata.get("file_name", f"{tester} test for {sample.name}")
    )

    # Instantiate a TestResult document
    from battery_analysis import user_tracking

    current_user = user_tracking.get_current_user()

    test_result = models.TestResult(
        sample=sample,
        tester=tester,
        name=name,
        cycles=cycle_docs,
        cycle_count=metrics.get("cycle_count"),
        initial_capacity=metrics.get("initial_capacity"),
        final_capacity=metrics.get("final_capacity"),
        capacity_retention=metrics.get("capacity_retention"),
        avg_coulombic_eff=metrics.get("avg_coulombic_eff"),
        created_by=current_user,
        last_modified_by=current_user,
    )
    if current_user:
        test_result.notes_log.append(
            {"timestamp": datetime.datetime.utcnow(), "note": "created"}
        )

    # Infer protocol-related information
    try:
        from battery_analysis.analysis.protocol_detection import (
            detect_and_update_test_protocol,
        )

        detect_and_update_test_protocol(test_result, cycles_summary, prompt=False)
    except Exception:
        pass

    # Add optional metrics if available
    if "avg_energy_efficiency" in metrics:
        test_result.avg_energy_efficiency = metrics["avg_energy_efficiency"]

    # Add metadata fields if available
    for field in [
        "temperature",
        "upper_cutoff_voltage",
        "lower_cutoff_voltage",
        "charge_rate",
        "discharge_rate",
        "file_path",
    ]:
        if field in metadata:
            setattr(test_result, field, metadata[field])

    # Add test_type if available in metadata
    if (
        "test_type" in metadata
        and metadata["test_type"] in models.TestResult.test_type.choices
    ):
        test_result.test_type = metadata["test_type"]

    # Save the TestResult to the database
    try:
        if hasattr(test_result, "full_clean"):
            test_result.full_clean()
        test_result.save()

        # Link this test to the sample
        sample.tests.append(test_result)
        sample.save()

        # Update inferred properties on the sample
        update_sample_properties(sample)

        return test_result

    except Exception as e:
        logging.error(f"Error saving test result: {e}")
        raise


def update_sample_properties(sample, save=True):
    """
    Compute and update the Sample's inferred properties based on its tests.

    If the Sample has a parent, propagate the update to the parent as well.

    Args:
        sample: Sample object to update
        save: Whether to save the updated Sample to the database

    Returns:
        Sample: The updated Sample object
    """
    tests = sample.tests  # List of TestResult references

    if not tests or len(tests) == 0:
        return sample

    # Ensure we have actual TestResult objects (dereference if needed)
    if hasattr(models.TestResult, "objects"):
        tests = [
            (
                t
                if isinstance(t, models.TestResult)
                else models.TestResult.objects(id=t.id).first()
            )
            for t in tests
        ]
    else:  # dataclass fallback for tests
        tests = [t for t in tests]
    tests = [t for t in tests if t is not None]  # Remove any None values

    if not tests:
        return sample

    # Compute aggregated properties for this sample
    initial_caps = [t.initial_capacity for t in tests if t.initial_capacity is not None]
    final_caps = [t.final_capacity for t in tests if t.final_capacity is not None]
    retentions = [
        t.capacity_retention for t in tests if t.capacity_retention is not None
    ]
    effs = [t.avg_coulombic_eff for t in tests if t.avg_coulombic_eff is not None]

    # Energy efficiency if available
    energy_effs = [
        t.avg_energy_efficiency
        for t in tests
        if hasattr(t, "avg_energy_efficiency") and t.avg_energy_efficiency is not None
    ]

    # Internal resistance if available
    internal_resistances = []
    for t in tests:
        if (
            hasattr(t, "median_internal_resistance")
            and t.median_internal_resistance is not None
        ):
            internal_resistances.append(t.median_internal_resistance)

    # Update sample properties with aggregated values
    if initial_caps:
        sample.avg_initial_capacity = float(np.mean(initial_caps))

    if final_caps:
        sample.avg_final_capacity = float(np.mean(final_caps))

    if retentions:
        sample.avg_capacity_retention = float(np.mean(retentions))

    if effs:
        sample.avg_coulombic_eff = float(np.mean(effs))

    if energy_effs:
        sample.avg_energy_efficiency = float(np.mean(energy_effs))

    if internal_resistances:
        sample.median_internal_resistance = float(np.median(internal_resistances))

    # Save the updated sample
    if save:
        sample.save()

    # Propagate to parent sample, if any
    if sample.parent:
        parent = sample.parent

        # We'll update parent properties based on all its children
        children = models.Sample.objects(parent=parent)

        if children:
            # Aggregate across all child samples
            child_initial_caps = [
                child.avg_initial_capacity
                for child in children
                if child.avg_initial_capacity is not None
            ]

            child_final_caps = [
                child.avg_final_capacity
                for child in children
                if child.avg_final_capacity is not None
            ]

            child_retentions = [
                child.avg_capacity_retention
                for child in children
                if child.avg_capacity_retention is not None
            ]

            child_effs = [
                child.avg_coulombic_eff
                for child in children
                if child.avg_coulombic_eff is not None
            ]

            # Update parent properties
            if child_initial_caps:
                parent.avg_initial_capacity = float(np.mean(child_initial_caps))

            if child_final_caps:
                parent.avg_final_capacity = float(np.mean(child_final_caps))

            if child_retentions:
                parent.avg_capacity_retention = float(np.mean(child_retentions))

            if child_effs:
                parent.avg_coulombic_eff = float(np.mean(child_effs))

            # Energy efficiency
            child_energy_effs = [
                child.avg_energy_efficiency
                for child in children
                if hasattr(child, "avg_energy_efficiency")
                and child.avg_energy_efficiency is not None
            ]

            if child_energy_effs:
                parent.avg_energy_efficiency = float(np.mean(child_energy_effs))

            # Internal resistance
            child_internal_resistances = [
                child.median_internal_resistance
                for child in children
                if hasattr(child, "median_internal_resistance")
                and child.median_internal_resistance is not None
            ]

            if child_internal_resistances:
                parent.median_internal_resistance = float(
                    np.median(child_internal_resistances)
                )

            if save:
                parent.save()

            # If there's a higher-level parent, recursively propagate upward
            if parent.parent:
                update_sample_properties(parent, save=save)

    return sample


def compare_samples(sample_ids, metric="capacity_retention"):
    """
    Compare multiple samples based on a selected metric.

    Args:
        sample_ids: List of sample IDs to compare
        metric: Metric to use for comparison (default: 'capacity_retention')

    Returns:
        dict: Dictionary with sample IDs as keys and metric values as values,
              sorted by metric value in descending order
    """
    valid_metrics = [
        "avg_initial_capacity",
        "avg_final_capacity",
        "avg_capacity_retention",
        "avg_coulombic_eff",
        "avg_energy_efficiency",
        "median_internal_resistance",
    ]

    if metric not in valid_metrics:
        raise ValueError(
            f"Invalid metric: {metric}. Valid options are: {valid_metrics}"
        )

    # Get samples
    samples = [models.Sample.objects(id=sid).first() for sid in sample_ids]
    samples = [s for s in samples if s is not None]

    # Create comparison dictionary
    comparison = {}
    for sample in samples:
        value = getattr(sample, metric, None)
        if value is not None:
            comparison[str(sample.id)] = {"name": sample.name, "value": float(value)}

    # Sort by metric value (descending, except for internal resistance which is ascending)
    if metric == "median_internal_resistance":
        # Lower resistance is better
        sorted_comparison = dict(
            sorted(comparison.items(), key=lambda x: x[1]["value"])
        )
    else:
        # Higher values are better
        sorted_comparison = dict(
            sorted(comparison.items(), key=lambda x: x[1]["value"], reverse=True)
        )

    return sorted_comparison


def get_cycle_data(test_id, include_raw=False):
    """
    Get cycle data for a specific test, optionally including raw data if available.

    Args:
        test_id: ID of the TestResult to analyze
        include_raw: Whether to include raw data points (default: False)

    Returns:
        dict: Dictionary containing cycle data and metrics
    """
    test = models.TestResult.objects(id=test_id).first()

    if test is None:
        raise ValueError(f"Test with ID {test_id} not found")

    # Basic test info
    result = {
        "test_id": str(test.id),
        "sample_name": utils.get_sample_name(test.sample),
        "test_name": test.name,
        "tester": test.tester,
        "metrics": {
            "cycle_count": test.cycle_count,
            "initial_capacity": test.initial_capacity,
            "final_capacity": test.final_capacity,
            "capacity_retention": test.capacity_retention,
            "avg_coulombic_eff": test.avg_coulombic_eff,
        },
        "cycles": [],
    }

    # Add optional metrics if available
    if (
        hasattr(test, "avg_energy_efficiency")
        and test.avg_energy_efficiency is not None
    ):
        result["metrics"]["avg_energy_efficiency"] = test.avg_energy_efficiency

    # Add cycle data
    for cycle in test.cycles:
        cycle_data = {
            "cycle_index": cycle.cycle_index,
            "charge_capacity": cycle.charge_capacity,
            "discharge_capacity": cycle.discharge_capacity,
            "coulombic_efficiency": cycle.coulombic_efficiency,
        }

        # Add optional fields if available
        if hasattr(cycle, "charge_energy") and cycle.charge_energy is not None:
            cycle_data["charge_energy"] = cycle.charge_energy

        if hasattr(cycle, "discharge_energy") and cycle.discharge_energy is not None:
            cycle_data["discharge_energy"] = cycle.discharge_energy

        if hasattr(cycle, "energy_efficiency") and cycle.energy_efficiency is not None:
            cycle_data["energy_efficiency"] = cycle.energy_efficiency

        if (
            hasattr(cycle, "internal_resistance")
            and cycle.internal_resistance is not None
        ):
            cycle_data["internal_resistance"] = cycle.internal_resistance

        # If requested, gather raw data either from the document or GridFS
        if include_raw:
            raw = {}

            if hasattr(cycle, "voltage_charge") and getattr(cycle, "voltage_charge"):
                raw["charge"] = {
                    "voltage": list(getattr(cycle, "voltage_charge", [])),
                    "current": list(getattr(cycle, "current_charge", [])),
                    "capacity": list(getattr(cycle, "capacity_charge", [])),
                    "time": list(getattr(cycle, "time_charge", [])),
                }

            if hasattr(cycle, "voltage_discharge") and getattr(
                cycle, "voltage_discharge"
            ):
                raw["discharge"] = {
                    "voltage": list(getattr(cycle, "voltage_discharge", [])),
                    "current": list(getattr(cycle, "current_discharge", [])),
                    "capacity": list(getattr(cycle, "capacity_discharge", [])),
                    "time": list(getattr(cycle, "time_discharge", [])),
                }

            if not raw:
                try:
                    raw = test.get_cycle_detail(cycle.cycle_index)
                except Exception as exc:  # pragma: no cover - optional
                    logging.debug(
                        f"Error loading detailed data for cycle {cycle.cycle_index}: {exc}"
                    )

            if raw:
                cycle_data["raw"] = raw

        result["cycles"].append(cycle_data)

    # No further raw handling

    return result


def compute_metrics(cycles_summary):
    """
    Compute overall metrics from cycle summary data.

    Args:
        cycles_summary: List of cycle summary dictionaries

    Returns:
        dict: Dictionary with computed metrics
    """
    if not cycles_summary:
        return {}

    # Convert to DataFrame for easier calculations
    df = pd.DataFrame(cycles_summary)

    # Safely handle missing columns
    discharge = (
        df["discharge_capacity"]
        if "discharge_capacity" in df.columns
        else pd.Series(dtype=float)
    )
    charge_eff = (
        df["coulombic_efficiency"]
        if "coulombic_efficiency" in df.columns
        else pd.Series(dtype=float)
    )

    # Use the first cycle's discharge capacity as the baseline for
    # initial capacity and capacity retention calculations
    initial_cap = discharge.iloc[0] if not discharge.empty else None
    final_cap = discharge.iloc[-1] if not discharge.empty else None
    if initial_cap and final_cap is not None and initial_cap > 0:
        retention = final_cap / initial_cap
    else:
        retention = None

    metrics = {
        "cycle_count": len(df),
        "initial_capacity": initial_cap,
        "final_capacity": final_cap,
        "capacity_retention": retention,
        "avg_coulombic_eff": float(charge_eff.mean()) if not charge_eff.empty else None,
        "avg_energy_efficiency": (
            float(df["energy_efficiency"].mean())
            if "energy_efficiency" in df.columns and not df["energy_efficiency"].empty
            else None
        ),
    }

    return metrics
