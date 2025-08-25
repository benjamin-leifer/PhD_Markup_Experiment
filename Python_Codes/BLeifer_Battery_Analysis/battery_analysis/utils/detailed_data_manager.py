# flake8: noqa
"""
Manages detailed battery test data using GridFS.

This module handles the storage and retrieval of detailed cycle data,
which is too large to store efficiently in the main MongoDB documents.
"""

import io
import logging
import pickle
from typing import Any, Dict, Optional

import gridfs
from battery_analysis.models import CycleDetailData, TestResult
from bson import ObjectId

from Mongodb_implementation import get_client

from .db import ensure_connection


def store_detailed_cycle_data(
    test_id: str, detailed_cycles: Dict[int, Dict[str, Any]]
) -> bool:
    """
    Store detailed cycle data in GridFS after a TestResult has been created.

    Args:
        test_id: ID of the TestResult document
        detailed_cycles: Dict with {cycle_index: {charge_data: {...}, discharge_data: {...}}}

    Returns:
        bool: Success status
    """
    if not ensure_connection():
        logging.error("Database connection not available")
        return False
    try:
        # Get the TestResult to make sure it exists
        test = TestResult.objects(id=test_id).first()
        if not test:
            logging.error(f"TestResult with ID {test_id} not found")
            return False

        success_count = 0
        for cycle_index, cycle_data in detailed_cycles.items():
            if "charge_data" in cycle_data and "discharge_data" in cycle_data:
                try:
                    # Check if entry already exists
                    existing = CycleDetailData.objects(
                        test_result=test_id, cycle_index=cycle_index
                    ).first()
                    if existing:
                        existing.delete()

                    # Create binary data
                    charge_bytes = io.BytesIO()
                    pickle.dump(cycle_data["charge_data"], charge_bytes)
                    charge_bytes.seek(0)

                    discharge_bytes = io.BytesIO()
                    pickle.dump(cycle_data["discharge_data"], discharge_bytes)
                    discharge_bytes.seek(0)

                    # Create the document
                    detail_data = CycleDetailData(
                        test_result=test_id, cycle_index=cycle_index
                    )

                    # Store the files
                    detail_data.charge_data.put(
                        charge_bytes,
                        content_type="application/python-pickle",
                        filename=f"charge_data_cycle_{cycle_index}.pkl",
                    )

                    detail_data.discharge_data.put(
                        discharge_bytes,
                        content_type="application/python-pickle",
                        filename=f"discharge_data_cycle_{cycle_index}.pkl",
                    )

                    # Save the document
                    detail_data.save()

                    # Mark the cycle as having detailed data and remove large arrays if present
                    for cyc in test.cycles:
                        if cyc.cycle_index == cycle_index:
                            cyc.has_detailed_data = True
                            for attr in [
                                "voltage_charge",
                                "current_charge",
                                "capacity_charge",
                                "time_charge",
                                "voltage_discharge",
                                "current_discharge",
                                "capacity_discharge",
                                "time_discharge",
                            ]:
                                if hasattr(cyc, attr):
                                    delattr(cyc, attr)
                            break

                    success_count += 1

                except Exception as e:
                    logging.error(
                        f"Error storing detailed data for cycle {cycle_index}: {str(e)}"
                    )

        if success_count:
            # Persist flag updates to the TestResult
            test.save()
        logging.info(
            f"Stored detailed data for {success_count} cycles of test {test_id}"
        )
        return True
    except Exception as e:
        logging.error(f"Error storing detailed cycle data: {str(e)}")
        return False


def get_detailed_cycle_data(
    test_id: str,
    cycle_index: Optional[int] = None,
    include_incomplete: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """Retrieve detailed cycle data from GridFS.

    If ``test_id`` is not a valid :class:`bson.ObjectId`, the function logs a
    message and returns an empty dict without querying the database.

    Parameters
    ----------
    test_id: str
        ID of the ``TestResult`` document.
    cycle_index: int | None
        Specific cycle to retrieve. If ``None``, all cycles are returned.
    include_incomplete: bool, optional
        When ``False`` (default) cycles lacking charge or discharge capacity are
        omitted. Set to ``True`` to include all cycles.

    Returns
    -------
    dict
        Mapping of cycle index to detailed cycle data dictionaries.
    """
    logging.info(
        f"Attempting to retrieve GridFS data for test {test_id}, cycle {cycle_index}"
    )
    if not ObjectId.is_valid(test_id):
        logging.warning(f"Invalid test id {test_id}")
        return {}

    use_mongoengine = hasattr(TestResult, "objects") and hasattr(
        CycleDetailData, "objects"
    )

    if use_mongoengine:
        if not ensure_connection():
            logging.error("Database connection not available")
            return {}
    else:
        logging.debug("MongoEngine models not available; using raw MongoDB client")
        client = get_client()
        db = client["battery_test_db"]
        fs = gridfs.GridFS(db)
        tr_coll = db["test_results"]
        cd_coll = db["cycle_detail_data"]

    try:
        if cycle_index is not None:
            if use_mongoengine:
                if not include_incomplete:
                    test = TestResult.objects(id=test_id).only("cycles").first()
                    if test:
                        for cyc in test.cycles:
                            if cyc.cycle_index == cycle_index:
                                if (
                                    getattr(cyc, "charge_capacity", 0) <= 0
                                    or getattr(cyc, "discharge_capacity", 0) <= 0
                                ):
                                    logging.info(
                                        "Skipping incomplete cycle %s for test %s",
                                        cycle_index,
                                        test_id,
                                    )
                                    return {}
                                break

                detail_data = CycleDetailData.objects(
                    test_result=test_id, cycle_index=cycle_index
                ).first()

                if detail_data:
                    logging.info(
                        f"Found GridFS data for test {test_id}, cycle {cycle_index}"
                    )
                    try:
                        charge_bytes = detail_data.charge_data.read()
                        charge_data = pickle.loads(charge_bytes)
                        discharge_bytes = detail_data.discharge_data.read()
                        discharge_data = pickle.loads(discharge_bytes)
                        return {
                            cycle_index: {
                                "charge": charge_data,
                                "discharge": discharge_data,
                            }
                        }
                    except Exception as e:
                        logging.error(f"Error unpickling data: {e}")
                        return {}
                logging.warning(
                    f"No GridFS data found for test {test_id}, cycle {cycle_index}"
                )
                return {}

            # -- raw MongoDB path -------------------------------------------------
            test_oid = ObjectId(test_id)
            if not include_incomplete:
                logging.debug("Querying test_results for %s", test_id)
                test_doc = tr_coll.find_one({"_id": test_oid}, {"cycles": 1})
                if test_doc:
                    for cyc in test_doc.get("cycles", []):
                        if cyc.get("cycle_index") == cycle_index:
                            if (
                                cyc.get("charge_capacity", 0) <= 0
                                or cyc.get("discharge_capacity", 0) <= 0
                            ):
                                logging.info(
                                    "Skipping incomplete cycle %s for test %s",
                                    cycle_index,
                                    test_id,
                                )
                                return {}
                            break

            logging.debug(
                "Querying cycle_detail_data for test %s, cycle %s",
                test_id,
                cycle_index,
            )
            detail_doc = cd_coll.find_one(
                {"test_result": test_oid, "cycle_index": cycle_index}
            )
            if detail_doc:
                logging.info(
                    "Found detail document for test %s, cycle %s",
                    test_id,
                    cycle_index,
                )
                try:
                    charge_bytes = fs.get(detail_doc["charge_data"]).read()
                    charge_data = pickle.loads(charge_bytes)
                    discharge_bytes = fs.get(detail_doc["discharge_data"]).read()
                    discharge_data = pickle.loads(discharge_bytes)
                    return {
                        cycle_index: {
                            "charge": charge_data,
                            "discharge": discharge_data,
                        }
                    }
                except Exception as e:
                    logging.error(f"Error unpickling data: {e}")
                    return {}
            logging.warning(
                "No GridFS data found for test %s, cycle %s", test_id, cycle_index
            )
            return {}

        # ------------------------------------------------------------------
        if use_mongoengine:
            test = TestResult.objects(id=test_id).first()
            if not test:
                logging.error(f"TestResult with ID {test_id} not found")
                return {}

            cycle_indices = [
                c.cycle_index
                for c in test.cycles
                if include_incomplete
                or (
                    getattr(c, "charge_capacity", 0) > 0
                    and getattr(c, "discharge_capacity", 0) > 0
                )
            ]

            if not cycle_indices:
                cycle_indices = [
                    c.cycle_index
                    for c in CycleDetailData.objects(test_result=test_id).only(
                        "cycle_index"
                    )
                ]

        else:
            test_oid = ObjectId(test_id)
            logging.debug("Fetching test document for %s", test_id)
            test_doc = tr_coll.find_one({"_id": test_oid}, {"cycles": 1})
            if not test_doc:
                logging.error(f"TestResult with ID {test_id} not found")
                return {}

            cycle_indices = [
                c.get("cycle_index")
                for c in test_doc.get("cycles", [])
                if include_incomplete
                or (
                    c.get("charge_capacity", 0) > 0
                    and c.get("discharge_capacity", 0) > 0
                )
            ]

            if not cycle_indices:
                logging.debug(
                    "No cycle summaries found; querying cycle_detail_data collection"
                )
                cycle_indices = [
                    d["cycle_index"]
                    for d in cd_coll.find({"test_result": test_oid}, {"cycle_index": 1})
                ]

        result = {}
        for idx in cycle_indices:
            try:
                cycle_data = get_detailed_cycle_data(
                    test_id, idx, include_incomplete=include_incomplete
                )
                if cycle_data and idx in cycle_data:
                    result[idx] = cycle_data[idx]
            except Exception as e:
                logging.debug("No detailed data found for cycle %s: %s", idx, e)

        logging.info(
            f"Retrieved {len(result)} cycles of detailed data for test {test_id}"
        )
        return result
    except Exception as e:
        logging.error(f"Error retrieving detailed cycle data: {str(e)}")
        return {}


def migrate_cycle_summary_data(test_id: str) -> int:
    """Move inline detailed arrays from ``CycleSummary`` documents to GridFS.

    This helper scans the cycles of a :class:`~battery_analysis.models.TestResult`
    for legacy array fields (e.g. ``voltage_charge``) and stores them using
    :class:`CycleDetailData` in GridFS. After storage the large arrays are
    removed from the embedded documents and the ``has_detailed_data`` flag is
    set.

    Args:
        test_id: ID of the :class:`TestResult` to migrate.

    Returns:
        int: Number of cycles migrated.
    """

    if not ensure_connection():
        logging.error("Database connection not available")
        return 0

    try:
        test = TestResult.objects(id=test_id).first()
        if not test:
            logging.error(f"TestResult with ID {test_id} not found")
            return 0

        detailed_cycles = {}
        for cycle in test.cycles:
            has_charge = hasattr(cycle, "voltage_charge") and getattr(
                cycle, "voltage_charge"
            )
            has_discharge = hasattr(cycle, "voltage_discharge") and getattr(
                cycle, "voltage_discharge"
            )

            if has_charge or has_discharge:
                charge_data = {
                    "voltage": getattr(cycle, "voltage_charge", []),
                    "current": getattr(cycle, "current_charge", []),
                    "capacity": getattr(cycle, "capacity_charge", []),
                    "time": getattr(cycle, "time_charge", []),
                }
                discharge_data = {
                    "voltage": getattr(cycle, "voltage_discharge", []),
                    "current": getattr(cycle, "current_discharge", []),
                    "capacity": getattr(cycle, "capacity_discharge", []),
                    "time": getattr(cycle, "time_discharge", []),
                }
                detailed_cycles[cycle.cycle_index] = {
                    "charge_data": charge_data,
                    "discharge_data": discharge_data,
                }

        if not detailed_cycles:
            return 0

        store_detailed_cycle_data(test_id, detailed_cycles)
        return len(detailed_cycles)
    except Exception as exc:  # pragma: no cover - migration is best effort
        logging.error(f"Error migrating cycle data: {exc}")
        return 0
