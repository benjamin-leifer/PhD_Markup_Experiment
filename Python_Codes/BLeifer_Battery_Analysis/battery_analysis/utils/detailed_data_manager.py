"""
Manages detailed battery test data using GridFS.

This module handles the storage and retrieval of detailed cycle data,
which is too large to store efficiently in the main MongoDB documents.
"""

import logging
from battery_analysis.models import TestResult, CycleDetailData
import io
import pickle

def store_detailed_cycle_data(test_id, detailed_cycles):
    """
    Store detailed cycle data in GridFS after a TestResult has been created.

    Args:
        test_id: ID of the TestResult document
        detailed_cycles: Dict with {cycle_index: {charge_data: {...}, discharge_data: {...}}}

    Returns:
        bool: Success status
    """
    try:
        # Get the TestResult to make sure it exists
        test = TestResult.objects(id=test_id).first()
        if not test:
            logging.error(f"TestResult with ID {test_id} not found")
            return False

        success_count = 0
        for cycle_index, cycle_data in detailed_cycles.items():
            if 'charge_data' in cycle_data and 'discharge_data' in cycle_data:
                try:
                    # Check if entry already exists
                    existing = CycleDetailData.objects(test_result=test_id, cycle_index=cycle_index).first()
                    if existing:
                        existing.delete()

                    # Create binary data
                    charge_bytes = io.BytesIO()
                    pickle.dump(cycle_data['charge_data'], charge_bytes)
                    charge_bytes.seek(0)

                    discharge_bytes = io.BytesIO()
                    pickle.dump(cycle_data['discharge_data'], discharge_bytes)
                    discharge_bytes.seek(0)

                    # Create the document
                    detail_data = CycleDetailData(
                        test_result=test_id,
                        cycle_index=cycle_index
                    )

                    # Store the files
                    detail_data.charge_data.put(
                        charge_bytes,
                        content_type='application/python-pickle',
                        filename=f'charge_data_cycle_{cycle_index}.pkl'
                    )

                    detail_data.discharge_data.put(
                        discharge_bytes,
                        content_type='application/python-pickle',
                        filename=f'discharge_data_cycle_{cycle_index}.pkl'
                    )

                    # Save the document
                    detail_data.save()
                    success_count += 1

                except Exception as e:
                    logging.error(f"Error storing detailed data for cycle {cycle_index}: {str(e)}")

        logging.info(f"Stored detailed data for {success_count} cycles of test {test_id}")
        return True
    except Exception as e:
        logging.error(f"Error storing detailed cycle data: {str(e)}")
        return False

def get_detailed_cycle_data(test_id, cycle_index=None):
    """
    Retrieve detailed cycle data from GridFS.

    Args:
        test_id: ID of the TestResult document
        cycle_index: Specific cycle to retrieve (if None, retrieves all available)

    Returns:
        dict: Dictionary with detailed cycle data
    """
    logging.info(f"Attempting to retrieve GridFS data for test {test_id}, cycle {cycle_index}")
    try:
        if cycle_index is not None:
            # Get specific cycle
            detail_data = CycleDetailData.objects(test_result=test_id, cycle_index=cycle_index).first()

            if detail_data:
                logging.info(f"Found GridFS data for test {test_id}, cycle {cycle_index}")
                # Retrieve and unpickle the data
                try:
                    charge_bytes = detail_data.charge_data.read()
                    charge_data = pickle.loads(charge_bytes)

                    discharge_bytes = detail_data.discharge_data.read()
                    discharge_data = pickle.loads(discharge_bytes)

                    return {cycle_index: {
                        'charge': charge_data,
                        'discharge': discharge_data
                    }}
                except Exception as e:
                    logging.error(f"Error unpickling data: {e}")
                    return {}
            else:
                logging.warning(f"No GridFS data found for test {test_id}, cycle {cycle_index}")
                return {}
        else:
            # Get all cycles for this test
            test = TestResult.objects(id=test_id).first()
            if not test:
                logging.error(f"TestResult with ID {test_id} not found")
                return {}

            cycle_indices = [c.cycle_index for c in test.cycles]
            result = {}

            for idx in cycle_indices:
                try:
                    cycle_data = get_detailed_cycle_data(test_id, idx)
                    if cycle_data and idx in cycle_data:
                        result[idx] = cycle_data[idx]
                except Exception as e:
                    logging.debug(f"No detailed data found for cycle {idx}: {str(e)}")

            logging.info(f"Retrieved {len(result)} cycles of detailed data for test {test_id}")
            return result
    except Exception as e:
        logging.error(f"Error retrieving detailed cycle data: {str(e)}")
        return {}
