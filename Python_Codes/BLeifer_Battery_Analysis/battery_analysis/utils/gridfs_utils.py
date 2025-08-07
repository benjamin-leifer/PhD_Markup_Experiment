"""
Utilities for storing and retrieving large files from GridFS.

This module provides functions to store and retrieve large files using MongoDB's GridFS,
including raw data files and detailed cycle-by-cycle test data.
"""

import io
import pickle
import numpy as np
import pandas as pd
from battery_analysis.models import CycleDetailData, RawDataFile, TestResult
from .db import ensure_connection


def store_cycle_detail_data(test_id, cycle_index, charge_data, discharge_data):
    """
    Store detailed cycle data in GridFS.

    Args:
        test_id: ID of TestResult document
        cycle_index: Cycle index
        charge_data: Dict with charge data arrays
        discharge_data: Dict with discharge data arrays

    Returns:
        CycleDetailData: The stored data document
    """
    if not ensure_connection():
        return None

    # Check if entry already exists
    existing = CycleDetailData.objects(test_result=test_id, cycle_index=cycle_index).first()
    if existing:
        existing.delete()

    # Create binary data
    charge_bytes = io.BytesIO()
    pickle.dump(charge_data, charge_bytes)
    charge_bytes.seek(0)

    discharge_bytes = io.BytesIO()
    pickle.dump(discharge_data, discharge_bytes)
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
    return detail_data


def get_cycle_detail_data(test_id, cycle_index):
    """
    Retrieve detailed cycle data from GridFS.

    Args:
        test_id: ID of TestResult document
        cycle_index: Cycle index

    Returns:
        dict: Dictionary with charge and discharge data
    """
    if not ensure_connection():
        return None

    # Find the document
    detail_data = CycleDetailData.objects(test_result=test_id, cycle_index=cycle_index).first()
    if not detail_data:
        return None

    # Retrieve and unpickle the data
    charge_bytes = detail_data.charge_data.read()
    charge_data = pickle.loads(charge_bytes)

    discharge_bytes = detail_data.discharge_data.read()
    discharge_data = pickle.loads(discharge_bytes)

    return {
        'charge': charge_data,
        'discharge': discharge_data
    }


def store_raw_data_file(file_path, test_result=None, file_type=None):
    """
    Store a raw data file in GridFS.

    Args:
        file_path: Path to the file to store
        test_result: TestResult document or ID to associate with the file
        file_type: Type of file (e.g., 'arbin_excel')

    Returns:
        RawDataFile: The stored file document
    """
    if not ensure_connection():
        return None

    import os
    import datetime

    filename = os.path.basename(file_path)

    # Determine file type if not provided
    if not file_type:
        extension = os.path.splitext(filename)[1].lower()
        if extension in ['.xlsx', '.xls']:
            if any(pattern in filename for pattern in ['Channel', '_Wb_', 'Rate_Test']):
                file_type = 'arbin_excel'
            else:
                file_type = 'excel'
        elif extension in ['.mpt', '.mpr']:
            file_type = 'biologic'
        else:
            file_type = extension[1:]  # Remove the dot

    # Convert test_result ID to document if needed
    if test_result and isinstance(test_result, str):
        test_result = TestResult.objects(id=test_result).first()

    # Check if file already exists for this test
    if test_result:
        existing = RawDataFile.objects(test_result=test_result).first()
        if existing:
            # Update existing file
            with open(file_path, 'rb') as f:
                existing.file_data.replace(f, filename=filename)
            existing.filename = filename
            existing.file_type = file_type
            existing.upload_date = datetime.datetime.now()
            existing.save()
            return existing

    # Create new file document
    raw_file = RawDataFile(
        filename=filename,
        file_type=file_type,
        upload_date=datetime.datetime.now()
    )

    # Store the file
    with open(file_path, 'rb') as f:
        raw_file.file_data.put(f, filename=filename)

    # Link to test result if provided
    if test_result:
        raw_file.test_result = test_result

    # Save and return
    raw_file.save()
    return raw_file


def get_raw_data_file(test_id, as_file_path=False):
    """
    Retrieve a raw data file from GridFS.

    Args:
        test_id: ID of the TestResult to get file for
        as_file_path: If True, saves to temp file and returns path

    Returns:
        bytes or str: The file data or path to temp file
    """
    if not ensure_connection():
        raise ValueError("Database connection not available")

    # Find the file
    raw_file = RawDataFile.objects(test_result=test_id).first()
    if not raw_file:
        raise ValueError(f"No raw data file found for test {test_id}")

    if as_file_path:
        # Save to a temporary file
        import tempfile
        import os

        ext = os.path.splitext(raw_file.filename)[1]
        temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        temp_file.write(raw_file.file_data.read())
        temp_file.close()
        return temp_file.name
    else:
        # Return the raw bytes
        return raw_file.file_data.read()
