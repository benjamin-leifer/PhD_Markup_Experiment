import os
import datetime
from battery_analysis.models import RawDataFile, TestResult


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
    # Find the file
    raw_file = RawDataFile.objects(test_result=test_id).first()
    if not raw_file:
        raise ValueError(f"No raw data file found for test {test_id}")

    if as_file_path:
        # Save to a temporary file
        import tempfile
        ext = os.path.splitext(raw_file.filename)[1]
        temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        temp_file.write(raw_file.file_data.read())
        temp_file.close()
        return temp_file.name
    else:
        # Return the raw bytes
        return raw_file.file_data.read()


def get_raw_data_file_by_id(file_id, as_file_path=False):
    """Retrieve a raw data file directly by its ``RawDataFile`` ID.

    Parameters
    ----------
    file_id:
        The ``id`` of the :class:`~battery_analysis.models.RawDataFile` to
        retrieve.
    as_file_path:
        When ``True``, the file is written to a temporary location and the
        path is returned. When ``False`` (default) the raw bytes are returned.

    Returns
    -------
    bytes | str
        The file's bytes or the temporary file path, depending on
        ``as_file_path``.
    """

    raw_file = RawDataFile.objects(id=file_id).first()
    if not raw_file:
        raise ValueError(f"No raw data file found with id {file_id}")

    raw_file.file_data.seek(0)
    if as_file_path:
        import tempfile

        ext = os.path.splitext(raw_file.filename)[1]
        temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        temp_file.write(raw_file.file_data.read())
        temp_file.close()
        return temp_file.name

    return raw_file.file_data.read()
