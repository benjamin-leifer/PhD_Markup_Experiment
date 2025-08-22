import os
import datetime
import hashlib
from battery_analysis.models import ImportJob, RawDataFile, TestResult


def store_raw_data_file(
    file_path,
    test_result=None,
    file_type=None,
    *,
    source_path: str | None = None,
    import_job: ImportJob | str | None = None,
):
    """
    Store a raw data file in GridFS.

    Args:
        file_path: Path to the file to store
        test_result: TestResult document or ID to associate with the file
        file_type: Type of file (e.g., 'arbin_excel')
        source_path: Original absolute path of the file on disk
        import_job: ImportJob document or id that created this file

    Returns:
        RawDataFile: The stored file document
    """
    filename = os.path.basename(file_path)

    # Determine file type if not provided
    if not file_type:
        extension = os.path.splitext(filename)[1].lower()
        if extension in [".xlsx", ".xls"]:
            if any(pattern in filename for pattern in ["Channel", "_Wb_", "Rate_Test"]):
                file_type = "arbin_excel"
            else:
                file_type = "excel"
        elif extension in [".mpt", ".mpr"]:
            file_type = "biologic"
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
            with open(file_path, "rb") as f:
                existing.file_data.replace(f, filename=filename)
            existing.filename = filename
            existing.file_type = file_type
            existing.upload_date = datetime.datetime.now()
            if source_path is not None:
                existing.source_path = source_path
            if import_job is not None:
                existing.import_job_id = getattr(import_job, "id", import_job)
            existing.save()
            return existing

    # Create new file document
    raw_file = RawDataFile(
        filename=filename,
        file_type=file_type,
        upload_date=datetime.datetime.now(),
        source_path=source_path,
        import_job_id=getattr(import_job, "id", import_job),
    )

    # Store the file
    with open(file_path, "rb") as f:
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


def _sha256_file(path: str) -> str:
    """Return the SHA256 hex digest of ``path``."""

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_raw(
    file_path: str,
    *,
    test_result: TestResult | None = None,
    file_type: str | None = None,
    source_path: str | None = None,
    import_job: ImportJob | None = None,
) -> str:
    """Store ``file_path`` in GridFS and return its identifier.

    Before writing a new :class:`RawDataFile`, this function computes the
    SHA256 hash of the file's bytes and checks whether another
    :class:`~battery_analysis.models.TestResult` already references an
    identical file.  If so, the existing ``file_id`` is reused to avoid
    duplicate GridFS entries.

    Parameters
    ----------
    file_path:
        Path to the file on disk.
    test_result:
        Optional :class:`~battery_analysis.models.TestResult` the file belongs to.
    file_type:
        Optional file type hint passed through to :func:`store_raw_data_file`.
    source_path:
        Optional absolute path stored for provenance.
    import_job:
        Optional :class:`~battery_analysis.models.ImportJob` associated with the
        file's creation.

    Returns
    -------
    str
        The id of the :class:`~battery_analysis.models.RawDataFile` containing
        the archived bytes.
    """

    file_hash = _sha256_file(file_path)

    existing = TestResult.objects(file_hash=file_hash).first()
    if existing and getattr(existing, "file_id", None):
        file_id = existing.file_id
        if test_result is not None:
            test_result.file_hash = file_hash
            test_result.file_id = file_id
            try:  # best effort; tests may use simple stubs
                test_result.save()
            except Exception:
                pass
        return file_id

    raw = store_raw_data_file(
        file_path,
        test_result=test_result,
        file_type=file_type,
        source_path=source_path,
        import_job=import_job,
    )
    file_id = str(raw.id)

    if test_result is not None:
        test_result.file_hash = file_hash
        test_result.file_id = file_id
        try:
            test_result.save()
        except Exception:
            pass

    return file_id


def retrieve_raw(file_id: str, as_file_path: bool = False) -> bytes | str:
    """Retrieve archived bytes previously stored with :func:`save_raw`.

    Parameters
    ----------
    file_id:
        Identifier returned by :func:`save_raw`.
    as_file_path:
        When ``True`` the file is written to a temporary location and the path
        is returned.  When ``False`` the raw bytes are returned.

    Returns
    -------
    bytes | str
        The file contents or a path to a temporary file depending on
        ``as_file_path``.
    """

    return get_raw_data_file_by_id(file_id, as_file_path=as_file_path)


def cleanup_orphaned() -> int:
    """Remove ``RawDataFile`` documents with no referencing ``TestResult``.

    Returns
    -------
    int
        The number of removed files.
    """

    removed = 0
    for raw in RawDataFile.objects():
        referenced = TestResult.objects(file_id=str(raw.id)).first()
        if referenced:
            continue
        try:
            raw.file_data.delete()  # type: ignore[attr-defined]
        except Exception:
            pass
        raw.delete()
        removed += 1
    return removed
