"""
Parsers for battery test data files from different equipment.

This package provides parsers for:
- Arbin battery testers (.xlsx, .csv)
- BioLogic potentiostats (.mpr, .mpt)
"""

import os
from . import arbin_parser
from . import biologic_parser


def parse_file(file_path):
    """
    Parse a battery test data file using the appropriate parser based on file extension.

    Args:
        file_path (str): Path to the data file

    Returns:
        tuple: (cycles_summary, metadata)
            - cycles_summary: List of dictionaries containing cycle data
            - metadata: Dictionary of test metadata

    Raises:
        ValueError: If the file format is not supported
    """
    file_path_lower = file_path.lower()

    # Determine parser based on file extension
    if file_path_lower.endswith(('.xlsx', '.xls', '.csv')):
        # Arbin files
        cycles_summary = arbin_parser.parse_arbin(file_path)
        metadata = arbin_parser.extract_test_metadata(file_path)
        return cycles_summary, metadata

    elif file_path_lower.endswith(('.mpr', '.mpt')):
        # BioLogic files
        cycles_summary = biologic_parser.parse_biologic(file_path)
        metadata = biologic_parser.extract_test_metadata(file_path)
        return cycles_summary, metadata

    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def get_supported_formats():
    """
    Get a list of supported file formats for battery test data.

    Returns:
        list: List of supported file extensions
    """
    return ['.xlsx', '.xls', '.csv', '.mpr', '.mpt']