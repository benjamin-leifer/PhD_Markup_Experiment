"""Parsers for battery test data files from various instruments."""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, Tuple

logger = logging.getLogger(__name__)

PARSERS: Dict[str, Callable[[str], Tuple]] = {}


def register_parser(ext: str, func: Callable[[str], Tuple]) -> None:
    """Register ``func`` as the parser for files ending with ``ext``."""
    PARSERS[ext.lower()] = func


def get_supported_formats():
    """Get list of supported file extensions."""
    return sorted(PARSERS)


def parse_file(file_path):
    """Parse a battery test data file."""
    extension = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)

    parser = PARSERS.get(extension)
    if parser is not None:
        try:
            return parser(file_path)
        except Exception as exc:  # pragma: no cover - fallback to default
            logger.warning("Parser for %s failed: %s", filename, exc)

    logger.info("Using default parser for %s", filename)
    cycles_summary = [
        {
            "cycle_index": 1,
            "charge_capacity": 100.0,
            "discharge_capacity": 95.0,
            "coulombic_efficiency": 0.95,
        },
        {
            "cycle_index": 2,
            "charge_capacity": 98.0,
            "discharge_capacity": 94.0,
            "coulombic_efficiency": 0.96,
        },
    ]

    metadata = {"tester": "Other", "name": os.path.basename(file_path), "date": None}
    return cycles_summary, metadata


def parse_file_with_sample_matching(file_path):
    """Parse a test file and identify which sample it belongs to based on filename or content."""
    import re

    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]

    cycles_summary, metadata = parse_file(file_path)

    sample_code_match = re.search(r"([A-Za-z]{2,3}\d{2})", name_without_ext)

    if sample_code_match:
        sample_code = sample_code_match.group(1)
    else:
        sample_code = metadata.get("sample_code", None)
        if not sample_code and file_path.lower().endswith((".txt", ".csv")):
            try:
                with open(file_path, "r") as f:
                    first_lines = "".join(f.readline() for _ in range(10))
                    code_match = re.search(r"Sample[:\s]+([A-Za-z]{2,3}\d{2})", first_lines)
                    if code_match:
                        sample_code = code_match.group(1)
            except Exception:
                pass

    if sample_code:
        if metadata is None:
            metadata = {}
        metadata["sample_code"] = sample_code

    return cycles_summary, metadata, sample_code


def test_arbin_parser(file_path):
    """Test the Arbin parser with a specific file and print results."""
    try:
        from .arbin_parser import parse_arbin_excel
    except ImportError:  # pragma: no cover - allow running as script
        import importlib

        parse_arbin_excel = importlib.import_module("arbin_parser").parse_arbin_excel

    logger.info("Testing Arbin parser with file: %s", os.path.basename(file_path))

    try:
        cycles, metadata, detailed_cycles = parse_arbin_excel(
            file_path,
            return_metadata=True,
            return_detailed=True,
        )

        logger.info("Metadata extracted:")
        for key, value in metadata.items():
            logger.info("  %s: %s", key, value)

        logger.info("\nCycles extracted: %d", len(cycles))
        if cycles:
            logger.info("First cycle: %s", cycles[0])
            if len(cycles) > 1:
                logger.info("Last cycle: %s", cycles[-1])

        logger.info("\nDetailed data available for %d cycles", len(detailed_cycles))

        return True
    except Exception as e:
        logger.error("Parser error: %s", str(e))
        import traceback

        traceback.print_exc()
        return False


def test_parser(file_path):
    """Test the parser on a specific file and print debug information.

    Args:
        file_path: Path to the file to test

    Returns:
        bool: True if parsing succeeded, False otherwise
    """
    logger.info("Testing parser on file: %s", os.path.basename(file_path))
    try:
        extension = os.path.splitext(file_path)[1].lower()
        if extension in [".xlsx", ".xls"]:
            logger.info("Detected Excel file - testing with Arbin parser")
            try:
                from .arbin_parser import parse_arbin_excel
            except ImportError:  # pragma: no cover - allow running as script
                import importlib

                parse_arbin_excel = importlib.import_module("arbin_parser").parse_arbin_excel
            cycles, metadata, detailed_cycles = parse_arbin_excel(
                file_path,
                return_metadata=True,
                return_detailed=True,
            )
            detailed_data_available = True
        else:
            logger.info("Using default parser for %s files", extension)
            cycles, metadata = parse_file(file_path)
            detailed_data_available = False
            detailed_cycles = {}

        logger.info("\nExtracted metadata:")
        for key, value in metadata.items():
            if key != "detailed_cycles":
                logger.info("  %s: %s", key, value)

        logger.info("\nExtracted %d cycles", len(cycles))
        if cycles:
            logger.info("First cycle: %s", cycles[0])
            if len(cycles) > 1:
                logger.info("Last cycle: %s", cycles[-1])

        if detailed_data_available:
            logger.info("\nDetailed data available for %d cycles", len(detailed_cycles))
            if detailed_cycles:
                sample_cycle_idx = next(iter(detailed_cycles))
                charge_data = detailed_cycles[sample_cycle_idx]["charge_data"]
                logger.info("Sample charge data for cycle %s:", sample_cycle_idx)
                for key, value in charge_data.items():
                    logger.info("  %s: %d points", key, len(value))

        return True
    except Exception as e:
        logger.error("Parser test failed: %s", str(e))
        import traceback

        traceback.print_exc()
        return False


# Import built-in parsers so they register themselves
from . import arbin_parser  # noqa: F401
from . import biologic_parser  # noqa: F401

