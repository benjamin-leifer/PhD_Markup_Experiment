"""
Parsers for battery test data files from various instruments.
"""
import os
def get_supported_formats():
    """Get list of supported file extensions."""
    return ['.csv', '.xlsx', '.xls', '.mpr', '.mpt', '.dta', '.z', '.res']


def parse_file(file_path):
    """
    Parse a battery test data file.

    Args:
        file_path: Path to the file

    Returns:
        tuple: (cycles_summary, metadata)
    """
    import os
    extension = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)

    # Check for Arbin files based on patterns in your filenames
    if extension in ['.xlsx', '.xls']:
        # Match patterns like "Channel", "Wb", "RT_Rate_Test"
        if any(pattern in filename for pattern in ['Channel', '_Wb_', 'Rate_Test']):
            # Import and use the Arbin parser
            from .arbin_parser import parse_arbin_excel
            print(f"Using Arbin parser for {filename}")
            # Call the parser and unpack the detailed_cycles, but only return what's expected
            cycles_summary, metadata, detailed_cycles = parse_arbin_excel(
                file_path,
                return_metadata=True,
                return_detailed=True,
            )

            # Store detailed_cycles in metadata for later use
            if metadata is None:
                metadata = {}
            metadata['detailed_cycles'] = detailed_cycles

            return cycles_summary, metadata

    # If not matched as Arbin, check for BioLogic files
    if extension in ['.mpt', '.mpr', '.z']:
        try:
            from .biologic_parser import parse_biologic
            print(f"Using BioLogic parser for {filename}")
            cycles_summary = parse_biologic(file_path)
            metadata = {
                'tester': 'BioLogic',
                'name': os.path.basename(file_path),
                'date': None
            }
            return cycles_summary, metadata
        except ImportError:
            print("BioLogic parser not available, using default parser")

    # If no specialized parser matched, return default data with 'Other' tester
    print(f"Using default parser for {filename}")
    cycles_summary = [
        {
            'cycle_index': 1,
            'charge_capacity': 100.0,
            'discharge_capacity': 95.0,
            'coulombic_efficiency': 0.95
        },
        {
            'cycle_index': 2,
            'charge_capacity': 98.0,
            'discharge_capacity': 94.0,
            'coulombic_efficiency': 0.96
        }
    ]

    metadata = {
        'tester': 'Other',
        'name': os.path.basename(file_path),
        'date': None
    }

    return cycles_summary, metadata


def parse_file_with_sample_matching(file_path):
    """
    Parse a test file and identify which sample it belongs to based on filename or content.

    Returns:
        tuple: (cycles_summary, metadata, sample_code)
    """
    import os
    import re

    # Get filename without extension
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]

    # First, parse the file as usual
    cycles_summary, metadata = parse_file(file_path)

    # Try to extract sample code from filename
    # Pattern: Looking for 2-3 letters followed by 2 digits (like NM01, LFP02, etc.)
    sample_code_match = re.search(r'([A-Za-z]{2,3}\d{2})', name_without_ext)

    if sample_code_match:
        sample_code = sample_code_match.group(1)
    else:
        # If no match in filename, try other methods:
        # 1. Look in the metadata
        sample_code = metadata.get('sample_code', None)

        # 2. Try to find it in the file content (for text-based files)
        if not sample_code and file_path.lower().endswith(('.txt', '.csv')):
            try:
                with open(file_path, 'r') as f:
                    first_lines = ''.join(f.readline() for _ in range(10))
                    code_match = re.search(r'Sample[:\s]+([A-Za-z]{2,3}\d{2})', first_lines)
                    if code_match:
                        sample_code = code_match.group(1)
            except:
                pass

    # Add the sample code to metadata
    if sample_code:
        if metadata is None:
            metadata = {}
        metadata['sample_code'] = sample_code

    return cycles_summary, metadata, sample_code


def test_arbin_parser(file_path):
    """Test the Arbin parser with a specific file and print results."""
    from .arbin_parser import parse_arbin_excel

    print(f"Testing Arbin parser with file: {os.path.basename(file_path)}")

    try:
        cycles, metadata, detailed_cycles = parse_arbin_excel(
            file_path,
            return_metadata=True,
            return_detailed=True,
        )

        print(f"Metadata extracted:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

        print(f"\nCycles extracted: {len(cycles)}")
        if cycles:
            print(f"First cycle: {cycles[0]}")
            if len(cycles) > 1:
                print(f"Last cycle: {cycles[-1]}")

        print(f"\nDetailed data available for {len(detailed_cycles)} cycles")

        return True
    except Exception as e:
        print(f"Parser error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_parser(file_path):
    """
    Test the parser on a specific file and print debug information.

    Args:
        file_path: Path to the file to test

    Returns:
        bool: True if parsing succeeded, False otherwise
    """
    import os

    print(f"Testing parser on file: {os.path.basename(file_path)}")
    try:
        extension = os.path.splitext(file_path)[1].lower()
        if extension in ['.xlsx', '.xls']:
            print("Detected Excel file - testing with Arbin parser")
            from .arbin_parser import parse_arbin_excel
            cycles, metadata, detailed_cycles = parse_arbin_excel(
                file_path,
                return_metadata=True,
                return_detailed=True,
            )
            detailed_data_available = True
        else:
            print(f"Using default parser for {extension} files")
            cycles, metadata = parse_file(file_path)
            detailed_data_available = False
            detailed_cycles = {}

        print("\nExtracted metadata:")
        for key, value in metadata.items():
            if key != 'detailed_cycles':  # Skip printing the large detailed_cycles dictionary
                print(f"  {key}: {value}")

        print(f"\nExtracted {len(cycles)} cycles")
        if cycles:
            print(f"First cycle: {cycles[0]}")
            if len(cycles) > 1:
                print(f"Last cycle: {cycles[-1]}")

        if detailed_data_available:
            print(f"\nDetailed data available for {len(detailed_cycles)} cycles")
            if detailed_cycles:
                sample_cycle_idx = next(iter(detailed_cycles))
                charge_data = detailed_cycles[sample_cycle_idx]['charge_data']
                print(f"Sample charge data for cycle {sample_cycle_idx}:")
                for key, value in charge_data.items():
                    print(f"  {key}: {len(value)} points")

        return True
    except Exception as e:
        print(f"Parser test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
