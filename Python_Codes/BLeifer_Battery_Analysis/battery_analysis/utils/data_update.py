"""
Utilities for handling data updates and file processing.
"""
import os
import logging
import datetime

# Update imports to avoid circular dependencies
from battery_analysis import models, parsers
# Import directly from the analysis package
from battery_analysis.analysis import compute_metrics, update_sample_properties, create_test_result

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
    identifiers = {
        'file_name': os.path.basename(file_path),
        'tester': metadata.get('tester', None),
        'test_name': metadata.get('name', None)
    }

    # Add more identifying information as available
    if 'date' in metadata:
        identifiers['date'] = metadata['date']

    if 'test_id' in metadata:
        identifiers['test_id'] = metadata['test_id']

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

    matches = []

    # Check each test for this sample
    for test_ref in sample.tests:
        test = models.TestResult.objects(id=test_ref.id).first()
        if not test:
            continue

        # Check for name match
        if identifiers.get('test_name') and test.name == identifiers['test_name']:
            matches.append(test)
            continue

        # Check for file name match (if test has file_path)
        if identifiers.get('file_name') and hasattr(test, 'file_path') and test.file_path:
            if os.path.basename(test.file_path) == identifiers['file_name']:
                matches.append(test)
                continue

    return matches

def update_test_data(existing_test, new_cycles, metadata, strategy='append'):
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
    if strategy == 'replace':
        # Replace all cycles
        existing_test.cycles = []

    # Get existing cycle indices
    existing_indices = set(c.cycle_index for c in existing_test.cycles)

    # Add new cycles that don't exist yet
    for cycle in new_cycles:
        if strategy == 'replace' or cycle['cycle_index'] not in existing_indices:
            # Create and add the cycle
            cycle_doc = models.CycleSummary(
                cycle_index=cycle['cycle_index'],
                charge_capacity=cycle['charge_capacity'],
                discharge_capacity=cycle['discharge_capacity'],
                coulombic_efficiency=cycle['coulombic_efficiency']
            )

            # Add optional fields if present
            if 'charge_energy' in cycle and cycle['charge_energy'] is not None:
                cycle_doc.charge_energy = cycle['charge_energy']

            if 'discharge_energy' in cycle and cycle['discharge_energy'] is not None:
                cycle_doc.discharge_energy = cycle['discharge_energy']

            if 'energy_efficiency' in cycle and cycle['energy_efficiency'] is not None:
                cycle_doc.energy_efficiency = cycle['energy_efficiency']

            if 'internal_resistance' in cycle and cycle['internal_resistance'] is not None:
                cycle_doc.internal_resistance = cycle['internal_resistance']

            existing_test.cycles.append(cycle_doc)

    # Update metadata if provided
    if metadata:
        # Extract detailed_cycles if present, then remove from metadata
        detailed_cycles = None
        if 'detailed_cycles' in metadata:
            detailed_cycles = metadata.pop('detailed_cycles')

        # Update any existing fields
        for key, value in metadata.items():
            if hasattr(existing_test, key):
                setattr(existing_test, key, value)

        # Update custom_data
        if not hasattr(existing_test, 'custom_data') or existing_test.custom_data is None:
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
            from battery_analysis.utils.detailed_data_manager import store_detailed_cycle_data
            store_detailed_cycle_data(str(existing_test.id), detailed_cycles)
            logging.info(f"Stored {len(detailed_cycles)} detailed cycles for test {existing_test.id}")
        except Exception as e:
            logging.error(f"Error storing detailed cycle data: {e}")

    # Update the sample
    update_sample_properties(existing_test.sample.fetch())

    return existing_test

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

    # Extract detailed_cycles if present in metadata
    detailed_cycles = None
    if metadata and 'detailed_cycles' in metadata:
        detailed_cycles = metadata.pop('detailed_cycles')
        logging.info(f"Extracted {len(detailed_cycles)} detailed cycle datasets")

    # Add file path to metadata
    if metadata is None:
        metadata = {}
    metadata['file_path'] = file_path

    # Extract identifiers
    identifiers = extract_test_identifiers(file_path, parsed_data, metadata)

    # Look for matching tests
    matching_tests = find_matching_tests(identifiers, sample.id)

    if matching_tests:
        # Update the existing test
        existing_test = matching_tests[0]
        updated_test = update_test_data(existing_test, parsed_data, metadata, strategy='append')

        # Store detailed cycle data if available
        if detailed_cycles:
            try:
                from battery_analysis.utils.detailed_data_manager import store_detailed_cycle_data
                store_detailed_cycle_data(str(updated_test.id), detailed_cycles)
                logging.info(f"Stored {len(detailed_cycles)} detailed cycles for test {updated_test.id}")
            except Exception as e:
                logging.error(f"Error storing detailed cycle data: {e}")

        return updated_test, True
    else:
        # Create a new test
        test_result = create_test_result(
            sample=sample,
            cycles_summary=parsed_data,
            tester=metadata.get('tester', 'Unknown'),
            metadata=metadata
        )

        # Store detailed cycle data if available
        if detailed_cycles:
            try:
                from battery_analysis.utils.detailed_data_manager import store_detailed_cycle_data
                store_detailed_cycle_data(str(test_result.id), detailed_cycles)
                logging.info(f"Stored {len(detailed_cycles)} detailed cycles for test {test_result.id}")
            except Exception as e:
                logging.error(f"Error storing detailed cycle data: {e}")

        return test_result, False
