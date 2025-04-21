"""
Parser for BioLogic battery test data files.

This module provides functions to parse BioLogic battery test files (.mpr or .mpt)
and extract structured cycle data.
"""

import os
import pandas as pd
import numpy as np
import logging
import re


def parse_biologic(file_path):
    """
    Parse a BioLogic test data file (.mpr or .mpt) and return a list of cycle summary dictionaries.

    Args:
        file_path (str): Path to the BioLogic data file

    Returns:
        list: A list of dictionaries containing cycle summary data

    Raises:
        ValueError: If the file format is not supported
        ImportError: If trying to read .mpr files without the galvani library
    """
    file_path_lower = file_path.lower()
    df = None

    if file_path_lower.endswith('.mpt'):
        # .mpt is a text format (exported from BioLogic EC-Lab)
        # These files have a header with metadata followed by tabular data

        # First, determine the number of header lines to skip
        header_lines = 0
        header_data = {}

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            line_num = 0
            for line in f:
                line_num += 1

                # BioLogic .mpt files typically include a line indicating "Nb header lines"
                if "nb header lines" in line.lower():
                    try:
                        nb_header = int(line.split(':')[1].strip())
                        header_lines = nb_header
                        continue
                    except (ValueError, IndexError):
                        # If parsing fails, continue counting lines
                        pass

                # Extract header metadata
                if ':' in line and line_num <= 100:  # Only look in the first 100 lines
                    try:
                        key, value = line.split(':', 1)
                        header_data[key.strip()] = value.strip()
                    except Exception:
                        pass

                # Break if we've found the header line count or reached a data line
                if line.strip() and line[0].isdigit() and '\t' in line:
                    break

        # Now read the data portion
        try:
            # Try tab-delimiter first (most common)
            df = pd.read_csv(file_path, sep='\t', skiprows=header_lines,
                             decimal='.', thousands=',', encoding='utf-8')
        except Exception:
            try:
                # Try comma-delimiter next
                df = pd.read_csv(file_path, sep=',', skiprows=header_lines,
                                 decimal='.', thousands=',', encoding='utf-8')
            except Exception as e:
                logging.error(f"Error reading .mpt file: {e}")
                # Try a more robust fallback method
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.readlines()[header_lines:]

                # Find the header line
                header_line = None
                for i, line in enumerate(content):
                    if line.strip() and not line[0].isdigit():
                        header_line = line
                        content = content[i + 1:]
                        break

                if header_line:
                    # Parse the header to get column names
                    if '\t' in header_line:
                        headers = [h.strip() for h in header_line.split('\t')]
                    else:
                        headers = [h.strip() for h in header_line.split(',')]

                    # Parse data rows
                    data = []
                    for line in content:
                        if not line.strip():
                            continue
                        if '\t' in line:
                            row = [val.strip() for val in line.split('\t')]
                        else:
                            row = [val.strip() for val in line.split(',')]

                        # Convert to proper types
                        row_data = {}
                        for i, val in enumerate(row):
                            if i < len(headers):
                                try:
                                    row_data[headers[i]] = float(val)
                                except ValueError:
                                    row_data[headers[i]] = val

                        data.append(row_data)

                    # Create DataFrame
                    df = pd.DataFrame(data)

    elif file_path_lower.endswith('.mpr'):
        try:
            # Attempt to use galvani if available
            from galvani import BioLogic

            # Load the binary file
            mpr = BioLogic.MPRfile(file_path)

            # mpr.data is typically a structured numpy array
            df = pd.DataFrame(mpr.data)

            # Add metadata if available
            # if hasattr(mpr, 'metadata'):
            #     print(f"Metadata from MPR file: {mpr.metadata}")

        except ImportError:
            raise ImportError(
                "Reading .mpr requires the 'galvani' library. "
                "Please install galvani or export .mpr to .mpt format."
            )
        except Exception as e:
            raise ValueError(f"Could not parse BioLogic .mpr file: {e}")

    else:
        raise ValueError(f"Unsupported file format for BioLogic parser: {file_path}")

    if df is None or df.empty:
        raise ValueError(f"Could not extract data from file: {file_path}")

    # Log basic info about the dataframe for debugging
    print(f"Loaded BioLogic data with {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")

    # Create a dictionary mapping lowercase column names to actual column names
    cols_map = {str(col).lower(): col for col in df.columns}

    # Find cycle or loop number column
    cycle_col = None
    cycle_candidates = ['cycle number', 'cycle_number', 'loop number', 'loop_number', 'cycle', 'loop']

    for candidate in cycle_candidates:
        if candidate in cols_map:
            cycle_col = cols_map[candidate]
            break

    # Try a more general approach if specific names not found
    if cycle_col is None:
        for col in df.columns:
            col_str = str(col).lower()
            if ('cycle' in col_str or 'loop' in col_str) and (
                    'number' in col_str or 'index' in col_str or 'nb' in col_str):
                cycle_col = col
                break

    # Create cycles based on half-cycle, mode, or step index if cycle column not found
    if cycle_col is None:
        # Try to find half-cycle or mode columns
        half_cycle_col = None
        mode_col = None

        for col in df.columns:
            col_str = str(col).lower()
            if 'half cycle' in col_str or 'half_cycle' in col_str:
                half_cycle_col = col
            elif 'mode' in col_str or 'state' in col_str:
                mode_col = col

        if half_cycle_col is not None:
            # Create cycle numbers (integer division by 2 of half-cycle + 1)
            df['Calculated_Cycle'] = (df[half_cycle_col] // 2) + 1
            cycle_col = 'Calculated_Cycle'
            print(f"Created cycle column from half-cycle column: {half_cycle_col}")

        elif mode_col is not None:
            # Create cycles based on mode changes
            # Typically mode/state 1=charge, 2=discharge, etc.
            mode_values = df[mode_col].values
            cycle_values = np.ones(len(mode_values), dtype=int)

            # Increment cycle on charge to discharge transition or reset to 1
            current_cycle = 1
            for i in range(1, len(mode_values)):
                # If we transition from charge to discharge, increment cycle
                if (mode_values[i - 1] == 1 and mode_values[i] == 2) or mode_values[i - 1] > mode_values[i]:
                    current_cycle += 1
                cycle_values[i] = current_cycle

            df['Calculated_Cycle'] = cycle_values
            cycle_col = 'Calculated_Cycle'
            print(f"Created cycle column from mode/state column: {mode_col}")

        elif all(col in df.columns for col in ['time/s', 'I/mA']):
            # Create cycles based on current sign changes
            # This approach works for simple C-D cycling with BioLogic
            current_values = df['I/mA'].values
            time_values = df['time/s'].values
            cycle_values = np.ones(len(current_values), dtype=int)

            current_cycle = 1
            sign_changed = False

            for i in range(1, len(current_values)):
                # If current changes from positive to negative, consider it's a new half-cycle
                if current_values[i - 1] > 0 and current_values[i] < 0:
                    sign_changed = True
                # After sign change from pos to neg, if it goes back to positive, it's a new cycle
                elif sign_changed and current_values[i - 1] < 0 and current_values[i] > 0:
                    current_cycle += 1
                    sign_changed = False

                cycle_values[i] = current_cycle

            df['Calculated_Cycle'] = cycle_values
            cycle_col = 'Calculated_Cycle'
            print("Created cycle column from current sign changes")

        else:
            # If we can't detect cycles, just treat the whole dataset as one cycle
            df['Calculated_Cycle'] = 1
            cycle_col = 'Calculated_Cycle'
            print("No cycle information found, treating all data as one cycle")

    # Identify capacity, current, and voltage columns
    capacity_col = None
    charge_cap_col = None
    discharge_cap_col = None
    current_col = None
    voltage_col = None

    # Common BioLogic column names
    for col in df.columns:
        col_str = str(col).lower()

        # Current column detection
        if col_str in ['i/ma', 'i', 'i/a', 'current', 'current/ma', 'current/a']:
            current_col = col

        # Voltage column detection
        elif col_str in ['ewe/v', 'ece/v', 'e/v', 'voltage', 'voltage/v', 'ecell/v']:
            voltage_col = col

        # Capacity column detection
        elif 'q charge' in col_str or 'qcharge' in col_str:
            charge_cap_col = col
        elif 'q discharge' in col_str or 'qdischarge' in col_str:
            discharge_cap_col = col
        elif col_str in ['q/ma.h', 'capacity/ma.h', 'capacity', 'q', 'q/mah']:
            capacity_col = col

    # Group data by cycle
    cycles_summary = []

    try:
        # Group by cycle
        cycle_groups = df.groupby(df[cycle_col])

        for cycle_index, group in cycle_groups:
            # Skip cycle 0 if present (often used for initialization)
            if cycle_index == 0 and len(cycle_groups) > 1:
                continue

            # Initialize with default values
            charge_capacity = None
            discharge_capacity = None

            # Case 1: Separate charge and discharge capacity columns
            if charge_cap_col is not None and discharge_cap_col is not None:
                charge_capacity = group[charge_cap_col].max()
                discharge_capacity = group[discharge_cap_col].max()

            # Case 2: Cumulative capacity with current direction
            elif capacity_col is not None and current_col is not None:
                # Separate by current direction
                charge_rows = group[group[current_col] > 0]
                discharge_rows = group[group[current_col] < 0]

                if not charge_rows.empty:
                    charge_capacity = charge_rows[capacity_col].max() - charge_rows[capacity_col].min()

                if not discharge_rows.empty:
                    discharge_capacity = discharge_rows[capacity_col].max() - discharge_rows[capacity_col].min()

            # Case 3: Rely on capacity column alone (more error-prone)
            elif capacity_col is not None:
                # Try to identify charge and discharge segments by capacity increase/decrease
                cap_values = group[capacity_col].values
                time_ordered = True

                # Check if data is time-ordered for this approach to work
                if 'time/s' in group.columns:
                    time_ordered = group['time/s'].is_monotonic_increasing

                if time_ordered:
                    # Find the max capacity point in this cycle
                    max_cap_idx = np.argmax(cap_values)

                    # Capacity increase (usually charging)
                    if max_cap_idx > 0:
                        charge_capacity = cap_values[max_cap_idx] - cap_values[0]

                    # Capacity decrease after max (usually discharging)
                    if max_cap_idx < len(cap_values) - 1:
                        discharge_capacity = cap_values[max_cap_idx] - cap_values[-1]

            # Ensure we have valid capacity values
            if charge_capacity is None or np.isnan(charge_capacity):
                charge_capacity = 0.0
            if discharge_capacity is None or np.isnan(discharge_capacity):
                discharge_capacity = 0.0

            # Make sure capacities are positive (in case of sign issues)
            charge_capacity = abs(float(charge_capacity))
            discharge_capacity = abs(float(discharge_capacity))

            # Calculate coulombic efficiency
            coulombic_eff = discharge_capacity / charge_capacity if charge_capacity > 0 else 0.0

            # Create cycle summary
            cycle_data = {
                "cycle_index": int(cycle_index),
                "charge_capacity": charge_capacity,
                "discharge_capacity": discharge_capacity,
                "coulombic_efficiency": float(coulombic_eff)
            }

            # Extract energy data if available
            for col in group.columns:
                col_str = str(col).lower()
                if 'energy charge' in col_str or 'echarge' in col_str:
                    cycle_data["charge_energy"] = float(group[col].max())
                elif 'energy discharge' in col_str or 'edischarge' in col_str:
                    cycle_data["discharge_energy"] = float(group[col].max())

            # Add energy efficiency if both energies are available
            if "charge_energy" in cycle_data and "discharge_energy" in cycle_data and cycle_data["charge_energy"] > 0:
                cycle_data["energy_efficiency"] = float(cycle_data["discharge_energy"] / cycle_data["charge_energy"])

            cycles_summary.append(cycle_data)

    except Exception as e:
        logging.error(f"Error processing cycles: {e}")
        # If grouping fails, create a single cycle summary from the entire dataset
        if capacity_col is not None:
            capacity_range = df[capacity_col].max() - df[capacity_col].min()
            cycles_summary = [{
                "cycle_index": 1,
                "charge_capacity": capacity_range,
                "discharge_capacity": capacity_range,
                "coulombic_efficiency": 1.0
            }]

    # Ensure cycles are sorted by index
    cycles_summary.sort(key=lambda x: x["cycle_index"])

    return cycles_summary


def extract_test_metadata(file_path):
    """
    Extract test metadata from BioLogic file if available.

    Args:
        file_path (str): Path to the BioLogic data file

    Returns:
        dict: Dictionary of metadata fields
    """
    metadata = {}
    file_path_lower = file_path.lower()

    # Initialize with file info
    metadata['file_name'] = os.path.basename(file_path)
    metadata['file_path'] = file_path
    metadata['tester'] = 'BioLogic'

    try:
        if file_path_lower.endswith('.mpt'):
            # Read header section
            header_data = {}
            header_lines = 100  # Just read first 100 lines max to find header

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= header_lines:
                        break

                    # Extract metadata from header
                    if ':' in line:
                        try:
                            key, value = line.split(':', 1)
                            header_data[key.strip()] = value.strip()
                        except Exception:
                            pass

            # Extract common metadata fields
            if 'Ewe control' in header_data:
                metadata['control_mode'] = header_data['Ewe control']

            # Extract voltage limits
            for key, value in header_data.items():
                if 'e low limit' in key.lower():
                    try:
                        metadata['lower_cutoff_voltage'] = float(value.split()[0])
                    except (ValueError, IndexError):
                        pass
                elif 'e high limit' in key.lower():
                    try:
                        metadata['upper_cutoff_voltage'] = float(value.split()[0])
                    except (ValueError, IndexError):
                        pass
                elif 'temperature' in key.lower():
                    try:
                        metadata['temperature'] = float(value.split()[0])
                    except (ValueError, IndexError):
                        pass

            # Try to determine C-rates from header or filename
            if 'I Range' in header_data:
                metadata['current_range'] = header_data['I Range']

        elif file_path_lower.endswith('.mpr'):
            try:
                from galvani import BioLogic
                mpr = BioLogic.MPRfile(file_path)

                # Extract metadata if available
                if hasattr(mpr, 'metadata'):
                    for key, value in mpr.metadata.items():
                        if isinstance(value, (int, float, str, bool)):
                            metadata[key] = value

            except ImportError:
                pass

    except Exception as e:
        logging.error(f"Error extracting BioLogic metadata: {e}")

    return metadata