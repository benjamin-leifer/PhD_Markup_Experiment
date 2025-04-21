"""
Parser for Arbin battery test data files.

This module provides functions to parse Arbin battery test files in various formats
(.xlsx, .csv, or .res through conversion) and extract structured cycle data.
"""

import os
import pandas as pd
import numpy as np


def parse_arbin(file_path):
    """
    Parse an Arbin test data file and return a list of cycle summary dictionaries.

    Supports .csv, .xlsx exports. For .res files, user may need to convert to .csv/.xlsx 
    or SQLite first.

    Args:
        file_path (str): Path to the Arbin data file

    Returns:
        list: A list of dictionaries containing cycle summary data

    Raises:
        ValueError: If the file format is not supported
        RuntimeError: If required columns are not found in the data
    """
    # Determine file type by extension
    file_path_lower = file_path.lower()

    if file_path_lower.endswith('.xlsx') or file_path_lower.endswith('.xls'):
        # Arbin Excel export may have multiple sheets
        try:
            # Attempt to read the main data sheet (typical Arbin export)
            df = pd.read_excel(file_path, sheet_name='Channel_Normal_Table')
        except Exception:
            try:
                # Try reading a sheet named "Data" if first attempt fails
                df = pd.read_excel(file_path, sheet_name='Data')
            except Exception:
                # If specific sheets not found, read the first sheet
                df = pd.read_excel(file_path)

    elif file_path_lower.endswith('.csv'):
        df = pd.read_csv(file_path)

    elif file_path_lower.endswith('.res'):
        raise ValueError(
            "Direct .res parsing is not supported in this parser. "
            "Convert .res to .csv/.xlsx or SQLite first, or use the galvani library."
        )

    else:
        raise ValueError(f"Unsupported file format for Arbin parser: {file_path}")

    # Log basic info about the dataframe for debugging
    print(f"Loaded Arbin data with {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")

    # Normalize column names (case-insensitive matching)
    # Create a dictionary mapping lowercase column names to actual column names
    cols_map = {col.lower(): col for col in df.columns}

    # Find cycle index column
    cycle_col = None
    cycle_candidates = ['cycle_index', 'cycle number', 'cycle_number', 'cycle', 'cycle id']
    for candidate in cycle_candidates:
        if candidate in cols_map:
            cycle_col = cols_map[candidate]
            break

    # Try a more general approach if specific names not found
    if cycle_col is None:
        for col in df.columns:
            if 'cycle' in col.lower() and ('index' in col.lower() or 'number' in col.lower() or 'id' in col.lower()):
                cycle_col = col
                break

    # Ensure we found a cycle column
    if cycle_col is None:
        # If no cycle column, try to detect "Step" or "Step_Index" and create cycles based on that
        step_col = None
        for candidate in ['step_index', 'step', 'step number', 'step id']:
            if candidate in cols_map:
                step_col = cols_map[candidate]
                break

        if step_col is not None:
            print(f"No cycle column found, creating cycles from {step_col}")
            # Create cycles based on step transitions
            # Assumes steps alternate between charge, discharge, rest in pairs
            step_values = df[step_col].values
            cycle_values = np.zeros(len(step_values), dtype=int)

            # Heuristic: each pair of steps (charge/discharge) makes a cycle
            current_cycle = 1
            prev_step = step_values[0]

            for i, step in enumerate(step_values):
                if step < prev_step:  # Step reset indicates new cycle
                    current_cycle += 1
                cycle_values[i] = current_cycle
                prev_step = step

            # Add cycle column to dataframe
            df['Calculated_Cycle'] = cycle_values
            cycle_col = 'Calculated_Cycle'
        else:
            raise RuntimeError("Cycle index column not found in Arbin data file.")

    # Find charge and discharge capacity columns
    charge_cap_col = None
    discharge_cap_col = None

    # Candidates for charge capacity column
    charge_candidates = ['charge_capacity', 'chg_capacity', 'charge cap', 'capacity_charge']
    for candidate in charge_candidates:
        if candidate in cols_map:
            charge_cap_col = cols_map[candidate]
            break

    # Candidates for discharge capacity column
    discharge_candidates = ['discharge_capacity', 'dischg_capacity', 'dis_capacity', 'capacity_discharge']
    for candidate in discharge_candidates:
        if candidate in cols_map:
            discharge_cap_col = cols_map[candidate]
            break

    # If still not found, look for any column with charge or discharge in the name
    if charge_cap_col is None:
        for col in df.columns:
            if ('charg' in col.lower() or 'chg' in col.lower()) and 'cap' in col.lower():
                charge_cap_col = col
                break

    if discharge_cap_col is None:
        for col in df.columns:
            if ('discharge' in col.lower() or 'dis' in col.lower()) and 'cap' in col.lower():
                discharge_cap_col = col
                break

    # Find current column (might be needed if capacity columns aren't found)
    current_col = None
    for candidate in ['current', 'current(a)', 'i', 'current_a']:
        if candidate in cols_map:
            current_col = cols_map[candidate]
            break

    # Find capacity column (if separate charge/discharge not available)
    capacity_col = None
    if charge_cap_col is None or discharge_cap_col is None:
        for candidate in ['capacity', 'capacity(ah)', 'capacity_ah']:
            if candidate in cols_map:
                capacity_col = cols_map[candidate]
                break

    # Group data by cycle index to accumulate cycle summaries
    try:
        cycle_groups = df.groupby(df[cycle_col])
    except KeyError:
        # If we still can't find the cycle column, raise an error
        raise RuntimeError(f"Cycle column '{cycle_col}' not found in data. Available columns: {df.columns.tolist()}")

    cycles_summary = []

    for cycle_index, group in cycle_groups:
        # Compute total charge and discharge capacity for this cycle
        charge_capacity = None
        discharge_capacity = None

        # If we have dedicated charge/discharge capacity columns
        if charge_cap_col is not None and charge_cap_col in group.columns:
            charge_capacity = group[charge_cap_col].max()

        if discharge_cap_col is not None and discharge_cap_col in group.columns:
            discharge_capacity = group[discharge_cap_col].max()

        # If we don't have dedicated columns but have current and capacity
        if (
                charge_capacity is None or discharge_capacity is None) and current_col is not None and capacity_col is not None:
            # Filter for charge steps (current > 0) and discharge steps (current < 0)
            charge_data = group[group[current_col] > 0]
            discharge_data = group[group[current_col] < 0]

            if not charge_data.empty and charge_capacity is None:
                # The maximum capacity during charging steps
                charge_capacity = charge_data[capacity_col].max()

            if not discharge_data.empty and discharge_capacity is None:
                # The maximum capacity during discharging steps
                discharge_capacity = discharge_data[capacity_col].max()

        # Fallback for older Arbin files with step_index and capacity
        if (
                charge_capacity is None or discharge_capacity is None) and 'step_index' in group.columns and capacity_col is not None:
            # Assuming even-numbered steps are discharge, odd-numbered are charge
            steps = group['step_index'].unique()

            for step in steps:
                step_data = group[group['step_index'] == step]
                if step % 2 == 0 and discharge_capacity is None:  # Even step, discharge
                    discharge_capacity = step_data[capacity_col].max()
                elif step % 2 == 1 and charge_capacity is None:  # Odd step, charge
                    charge_capacity = step_data[capacity_col].max()

        # Ensure we have values
        if charge_capacity is None:
            charge_capacity = 0.0
        if discharge_capacity is None:
            discharge_capacity = 0.0

        # Compute coulombic efficiency (discharge/charge) for this cycle
        coulombic_eff = float(discharge_capacity / charge_capacity) if charge_capacity and charge_capacity > 0 else 0.0

        # Create the cycle summary
        cycle_data = {
            "cycle_index": int(cycle_index),
            "charge_capacity": float(charge_capacity),
            "discharge_capacity": float(discharge_capacity),
            "coulombic_efficiency": float(coulombic_eff)
        }

        # Extract energy data if available
        charge_energy = None
        discharge_energy = None

        # Look for energy columns
        for col in group.columns:
            if ('charg' in col.lower() or 'chg' in col.lower()) and 'energy' in col.lower():
                charge_energy = group[col].max()
            if ('discharge' in col.lower() or 'dis' in col.lower()) and 'energy' in col.lower():
                discharge_energy = group[col].max()

        # Add energy data if found
        if charge_energy is not None:
            cycle_data["charge_energy"] = float(charge_energy)
        if discharge_energy is not None:
            cycle_data["discharge_energy"] = float(discharge_energy)

        # Add energy efficiency if both energies are available
        if charge_energy is not None and discharge_energy is not None and charge_energy > 0:
            cycle_data["energy_efficiency"] = float(discharge_energy / charge_energy)

        cycles_summary.append(cycle_data)

    # Ensure cycles are sorted by cycle_index
    cycles_summary.sort(key=lambda x: x["cycle_index"])

    return cycles_summary


def extract_test_metadata(file_path):
    """
    Extract test metadata from Arbin file if available.

    Args:
        file_path (str): Path to the Arbin data file

    Returns:
        dict: Dictionary of metadata fields
    """
    metadata = {}
    file_path_lower = file_path.lower()

    try:
        if file_path_lower.endswith('.xlsx') or file_path_lower.endswith('.xls'):
            # Try to read from a sheet named "Info" or "Metadata" if present
            try:
                info_df = pd.read_excel(file_path, sheet_name='Info')
                # Convert to dict if it's key-value format
                if 'Parameter' in info_df.columns and 'Value' in info_df.columns:
                    for _, row in info_df.iterrows():
                        metadata[row['Parameter']] = row['Value']
            except Exception:
                # No metadata sheet
                pass

            # Try to read the main data to extract metadata from column values
            try:
                df = pd.read_excel(file_path)
                # Extract temperature if available
                for col in df.columns:
                    if 'temperature' in col.lower() or 'temp' in col.lower():
                        metadata['temperature'] = df[col].mean()
                    if 'voltage' in col.lower() and ('upper' in col.lower() or 'max' in col.lower()):
                        metadata['upper_cutoff_voltage'] = df[col].max()
                    if 'voltage' in col.lower() and ('lower' in col.lower() or 'min' in col.lower()):
                        metadata['lower_cutoff_voltage'] = df[col].min()
            except Exception:
                pass

        # Add file metadata
        metadata['file_name'] = os.path.basename(file_path)
        metadata['file_path'] = file_path
        metadata['tester'] = 'Arbin'

    except Exception as e:
        print(f"Error extracting metadata: {e}")

    return metadata