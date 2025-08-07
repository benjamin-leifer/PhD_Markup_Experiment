"""
Functions for analyzing detailed cycle data stored in GridFS.

This module provides functions for retrieving and analyzing detailed cycle data,
which may include voltage curves, current profiles, and capacity data for individual cycles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, interpolate
import logging


def get_cycle_voltage_vs_capacity(test_id, cycle_number):
    """
    Retrieve voltage vs capacity data for a specific cycle.

    Args:
        test_id: ID of the TestResult to analyze
        cycle_number: Specific cycle to extract

    Returns:
        dict: Dictionary with charge and discharge data
    """
    from battery_analysis.models import TestResult

    # Try to get data from GridFS via the model helper
    test = TestResult.objects(id=test_id).first()
    if not test:
        raise ValueError(f"Test with ID {test_id} not found")

    cycle_data = test.get_cycle_detail(cycle_number)

    if cycle_data:
        charge_data = cycle_data.get("charge", {})
        discharge_data = cycle_data.get("discharge", {})

        if (
            "voltage" in charge_data
            and "capacity" in charge_data
            and "voltage" in discharge_data
            and "capacity" in discharge_data
        ):
            return cycle_data

    # If no detailed data in GridFS, try to get it from the CycleSummary document

    for cycle in test.cycles:
        if cycle.cycle_index == cycle_number:
            # Check if this cycle has detailed data stored inline
            if (hasattr(cycle, 'voltage_charge') and cycle.voltage_charge and
                    hasattr(cycle, 'capacity_charge') and cycle.capacity_charge and
                    hasattr(cycle, 'voltage_discharge') and cycle.voltage_discharge and
                    hasattr(cycle, 'capacity_discharge') and cycle.capacity_discharge):
                return {
                    'charge': {
                        'voltage': cycle.voltage_charge,
                        'capacity': cycle.capacity_charge,
                        'current': cycle.current_charge if hasattr(cycle, 'current_charge') else [],
                        'time': cycle.time_charge if hasattr(cycle, 'time_charge') else []
                    },
                    'discharge': {
                        'voltage': cycle.voltage_discharge,
                        'capacity': cycle.capacity_discharge,
                        'current': cycle.current_discharge if hasattr(cycle, 'current_discharge') else [],
                        'time': cycle.time_discharge if hasattr(cycle, 'time_discharge') else []
                    }
                }

            break

    raise ValueError(f"No detailed data available for cycle {cycle_number} of test {test_id}")


def calculate_differential_capacity(test_id, cycle_number, smoothing=True):
    """
    Calculate differential capacity (dQ/dV) for a specific cycle.

    Args:
        test_id: ID of the TestResult to analyze
        cycle_number: Specific cycle to analyze
        smoothing: Whether to apply smoothing to the result

    Returns:
        dict: Dictionary with charge and discharge dQ/dV data
    """
    # Get the detailed cycle data
    cycle_data = get_cycle_voltage_vs_capacity(test_id, cycle_number)

    result = {'charge': {}, 'discharge': {}}

    # Process charge data
    if 'charge' in cycle_data and 'voltage' in cycle_data['charge'] and 'capacity' in cycle_data['charge']:
        charge_result = _calculate_dqdv_segment(
            cycle_data['charge']['voltage'],
            cycle_data['charge']['capacity'],
            smoothing,
            segment_type='charge'
        )
        if charge_result:
            result['charge'] = charge_result

    # Process discharge data
    if 'discharge' in cycle_data and 'voltage' in cycle_data['discharge'] and 'capacity' in cycle_data['discharge']:
        discharge_result = _calculate_dqdv_segment(
            cycle_data['discharge']['voltage'],
            cycle_data['discharge']['capacity'],
            smoothing,
            segment_type='discharge'
        )
        if discharge_result:
            result['discharge'] = discharge_result

    return result


def _calculate_dqdv_segment(voltage, capacity, smoothing=True, segment_type='charge'):
    """
    Helper function to calculate dQ/dV for a specific segment (charge or discharge).

    Args:
        voltage: Array of voltage values
        capacity: Array of capacity values
        smoothing: Whether to apply smoothing
        segment_type: 'charge' or 'discharge'

    Returns:
        dict: Dictionary with voltage and dQ/dV arrays
    """
    try:
        # Convert inputs to numpy arrays if they aren't already
        v = np.asarray(voltage, dtype=float)
        q = np.asarray(capacity, dtype=float)

        # Check for empty data
        if len(v) < 3 or len(q) < 3:
            logging.warning(f"Not enough data points for dQ/dV calculation ({len(v)} points)")
            return None

        # Sort data based on segment type
        if segment_type == 'charge':
            # For charge, sort by increasing voltage
            sort_idx = np.argsort(v)
        else:
            # For discharge, sort by decreasing voltage
            sort_idx = np.argsort(v)[::-1]

        v_sorted = v[sort_idx]
        q_sorted = q[sort_idx]

        # Remove duplicates in voltage (they cause issues with gradient calculation)
        if segment_type == 'charge':
            unique_idx = np.concatenate(([True], np.diff(v_sorted) > 1e-5))
        else:
            unique_idx = np.concatenate(([True], np.diff(v_sorted) < -1e-5))

        v_unique = v_sorted[unique_idx]
        q_unique = q_sorted[unique_idx]

        # Check if we have enough points
        if len(v_unique) <= 3:
            logging.warning(f"Not enough unique voltage points for dQ/dV calculation ({len(v_unique)} points)")
            return None

        # Calculate numerical derivative (dQ/dV)
        dq = np.diff(q_unique)
        dv = np.diff(v_unique)

        # Avoid division by zero or very small numbers
        valid_idx = np.abs(dv) > 1e-6
        dqdv = np.zeros_like(dv)
        dqdv[valid_idx] = dq[valid_idx] / dv[valid_idx]

        # Calculate voltage centers (midpoints between voltage values)
        v_centers = (v_unique[1:] + v_unique[:-1]) / 2

        # Apply smoothing if requested
        if smoothing and len(dqdv) > 10:
            try:
                from scipy.signal import savgol_filter

                # Set window size to be odd and less than the data length
                window = min(11, len(dqdv) - 2 if len(dqdv) % 2 == 0 else len(dqdv) - 1)
                window = max(3, window - 1 if window % 2 == 0 else window)

                dqdv_smooth = savgol_filter(dqdv, window, 1)

                return {
                    'voltage': v_centers.tolist(),
                    'dqdv': dqdv_smooth.tolist()
                }
            except Exception as e:
                logging.warning(f"Smoothing failed: {e}, returning unsmoothed data")

        # Return unsmoothed data if smoothing isn't requested or failed
        return {
            'voltage': v_centers.tolist(),
            'dqdv': dqdv.tolist()
        }

    except Exception as e:
        logging.error(f"Error calculating dQ/dV: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None


def plot_cycle_voltage_capacity(test_id, cycle_number):
    """
    Generate a voltage vs. capacity plot for a specific cycle.

    Args:
        test_id: ID of the TestResult to analyze
        cycle_number: Specific cycle to plot

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        # Get the cycle data
        cycle_data = get_cycle_voltage_vs_capacity(test_id, cycle_number)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot charge data
        if 'charge' in cycle_data and 'voltage' in cycle_data['charge'] and 'capacity' in cycle_data['charge']:
            ax.plot(
                cycle_data['charge']['capacity'],
                cycle_data['charge']['voltage'],
                'b-',
                label='Charge'
            )

        # Plot discharge data
        if 'discharge' in cycle_data and 'voltage' in cycle_data['discharge'] and 'capacity' in cycle_data['discharge']:
            ax.plot(
                cycle_data['discharge']['capacity'],
                cycle_data['discharge']['voltage'],
                'r-',
                label='Discharge'
            )

        # Set plot properties
        ax.set_xlabel('Capacity (mAh)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'Voltage vs. Capacity - Cycle {cycle_number}')
        ax.grid(True)
        ax.legend()

        return fig

    except Exception as e:
        print(f"Error plotting voltage-capacity: {e}")

        # Return an error figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error: {str(e)}",
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                color='red')
        ax.axis('off')
        return fig


def plot_differential_capacity(test_id, cycle_number, smoothing=True):
    """
    Generate a differential capacity (dQ/dV) plot for a specific cycle.

    Args:
        test_id: ID of the TestResult to analyze
        cycle_number: Specific cycle to plot
        smoothing: Whether to apply smoothing to the dQ/dV data

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        # Get the detailed cycle data
        logging.info(f"Getting data for dQ/dV calculation: test={test_id}, cycle={cycle_number}")
        cycle_data = get_cycle_voltage_vs_capacity(test_id, cycle_number)

        # Log what data was found
        if 'charge' in cycle_data and 'voltage' in cycle_data['charge']:
            logging.info(f"Found charge data: {len(cycle_data['charge']['voltage'])} points")
        if 'discharge' in cycle_data and 'voltage' in cycle_data['discharge']:
            logging.info(f"Found discharge data: {len(cycle_data['discharge']['voltage'])} points")

        try:
            # Calculate dQ/dV
            dqdv_data = calculate_differential_capacity(test_id, cycle_number, smoothing)

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot charge data
            if 'charge' in dqdv_data and dqdv_data['charge']:
                ax.plot(
                    dqdv_data['charge']['voltage'],
                    dqdv_data['charge']['dqdv'],
                    'b-',
                    label='Charge'
                )

            # Plot discharge data
            if 'discharge' in dqdv_data and dqdv_data['discharge']:
                ax.plot(
                    dqdv_data['discharge']['voltage'],
                    dqdv_data['discharge']['dqdv'],
                    'r-',
                    label='Discharge'
                )

            # Set plot properties
            ax.set_xlabel('Voltage (V)')
            ax.set_ylabel('dQ/dV (mAh/V)')
            ax.set_title(f'Differential Capacity Analysis - Cycle {cycle_number}')
            ax.grid(True)
            ax.legend()

            # Detect peaks if scipy is available
            try:
                from scipy.signal import find_peaks

                # Find peaks in charge data
                # Find peaks in charge data
                if 'charge' in dqdv_data and dqdv_data['charge']:
                    # Make sure data is a numpy array
                    dqdv_array = np.array(dqdv_data['charge']['dqdv'])
                    charge_peaks, _ = find_peaks(dqdv_array,
                                                 height=0.1 * max(abs(dqdv_array)) if len(dqdv_array) > 0 else 0)

                    # Explicitly convert peaks to integers
                    charge_peaks = [int(idx) for idx in charge_peaks]

                    for idx in charge_peaks:
                        if 0 <= idx < len(dqdv_data['charge']['voltage']):  # Safety check
                            voltage = dqdv_data['charge']['voltage'][idx]
                            dqdv = dqdv_data['charge']['dqdv'][idx]
                            ax.plot(voltage, dqdv, 'bo', markersize=8)
                            ax.annotate(f"{voltage:.2f}V",
                                        (voltage, dqdv),
                                        xytext=(5, 5),
                                        textcoords='offset points')

                # Find peaks in discharge data
                if 'discharge' in dqdv_data and dqdv_data['discharge']:
                    # Make sure data is a numpy array
                    dqdv_array = np.array(dqdv_data['discharge']['dqdv'])
                    height_threshold = 0.1 * max(abs(dqdv_array)) if len(dqdv_array) > 0 else 0
                    discharge_peaks, _ = find_peaks(dqdv_array, height=height_threshold)

                    # Explicitly convert peaks to integers
                    discharge_peaks = [int(idx) for idx in discharge_peaks]

                    for idx in discharge_peaks:
                        if 0 <= idx < len(dqdv_data['discharge']['voltage']):  # Safety check
                            voltage = dqdv_data['discharge']['voltage'][idx]
                            dqdv = dqdv_data['discharge']['dqdv'][idx]
                            ax.plot(voltage, dqdv, 'ro', markersize=8)
                            ax.annotate(f"{voltage:.2f}V",
                                        (voltage, dqdv),
                                        xytext=(5, 5),
                                        textcoords='offset points')
            except:
                # Skip peak detection if scipy is not available
                pass

            return fig

        except Exception as e:
            print(f"Error plotting dQ/dV: {e}")

            # Return an error figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error: {str(e)}",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    color='red')
            ax.axis('off')
            return fig

    except Exception as e:
        logging.error(f"Error in dQ/dV calculation: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise
