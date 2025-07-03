"""
Advanced analysis module for battery test data.

This module provides advanced electrochemical analysis techniques for battery test data,
including differential capacity analysis, voltage curve analysis, and cycle life prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, interpolate, optimize, stats
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from . import models
from . import utils
from . import pybamm_models

# ---------------------------------------------------------------------------
# Column-name aliases so the code works with Arbin, BioLogic, Neware, etc.
# ---------------------------------------------------------------------------
ALIAS_CAPACITY = {
    "capacity",
    "Capacity",
    "q-q0",
    "Cap. Discharge",
    "Cap. Charge",
    "Specific Capacity",
}

ALIAS_VOLTAGE = {
    "voltage",
    "Voltage",
    "Ecell / V",
    "Ewe/V",
    "Ecell_V",
    "Potential (V)",
}

def _get_series(df, aliases):
    """Return the first matching column in *df* whose name is in *aliases*."""
    for key in aliases:
        if key in df.columns:
            return df[key]
    raise KeyError(
        f"None of the expected column names {sorted(aliases)} "
        f"were found in {list(df.columns)}"
    )


def get_voltage_capacity_data(test_id, cycle_number=None):
    """
    Return voltage-vs-capacity arrays for *test_id*.

    The routine now recognises multiple column-name variants (Arbin, BioLogic,
    Neware, etc.) so the “Could not identify required data columns” error
    should disappear.

    Parameters
    ----------
    test_id : ObjectId | str
        ID of the TestResult document.
    cycle_number : int | None, optional
        If given, only that cycle is returned; otherwise all cycles.

    Returns
    -------
    tuple[list[float], list[float]]
        (voltages, capacities)
    """
    import logging
    import os
    import pandas as pd
    from battery_analysis import models

    # ------------------------------------------------------------------ #
    # 1)  Try detailed data stored in GridFS
    # ------------------------------------------------------------------ #
    try:
        from battery_analysis.utils.detailed_data_manager import (
            get_detailed_cycle_data,
        )

        detailed_data = get_detailed_cycle_data(test_id, cycle_number)

        if detailed_data and cycle_number in detailed_data:
            cycle_data = detailed_data[cycle_number]

            if (
                "charge" in cycle_data
                and "voltage" in cycle_data["charge"]
                and "capacity" in cycle_data["charge"]
            ):
                return (
                    cycle_data["charge"]["voltage"],
                    cycle_data["charge"]["capacity"],
                )
            elif (
                "discharge" in cycle_data
                and "voltage" in cycle_data["discharge"]
                and "capacity" in cycle_data["discharge"]
            ):
                return (
                    cycle_data["discharge"]["voltage"],
                    cycle_data["discharge"]["capacity"],
                )

        logging.warning(
            f"GridFS data not found or incomplete for test {test_id}, "
            f"cycle {cycle_number}"
        )
    except Exception as exc:
        logging.warning(f"Error retrieving GridFS data for test {test_id}: {exc}")

    # ------------------------------------------------------------------ #
    # 2)  Fall back to the TestResult document and original data file
    # ------------------------------------------------------------------ #
    test = models.TestResult.objects(id=test_id).first()
    if test is None:
        raise ValueError(f"Test with ID {test_id} not found")

    if getattr(test, "file_path", None) and os.path.exists(test.file_path):
        try:
            logging.info(f"Reading data from original file: {test.file_path}")
            file_path = test.file_path
            file_ext = os.path.splitext(file_path)[1].lower()

            # ---------------- Excel / CSV into DataFrame ---------------- #
            if file_ext in {".xlsx", ".xls"}:
                xl = pd.ExcelFile(file_path)
                # pick a sheet that looks like data
                sheet_name = next(
                    (s for s in xl.sheet_names if "channel" in s.lower() or "data" in s.lower()),
                    xl.sheet_names[0],
                )
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            elif file_ext == ".csv":
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            # -------------- robust column-name matching ----------------- #
            VOLT_ALIASES = {
                "voltage",
                "potential",
                "volt",
                "ewe/v",
                "ewe",
                "ecell / v",
                "ecell_v",
                "cell voltage",
            }
            CAP_ALIASES = {
                "capacity",
                "q-q0",
                "cap. discharge",
                "cap discharge",
                "cap. charge",
                "cap charge",
                "specific capacity",
                "capacity (mah)",
                "cap(",
            }

            def _match_alias(col_name: str, aliases) -> bool:
                col = col_name.lower()
                return any(alias in col for alias in aliases)

            voltage_col = next((c for c in df.columns if _match_alias(c, VOLT_ALIASES)), None)
            capacity_col = next((c for c in df.columns if _match_alias(c, CAP_ALIASES)), None)

            if voltage_col is None or capacity_col is None:
                raise ValueError("Could not identify required data columns")

            # ------------ optional cycle filter ------------------------- #
            if cycle_number is not None:
                cycle_col = next(
                    (c for c in df.columns if "cycle" in str(c).lower() or "cyc" in str(c).lower()),
                    None,
                )
                if cycle_col is not None:
                    df = df[df[cycle_col] == cycle_number]

            return df[voltage_col].values.tolist(), df[capacity_col].values.tolist()

        except Exception as exc:
            logging.error(f"Error extracting data from file: {exc}")

    # ------------------------------------------------------------------ #
    # 3)  Still nothing?  Raise an error that caller can catch.
    # ------------------------------------------------------------------ #
    raise ValueError("Could not identify required data columns")



def differential_capacity_analysis(voltage_data, capacity_data, smooth=True, window_size=11):
    """
    Perform differential capacity analysis (dQ/dV) to identify phase transitions.

    Args:
        voltage_data: Array of voltage values
        capacity_data: Array of capacity values
        smooth: Whether to apply smoothing to the derivative (default: True)
        window_size: Window size for Savitzky-Golay filter if smoothing (default: 11)

    Returns:
        tuple: (voltage_centers, dq_dv_values) for plotting differential capacity
    """
    # Ensure data is sorted by voltage
    sort_idx = np.argsort(voltage_data)
    voltage_sorted = voltage_data[sort_idx]
    capacity_sorted = capacity_data[sort_idx]

    # Remove duplicates in voltage (they cause issues with gradient calculation)
    unique_mask = np.r_[True, np.diff(voltage_sorted) > 1e-5]
    voltage_unique = voltage_sorted[unique_mask]
    capacity_unique = capacity_sorted[unique_mask]

    if len(voltage_unique) < 10:
        raise ValueError("Not enough unique voltage points for analysis")

    # Create a smoothed capacity curve using interpolation
    voltage_fine = np.linspace(voltage_unique.min(), voltage_unique.max(), 1000)
    capacity_interpolator = interpolate.interp1d(voltage_unique, capacity_unique, kind='cubic')
    capacity_fine = capacity_interpolator(voltage_fine)

    # Calculate the derivative (dQ/dV)
    dq_dv = np.gradient(capacity_fine, voltage_fine)

    # Apply smoothing if requested
    if smooth and len(dq_dv) > window_size:
        dq_dv_smooth = signal.savgol_filter(dq_dv, window_size, 3)
        return voltage_fine, dq_dv_smooth
    else:
        return voltage_fine, dq_dv


def incremental_capacity_analysis(test_id, cycle_numbers=None, smooth=True):
    """Perform incremental capacity analysis (ICA) for selected cycles.

    Parameters
    ----------
    test_id : ObjectId | str
        ID of the ``TestResult`` to analyze.
    cycle_numbers : list[int] | None, optional
        Specific cycle numbers to analyse.  If ``None`` and the test
        contains more than four cycles, the routine analyses the first,
        a middle, and the last cycle.  Otherwise only the first and last
        cycles are used.
    smooth : bool, optional
        Whether to apply smoothing to the derivatives.

    Returns
    -------
    dict
        Dictionary mapping cycle numbers to ``(voltage, dQ/dV)`` tuples.
    """
    test = models.TestResult.objects(id=test_id).first()
    if not test:
        raise ValueError(f"Test with ID {test_id} not found")

    # Determine cycles to analyze
    if cycle_numbers is None:
        if len(test.cycles) > 1:
            cycle_numbers = [test.cycles[0].cycle_index, test.cycles[-1].cycle_index]
            if len(test.cycles) > 4:  # Add a middle cycle if enough cycles
                mid_idx = len(test.cycles) // 2
                cycle_numbers.append(test.cycles[mid_idx].cycle_index)
        else:
            cycle_numbers = [test.cycles[0].cycle_index]

    results = {}

    try:
        for cycle_num in cycle_numbers:
            voltage, capacity = get_voltage_capacity_data(test_id, cycle_num)
            v_centers, dq_dv = differential_capacity_analysis(voltage, capacity, smooth)
            results[cycle_num] = (v_centers, dq_dv)
    except Exception as e:
        raise ValueError(f"Error in incremental capacity analysis: {e}")

    return results


def voltage_efficiency_analysis(voltage_data, capacity_data, current_data=None):
    """
    Analyze voltage efficiency (hysteresis) between charge and discharge.

    Args:
        voltage_data: Array of voltage values
        capacity_data: Array of capacity values
        current_data: Optional array of current values

    Returns:
        dict: Dictionary with voltage efficiency metrics
    """
    # If current data is provided, use it to separate charge and discharge
    if current_data is not None:
        charge_mask = current_data > 0
        discharge_mask = current_data < 0
    else:
        # Without current data, try to infer from capacity pattern
        # Assuming capacity increases during charge and decreases during discharge
        capacity_diff = np.diff(capacity_data, prepend=capacity_data[0])
        charge_mask = capacity_diff > 0
        discharge_mask = capacity_diff < 0

    # Extract charge and discharge segments
    v_charge = voltage_data[charge_mask]
    cap_charge = capacity_data[charge_mask]
    v_discharge = voltage_data[discharge_mask]
    cap_discharge = capacity_data[discharge_mask]

    if len(v_charge) < 5 or len(v_discharge) < 5:
        raise ValueError("Not enough charge/discharge data points for analysis")

    # Calculate metrics
    v_charge_avg = np.mean(v_charge)
    v_discharge_avg = np.mean(v_discharge)
    v_hysteresis = v_charge_avg - v_discharge_avg

    # Calculate voltage efficiency
    v_efficiency = v_discharge_avg / v_charge_avg if v_charge_avg > 0 else 0

    # Create interpolation functions for charge and discharge curves
    # to find voltage at specific capacities
    try:
        cap_min = max(min(cap_charge), min(cap_discharge))
        cap_max = min(max(cap_charge), max(cap_discharge))

        cap_points = np.linspace(cap_min, cap_max, 100)

        # Interpolate charge and discharge curves
        f_charge = interpolate.interp1d(cap_charge, v_charge, bounds_error=False)
        f_discharge = interpolate.interp1d(cap_discharge, v_discharge, bounds_error=False)

        v_charge_interp = f_charge(cap_points)
        v_discharge_interp = f_discharge(cap_points)

        # Calculate average and max voltage gap
        valid_mask = ~(np.isnan(v_charge_interp) | np.isnan(v_discharge_interp))
        if sum(valid_mask) > 5:
            v_gaps = v_charge_interp[valid_mask] - v_discharge_interp[valid_mask]
            v_gap_avg = np.mean(v_gaps)
            v_gap_max = np.max(v_gaps)
        else:
            v_gap_avg = v_gap_max = None
    except Exception:
        v_gap_avg = v_gap_max = None

    return {
        'v_charge_avg': v_charge_avg,
        'v_discharge_avg': v_discharge_avg,
        'v_hysteresis': v_hysteresis,
        'v_efficiency': v_efficiency,
        'v_gap_avg': v_gap_avg,
        'v_gap_max': v_gap_max
    }


def capacity_fade_analysis(test_id):
    """
    Analyze capacity fade patterns and predict cycle life.

    Args:
        test_id: ID of the TestResult to analyze

    Returns:
        dict: Dictionary with capacity fade metrics and predictions
    """
    test = models.TestResult.objects(id=test_id).first()
    if not test:
        raise ValueError(f"Test with ID {test_id} not found")

    # Need at least 10 cycles for meaningful analysis
    if len(test.cycles) < 10:
        raise ValueError(f"Need at least 10 cycles for fade analysis, found {len(test.cycles)}")

    # Extract cycle numbers and discharge capacities
    cycle_nums = np.array([c.cycle_index for c in test.cycles])
    discharge_caps = np.array([c.discharge_capacity for c in test.cycles])

    # Calculate normalized capacity
    norm_caps = discharge_caps / discharge_caps[0]

    # Calculate fade rate (percent per cycle)
    fade_rate_pct = ((1 - norm_caps[-1]) / (cycle_nums[-1] - cycle_nums[0])) * 100

    # Try different fade models
    models_data = {}

    # 1. Linear fade model: cap = a*cycle + b
    try:
        linear_coef = np.polyfit(cycle_nums, discharge_caps, 1)
        linear_a, linear_b = linear_coef
        linear_fit = linear_a * cycle_nums + linear_b
        linear_r2 = np.corrcoef(discharge_caps, linear_fit)[0, 1] ** 2

        # Predict cycle where capacity reaches 80% of initial
        if linear_a < 0:  # Negative slope (capacity decreasing)
            eol_threshold = 0.8 * discharge_caps[0]
            eol_cycle_linear = (eol_threshold - linear_b) / linear_a
            eol_cycle_linear = max(cycle_nums[-1], eol_cycle_linear)  # Don't predict earlier than last measured cycle
        else:
            eol_cycle_linear = None

        models_data['linear'] = {
            'name': 'Linear',
            'params': {'slope': linear_a, 'intercept': linear_b},
            'r_squared': linear_r2,
            'eol_cycle': eol_cycle_linear
        }
    except Exception:
        pass

    # 2. Power law fade model: cap = a * cycle^b + c
    try:
        # Initial guess for optimization
        p0 = [-0.1, 0.5, discharge_caps[0]]

        def power_law(x, a, b, c):
            return a * np.power(x, b) + c

        params, _ = optimize.curve_fit(power_law, cycle_nums, discharge_caps, p0=p0, maxfev=10000)
        power_a, power_b, power_c = params

        power_fit = power_law(cycle_nums, power_a, power_b, power_c)
        power_r2 = 1 - (
                    np.sum((discharge_caps - power_fit) ** 2) / np.sum((discharge_caps - np.mean(discharge_caps)) ** 2))

        # Predict cycle where capacity reaches 80% of initial
        eol_threshold = 0.8 * discharge_caps[0]

        try:
            # Define a function to find the root (where capacity = eol_threshold)
            def find_eol(x):
                return power_law(x, power_a, power_b, power_c) - eol_threshold

            if power_a < 0:  # Ensure capacity is decreasing
                last_cycle = cycle_nums[-1]
                if find_eol(last_cycle) > 0:  # EOL hasn't been reached yet
                    eol_cycle_power = optimize.brentq(find_eol, last_cycle, last_cycle * 10)
                else:
                    eol_cycle_power = None
            else:
                eol_cycle_power = None
        except Exception:
            eol_cycle_power = None

        models_data['power'] = {
            'name': 'Power Law',
            'params': {'a': power_a, 'b': power_b, 'c': power_c},
            'r_squared': power_r2,
            'eol_cycle': eol_cycle_power
        }
    except Exception:
        pass

    # 3. Exponential fade model: cap = a * exp(b * cycle) + c
    try:
        # Initial guess for optimization
        p0 = [-0.2, -0.001, discharge_caps[0]]

        def exp_decay(x, a, b, c):
            return a * np.exp(b * x) + c

        params, _ = optimize.curve_fit(exp_decay, cycle_nums, discharge_caps, p0=p0, maxfev=10000)
        exp_a, exp_b, exp_c = params

        exp_fit = exp_decay(cycle_nums, exp_a, exp_b, exp_c)
        exp_r2 = 1 - (np.sum((discharge_caps - exp_fit) ** 2) / np.sum((discharge_caps - np.mean(discharge_caps)) ** 2))

        # Predict cycle where capacity reaches 80% of initial
        eol_threshold = 0.8 * discharge_caps[0]

        try:
            # Define a function to find the root (where capacity = eol_threshold)
            def find_eol(x):
                return exp_decay(x, exp_a, exp_b, exp_c) - eol_threshold

            if exp_b < 0:  # Ensure capacity is decreasing (negative exponent)
                last_cycle = cycle_nums[-1]
                if find_eol(last_cycle) > 0:  # EOL hasn't been reached yet
                    eol_cycle_exp = optimize.brentq(find_eol, last_cycle, last_cycle * 10)
                else:
                    eol_cycle_exp = None
            else:
                eol_cycle_exp = None
        except Exception:
            eol_cycle_exp = None

        models_data['exponential'] = {
            'name': 'Exponential',
            'params': {'a': exp_a, 'b': exp_b, 'c': exp_c},
            'r_squared': exp_r2,
            'eol_cycle': eol_cycle_exp
        }
    except Exception:
        pass

    # Determine best model based on R²
    best_model = None
    best_r2 = -1

    for model_name, model_info in models_data.items():
        if model_info['r_squared'] > best_r2:
            best_r2 = model_info['r_squared']
            best_model = model_name

    # Calculate confidence in prediction based on:
    # 1. Number of cycles
    # 2. R² of the best model
    # 3. Stability of recent fade rate

    confidence = None
    if best_model:
        # Base confidence on R²
        r2_factor = min(models_data[best_model]['r_squared'], 0.99)

        # Adjust based on number of cycles (more cycles = higher confidence)
        cycle_factor = min(len(test.cycles) / 100, 1.0)

        # Calculate stability of recent fade rate (using last 20% of cycles)
        if len(test.cycles) >= 10:
            n_recent = max(int(len(test.cycles) * 0.2), 5)
            recent_cycles = cycle_nums[-n_recent:]
            recent_caps = discharge_caps[-n_recent:]
            recent_slope, _, _, _, _ = stats.linregress(recent_cycles, recent_caps)
            overall_slope = (discharge_caps[-1] - discharge_caps[0]) / (cycle_nums[-1] - cycle_nums[0])

            # If recent slope is similar to overall slope, higher confidence
            slope_ratio = abs(recent_slope / overall_slope) if overall_slope != 0 else 0
            stability_factor = 1.0 - min(abs(1.0 - slope_ratio), 0.5)
        else:
            stability_factor = 0.5  # Default for short tests

        # Combine factors
        confidence = (r2_factor * 0.5) + (cycle_factor * 0.3) + (stability_factor * 0.2)

    # Return comprehensive analysis
    return {
        'cycle_count': len(test.cycles),
        'initial_capacity': discharge_caps[0],
        'final_capacity': discharge_caps[-1],
        'capacity_retention': norm_caps[-1],
        'fade_rate_pct_per_cycle': fade_rate_pct,
        'fade_models': models_data,
        'best_model': best_model,
        'confidence': confidence,
        'predicted_eol_cycle': models_data.get(best_model, {}).get('eol_cycle') if best_model else None
    }


def analyze_rate_capability(sample_id):
    """
    Analyze rate capability by comparing tests at different C-rates.

    Args:
        sample_id: ID of the Sample with multiple tests at different rates

    Returns:
        dict: Dictionary with rate capability metrics
    """
    sample = models.Sample.objects(id=sample_id).first()
    if not sample:
        raise ValueError(f"Sample with ID {sample_id} not found")

    # Need at least 2 tests for comparison
    if len(sample.tests) < 2:
        raise ValueError(f"Need at least 2 tests for rate analysis, found {len(sample.tests)}")

    # Get all tests for the sample
    tests = []
    for test_ref in sample.tests:
        test = models.TestResult.objects(id=test_ref.id).first()
        if test:
            tests.append(test)

    # Extract C-rates from test names or metadata
    test_rates = []
    for test in tests:
        rate = None

        # Try to get C-rate from metadata
        if hasattr(test, 'discharge_rate') and test.discharge_rate is not None:
            rate = test.discharge_rate

        # If not found, try to extract from test name
        if rate is None and test.name:
            extracted_rate = utils.extract_crate(test.name)
            if extracted_rate:
                rate = extracted_rate

        # If still not found, try file name
        if rate is None and hasattr(test, 'file_path') and test.file_path:
            extracted_rate = utils.extract_crate(os.path.basename(test.file_path))
            if extracted_rate:
                rate = extracted_rate

        if rate is not None:
            test_rates.append((test, rate))

    if len(test_rates) < 2:
        raise ValueError("Could not determine C-rates for at least 2 tests")

    # Sort tests by C-rate
    test_rates.sort(key=lambda x: x[1])

    # Analyze capacity vs. rate
    rates = []
    capacities = []
    retentions = []

    # Use lowest rate test as reference
    reference_test, reference_rate = test_rates[0]
    reference_capacity = reference_test.initial_capacity

    rate_capability_data = []

    for test, rate in test_rates:
        rates.append(rate)

        # Use initial capacity (first cycle) for rate comparison
        capacity = test.initial_capacity
        capacities.append(capacity)

        # Calculate retention relative to lowest rate
        retention = capacity / reference_capacity if reference_capacity else 0
        retentions.append(retention)

        rate_capability_data.append({
            'test_id': str(test.id),
            'test_name': test.name,
            'c_rate': rate,
            'capacity': capacity,
            'relative_capacity': retention
        })

    # Fit curve to capacity vs. rate data
    try:
        # Use logarithmic model for rate vs. capacity: cap = a * log(rate) + b
        log_rates = np.log(rates)
        params = np.polyfit(log_rates, capacities, 1)
        a, b = params

        # Calculate R²
        fit_capacities = a * log_rates + b
        ss_total = np.sum((capacities - np.mean(capacities)) ** 2)
        ss_residual = np.sum((capacities - fit_capacities) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        model = {
            'type': 'logarithmic',
            'params': {'a': float(a), 'b': float(b)},
            'r_squared': float(r_squared)
        }
    except Exception:
        model = None

    # Calculate additional metrics
    metrics = {
        'reference_rate': float(reference_rate),
        'reference_capacity': float(reference_capacity),
        'peukert_coefficient': calculate_peukert_coefficient(rates, capacities) if len(rates) >= 3 else None
    }

    return {
        'sample_name': sample.name,
        'test_count': len(test_rates),
        'rate_data': rate_capability_data,
        'model': model,
        'metrics': metrics
    }


def calculate_peukert_coefficient(rates, capacities):
    """
    Calculate Peukert coefficient for rate capability assessment.

    The Peukert equation relates discharge rate to capacity: C = I^k * t
    where k is the Peukert coefficient.

    Args:
        rates: List of C-rates
        capacities: List of corresponding capacities

    Returns:
        float: Peukert coefficient
    """
    try:
        # Need at least 3 points for reliable fit
        if len(rates) < 3:
            return None

        # For Peukert analysis, we need to work with actual current values
        # and discharge times, but we can approximate using C-rates and capacities

        # For Li-ion cells, k is typically 1.01-1.2
        # k = 1.0 means capacity is independent of rate
        # k > 1.0 means capacity decreases as rate increases

        # Take log of both rates and capacities
        log_rates = np.log(rates)
        log_capacities = np.log(capacities)

        # Fit a line to log-log data
        slope, _, _, _, _ = stats.linregress(log_rates, log_capacities)

        # Peukert coefficient is related to this slope
        # For ideal Peukert behavior: log(C) = -k * log(I) + log(C_0)
        # So our coefficient is approximately -slope
        peukert_k = max(1.0, -slope)

        return float(peukert_k)
    except Exception:
        return None


def coulombic_efficiency_analysis(test_id):
    """
    Analyze coulombic efficiency trends and patterns.

    Args:
        test_id: ID of the TestResult to analyze

    Returns:
        dict: Dictionary with CE analysis metrics
    """
    test = models.TestResult.objects(id=test_id).first()
    if not test:
        raise ValueError(f"Test with ID {test_id} not found")

    # Need at least 5 cycles for meaningful analysis
    if len(test.cycles) < 5:
        raise ValueError(f"Need at least 5 cycles for CE analysis, found {len(test.cycles)}")

    # Extract cycle numbers and coulombic efficiencies
    cycle_nums = np.array([c.cycle_index for c in test.cycles])
    ce_values = np.array([c.coulombic_efficiency for c in test.cycles])

    # Calculate basic statistics
    avg_ce = np.mean(ce_values)
    std_ce = np.std(ce_values)
    min_ce = np.min(ce_values)
    max_ce = np.max(ce_values)

    # Calculate first cycle irreversible capacity loss
    first_cycle_ce = ce_values[0]
    first_cycle_loss = 1.0 - first_cycle_ce

    # Calculate CE stability (standard deviation of CE after first 5 cycles)
    if len(ce_values) > 5:
        stabilized_ce = ce_values[5:]
        ce_stability = np.std(stabilized_ce)
    else:
        ce_stability = std_ce

    # Check for CE trends over cycles
    try:
        # Fit linear trend to CE vs cycle
        slope, intercept, r_value, p_value, std_err = stats.linregress(cycle_nums, ce_values)

        # Determine if there's a significant trend
        has_trend = False
        trend_direction = "stable"

        if abs(r_value) > 0.5 and p_value < 0.05:
            has_trend = True
            if slope > 0:
                trend_direction = "improving"
            elif slope < 0:
                trend_direction = "deteriorating"
    except Exception:
        slope = intercept = r_value = p_value = std_err = None
        has_trend = False
        trend_direction = "unknown"

    # Calculate CE deficit from 100%
    avg_ce_deficit = 1.0 - avg_ce

    # Project capacity retention based on CE
    # For n cycles, retention ≈ (CE)^n
    projected_retention_100 = avg_ce ** 100
    projected_retention_500 = avg_ce ** 500
    projected_retention_1000 = avg_ce ** 1000

    return {
        'cycle_count': len(test.cycles),
        'average_ce': float(avg_ce),
        'std_deviation': float(std_ce),
        'min_ce': float(min_ce),
        'max_ce': float(max_ce),
        'first_cycle_ce': float(first_cycle_ce),
        'first_cycle_loss': float(first_cycle_loss),
        'ce_stability': float(ce_stability),
        'trend': {
            'has_trend': has_trend,
            'direction': trend_direction,
            'slope': float(slope) if slope is not None else None,
            'r_squared': float(r_value ** 2) if r_value is not None else None,
            'p_value': float(p_value) if p_value is not None else None
        },
        'ce_deficit': float(avg_ce_deficit),
        'projected_retention': {
            '100_cycles': float(projected_retention_100),
            '500_cycles': float(projected_retention_500),
            '1000_cycles': float(projected_retention_1000)
        }
    }


def energy_analysis(test_id):
    """
    Analyze energy efficiency and energy density for a test.

    Args:
        test_id: ID of the TestResult to analyze

    Returns:
        dict: Dictionary with energy analysis metrics
    """
    test = models.TestResult.objects(id=test_id).first()
    if not test:
        raise ValueError(f"Test with ID {test_id} not found")

    # Need energy data for analysis
    if len(test.cycles) == 0:
        raise ValueError("No cycle data available for energy analysis")

    # Extract energy values if available
    discharge_energy = []
    energy_efficiency = []

    for cycle in test.cycles:
        if hasattr(cycle, 'discharge_energy') and cycle.discharge_energy is not None:
            discharge_energy.append(cycle.discharge_energy)
        else:
            # Estimate from capacity if energy not directly available
            # Assuming average voltage of 3.7V for Li-ion
            discharge_energy.append(cycle.discharge_capacity * 3.7 / 1000)  # Convert to Wh

        if hasattr(cycle, 'energy_efficiency') and cycle.energy_efficiency is not None:
            energy_efficiency.append(cycle.energy_efficiency)
        else:
            # Estimate from coulombic efficiency if energy efficiency not available
            # Energy efficiency is typically a bit lower than coulombic efficiency
            energy_efficiency.append(cycle.coulombic_efficiency * 0.95)  # 95% of CE

    # Calculate energy metrics
    if discharge_energy:
        initial_discharge_energy = discharge_energy[0]
        final_discharge_energy = discharge_energy[-1]
        energy_retention = final_discharge_energy / initial_discharge_energy if initial_discharge_energy > 0 else None

        # Calculate energy fade rate (%/cycle) if we have multiple cycles
        if len(test.cycles) > 1 and energy_retention is not None:
            cycle_count = test.cycles[-1].cycle_index - test.cycles[0].cycle_index
            energy_fade_rate = ((1 - energy_retention) / cycle_count) * 100
        else:
            energy_fade_rate = None
    else:
        initial_discharge_energy = final_discharge_energy = energy_retention = energy_fade_rate = None

    # Calculate average energy efficiency
    avg_energy_efficiency = np.mean(energy_efficiency) if energy_efficiency else None

    # Calculate energy density if we have sample mass or volume
    energy_density = {}
    sample = test.sample

    if hasattr(sample, 'mass') and sample.mass and sample.mass > 0:
        gravimetric_energy_density = initial_discharge_energy / (
                    sample.mass / 1000) if initial_discharge_energy else None
        energy_density['gravimetric'] = gravimetric_energy_density

    if hasattr(sample, 'volume') and sample.volume and sample.volume > 0:
        volumetric_energy_density = initial_discharge_energy / (
                    sample.volume / 1000) if initial_discharge_energy else None
        energy_density['volumetric'] = volumetric_energy_density

    # Build result dictionary
    result = {
        'initial_discharge_energy': initial_discharge_energy,
        'final_discharge_energy': final_discharge_energy,
        'energy_retention': energy_retention,
        'energy_fade_rate_pct_per_cycle': energy_fade_rate,
        'avg_energy_efficiency': avg_energy_efficiency,
        'energy_density': energy_density
    }

    # If we have charge energy data, add it
    charge_energy = []
    for cycle in test.cycles:
        if hasattr(cycle, 'charge_energy') and cycle.charge_energy is not None:
            charge_energy.append(cycle.charge_energy)

    if charge_energy:
        result['initial_charge_energy'] = charge_energy[0]
        result['final_charge_energy'] = charge_energy[-1]
        result['avg_charge_energy'] = np.mean(charge_energy)
        result['avg_discharge_energy'] = np.mean(discharge_energy)

    return result


def detect_anomalies(test_id, metric='discharge_capacity', n_sigma=3.0):
    """
    Detect anomalous cycles in a test result.

    Args:
        test_id: ID of the TestResult to analyze
        metric: Metric to analyze for anomalies ('discharge_capacity', 'coulombic_efficiency', etc.)
        n_sigma: Number of standard deviations to use for threshold (default: 3.0)

    Returns:
        dict: Dictionary with anomaly detection results
    """
    test = models.TestResult.objects(id=test_id).first()
    if not test:
        raise ValueError(f"Test with ID {test_id} not found")

    # Extract cycle data
    cycle_nums = [c.cycle_index for c in test.cycles]

    # Get metric values
    if metric == 'discharge_capacity':
        values = [c.discharge_capacity for c in test.cycles]
    elif metric == 'charge_capacity':
        values = [c.charge_capacity for c in test.cycles]
    elif metric == 'coulombic_efficiency':
        values = [c.coulombic_efficiency for c in test.cycles]
    elif hasattr(test.cycles[0], metric):
        values = [getattr(c, metric) for c in test.cycles if hasattr(c, metric)]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Make numpy array for calculations
    values_array = np.array(values)

    # Calculate values for anomaly detection

    # 1. Z-score method (how many std devs from the mean)
    mean = np.mean(values_array)
    std = np.std(values_array)
    z_scores = np.abs((values_array - mean) / std)

    # Find anomalies - values that are more than n_sigma std devs from the mean
    anomalies_zscore = []

    for i, (cycle, value, z) in enumerate(zip(cycle_nums, values, z_scores)):
        if z > n_sigma:
            anomalies_zscore.append({
                'cycle': cycle,
                'value': value,
                'z_score': z,
                'index': i
            })

    # 2. Moving average method (detect sudden jumps)
    window_size = min(5, len(values) // 10 + 1)  # Adaptive window size
    moving_avg = np.convolve(values_array, np.ones(window_size) / window_size, mode='valid')

    # Calculate % difference from moving average
    diffs = []
    for i in range(len(values) - window_size + 1):
        actual = values[i + window_size - 1]
        expected = moving_avg[i]
        if expected != 0:
            diff = (actual - expected) / expected * 100
        else:
            diff = 0
        diffs.append(diff)

    # Find points where difference exceeds threshold
    anomalies_mavg = []
    diff_threshold = 15  # 15% difference from moving average

    for i, diff in enumerate(diffs):
        if abs(diff) > diff_threshold:
            original_idx = i + window_size - 1
            anomalies_mavg.append({
                'cycle': cycle_nums[original_idx],
                'value': values[original_idx],
                'difference_pct': diff,
                'index': original_idx
            })

    # Combine anomalies from both methods (avoiding duplicates)
    combined_anomaly_indices = set()
    anomalies_combined = []

    for anomaly in anomalies_zscore:
        combined_anomaly_indices.add(anomaly['index'])
        anomalies_combined.append({
            'cycle': anomaly['cycle'],
            'value': anomaly['value'],
            'detection_method': 'z-score',
            'significance': anomaly['z_score']
        })

    for anomaly in anomalies_mavg:
        if anomaly['index'] not in combined_anomaly_indices:
            combined_anomaly_indices.add(anomaly['index'])
            anomalies_combined.append({
                'cycle': anomaly['cycle'],
                'value': anomaly['value'],
                'detection_method': 'moving_average',
                'significance': abs(anomaly['difference_pct'])
            })

    # Return results
    return {
        'test_id': str(test_id),
        'metric_analyzed': metric,
        'normal_range': {
            'mean': float(mean),
            'std_dev': float(std),
            'threshold': float(n_sigma * std)
        },
        'anomaly_count': len(anomalies_combined),
        'anomalies': sorted(anomalies_combined, key=lambda x: x['cycle'])
    }


def cluster_tests(test_ids, metrics=None, method='hierarchical', n_clusters=None):
    """
    Cluster tests based on their performance metrics.

    Args:
        test_ids: List of TestResult IDs to include in the clustering
        metrics: List of metrics to use for clustering (default: same as PCA)
        method: Clustering method ('hierarchical' or 'kmeans')
        n_clusters: Number of clusters (default: automatically determined)

    Returns:
        dict: Clustering results including cluster assignments for each test
    """
    if metrics is None:
        metrics = ['initial_capacity', 'final_capacity', 'capacity_retention',
                   'avg_coulombic_eff', 'cycle_count']

    # First perform PCA to reduce dimensionality
    pca_result = pca_analysis(test_ids, metrics)

    # Get principal components
    principal_components = pca_result.get('principal_components', [])

    # If no PCA components in results, create artificial x,y coordinates
    if not principal_components or len(principal_components) < 1:
        # Create a simple scatter based on cluster assignments
        x_coords = []
        y_coords = []

        for i, test in enumerate(pca_result['tests']):
            x_coords.append(i % 5)  # Arbitrary distribution
            y_coords.append(i // 5)

        principal_components = list(zip(x_coords, y_coords))

    # Get test info
    test_info = pca_result['tests']

    # Determine number of clusters if not specified
    if n_clusters is None:
        # Use a reasonable default based on dataset size
        n_clusters = max(2, min(5, len(test_ids) // 3))

    # Perform clustering
    if method == 'hierarchical':
        # Compute distance matrix
        dist_matrix = hierarchy.distance.pdist(principal_components)
        Z = hierarchy.linkage(dist_matrix, 'ward')

        # Get cluster assignments
        cluster_labels = hierarchy.fcluster(Z, n_clusters, criterion='maxclust')
    else:  # kmeans
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(principal_components)

    # Assign cluster labels to test info
    for i, label in enumerate(cluster_labels):
        if i < len(test_info):
            test_info[i]['cluster'] = int(label)

    # Group tests by cluster
    clusters = {}
    for i in range(1, n_clusters + 1):
        clusters[str(i)] = [test for test in test_info if test.get('cluster') == i]

    # Create result
    result = {
        'method': method,
        'n_clusters': n_clusters,
        'metrics_used': metrics,
        'clusters': clusters,
        'tests': test_info,
        'principal_components': principal_components
    }

    return result


def pca_analysis(test_ids, metrics=None):
    """
    Perform Principal Component Analysis on multiple tests to identify patterns.

    Args:
        test_ids: List of TestResult IDs to include in the analysis
        metrics: List of metrics to include (default: capacity, CE, capacity fade, cycle count)

    Returns:
        dict: PCA results including principal components and explained variance
    """
    if metrics is None:
        metrics = ['initial_capacity', 'final_capacity', 'capacity_retention',
                   'avg_coulombic_eff', 'cycle_count']

    # Get test results
    tests = []
    for test_id in test_ids:
        test = models.TestResult.objects(id=test_id).first()
        if test:
            tests.append(test)

    if len(tests) < 2:
        raise ValueError("Need at least 2 tests for PCA analysis")

    # Extract features for each test
    features = []
    test_info = []

    for test in tests:
        # Extract requested metrics
        test_features = []

        for metric in metrics:
            if hasattr(test, metric):
                value = getattr(test, metric)
                if value is not None:
                    test_features.append(float(value))
                else:
                    test_features.append(0.0)  # Replace None with 0
            else:
                test_features.append(0.0)  # Metric not available

        features.append(test_features)
        test_info.append({
            'test_id': str(test.id),
            'test_name': test.name,
            'sample_name': utils.get_sample_name(test.sample)
        })

    # Convert to numpy array
    X = np.array(features)

    # Standardize the features
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Apply PCA
    n_components = min(len(metrics), len(tests) - 1)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_std)

    # Create result
    result = {
        'metrics': metrics,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'component_matrix': pca.components_.tolist(),
        'principal_components': principal_components.tolist(),
        'tests': test_info
    }

    return result


def find_similar_tests(test_id, metrics=None, max_results=5):
    """
    Find tests similar to a reference test based on performance metrics.

    Args:
        test_id: ID of the reference TestResult
        metrics: List of metrics to consider (default: same as PCA)
        max_results: Maximum number of similar tests to return

    Returns:
        list: Ranked list of similar tests with similarity scores
    """
    if metrics is None:
        metrics = ['initial_capacity', 'final_capacity', 'capacity_retention',
                   'avg_coulombic_eff', 'cycle_count']

    # Get reference test
    reference_test = models.TestResult.objects(id=test_id).first()
    if not reference_test:
        raise ValueError(f"Test with ID {test_id} not found")

    # Get all tests (excluding reference)
    all_tests = list(models.TestResult.objects().all())

    # Create feature vectors
    reference_features = []
    for metric in metrics:
        if hasattr(reference_test, metric):
            value = getattr(reference_test, metric)
            if value is not None:
                reference_features.append(float(value))
            else:
                reference_features.append(0.0)
        else:
            reference_features.append(0.0)

    test_data = []
    for test in all_tests:
        if str(test.id) != str(test_id):  # Skip reference test
            test_features = []
            for metric in metrics:
                if hasattr(test, metric):
                    value = getattr(test, metric)
                    if value is not None:
                        test_features.append(float(value))
                    else:
                        test_features.append(0.0)
                else:
                    test_features.append(0.0)

            test_data.append((test, test_features))

    # Convert to numpy arrays
    reference_features = np.array(reference_features)

    # Calculate similarities
    similarities = []
    for test, features in test_data:
        features = np.array(features)

        # Normalize feature vectors
        norm_ref = np.linalg.norm(reference_features)
        norm_test = np.linalg.norm(features)

        if norm_ref > 0 and norm_test > 0:
            # Calculate cosine similarity
            cosine_sim = np.dot(reference_features, features) / (norm_ref * norm_test)

            # Calculate Euclidean distance (normalized to [0,1])
            euclidean_dist = np.linalg.norm(reference_features - features)
            max_dist = np.linalg.norm(np.ones_like(reference_features))
            euclidean_sim = 1 - (euclidean_dist / max_dist)

            # Combine similarities (weighted average)
            similarity = 0.7 * cosine_sim + 0.3 * euclidean_sim

            similarities.append({
                'test_id': str(test.id),
                'test_name': test.name,
                'sample_name': utils.get_sample_name(test.sample),
                'similarity': float(similarity),
                'cosine_similarity': float(cosine_sim),
                'euclidean_similarity': float(euclidean_sim)
            })

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x['similarity'], reverse=True)

    # Return top matches
    return similarities[:max_results]


def get_cycle_voltage_capacity_data(test_id, cycle_number):
    """
    Get detailed voltage vs capacity data for a specific cycle.

    Args:
        test_id: ID of the TestResult to analyze
        cycle_number: Specific cycle to extract

    Returns:
        dict: Dictionary with charge and discharge data
    """
    # Get TestResult object
    test = models.TestResult.objects(id=test_id).first()
    if not test:
        raise ValueError(f"Test with ID {test_id} not found")

    # Find the requested cycle
    cycle = None
    for c in test.cycles:
        if c.cycle_index == cycle_number:
            cycle = c
            break

    if not cycle:
        raise ValueError(f"Cycle {cycle_number} not found in test {test_id}")

    # Check if detailed data is available
    has_detailed_data = (
            hasattr(cycle, 'voltage_charge') and
            hasattr(cycle, 'capacity_charge') and
            hasattr(cycle, 'voltage_discharge') and
            hasattr(cycle, 'capacity_discharge')
    )

    if not has_detailed_data or not cycle.voltage_charge:
        raise ValueError(f"Detailed data not available for cycle {cycle_number}")

    # Return the detailed data
    return {
        'charge': {
            'voltage': cycle.voltage_charge,
            'current': cycle.current_charge if hasattr(cycle, 'current_charge') else [],
            'capacity': cycle.capacity_charge,
            'time': cycle.time_charge if hasattr(cycle, 'time_charge') else []
        },
        'discharge': {
            'voltage': cycle.voltage_discharge,
            'current': cycle.current_discharge if hasattr(cycle, 'current_discharge') else [],
            'capacity': cycle.capacity_discharge,
            'time': cycle.time_discharge if hasattr(cycle, 'time_discharge') else []
        }
    }


def plot_cycle_voltage_capacity(test_id, cycle_number, chemistry=None):
    """
    Plot voltage vs. capacity for a specific cycle.

    Args:
        test_id: ID of the TestResult to analyze
        cycle_number: Specific cycle to plot

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    import matplotlib.pyplot as plt

    # Get the cycle data
    cycle_data = get_cycle_voltage_capacity_data(test_id, cycle_number)

    # Determine label based on chemistry
    if chemistry:
        label = pybamm_models.chemistry_to_label(chemistry)
    else:
        label = "Cell"

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot charge data
    if cycle_data['charge']['voltage'] and cycle_data['charge']['capacity']:
        ax.plot(
            cycle_data['charge']['capacity'],
            cycle_data['charge']['voltage'],
            'b-',
            label=label
        )

    # Plot discharge data
    if cycle_data['discharge']['voltage'] and cycle_data['discharge']['capacity']:
        ax.plot(
            cycle_data['discharge']['capacity'],
            cycle_data['discharge']['voltage'],
            'r-',
            label=None
        )

    # Set labels and title
    ax.set_xlabel('Capacity (mAh)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title(f'Voltage vs. Capacity - Cycle {cycle_number}')
    ax.grid(True)
    ax.legend()

    return fig


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
    import numpy as np
    from scipy.signal import savgol_filter

    # Get the cycle data
    cycle_data = get_cycle_voltage_capacity_data(test_id, cycle_number)

    result = {'charge': {}, 'discharge': {}}

    # Process charge data
    if cycle_data['charge']['voltage'] and cycle_data['charge']['capacity']:
        v = np.array(cycle_data['charge']['voltage'])
        q = np.array(cycle_data['charge']['capacity'])

        # Sort by voltage (increasing)
        sort_idx = np.argsort(v)
        v_sorted = v[sort_idx]
        q_sorted = q[sort_idx]

        # Remove duplicates in voltage
        unique_idx = np.concatenate(([True], np.diff(v_sorted) > 1e-5))
        v_unique = v_sorted[unique_idx]
        q_unique = q_sorted[unique_idx]

        # Calculate dQ/dV
        if len(v_unique) > 3:
            dq = np.diff(q_unique)
            dv = np.diff(v_unique)
            dqdv = dq / dv
            v_centers = (v_unique[1:] + v_unique[:-1]) / 2

            # Apply smoothing if requested
            if smoothing and len(dqdv) > 10:
                window = min(11, len(dqdv) - 2 if len(dqdv) % 2 == 0 else len(dqdv) - 1)
                window = max(3, window - 1 if window % 2 == 0 else window)
                dqdv_smooth = savgol_filter(dqdv, window, 1)
                result['charge'] = {
                    'voltage': v_centers.tolist(),
                    'dqdv': dqdv_smooth.tolist()
                }
            else:
                result['charge'] = {
                    'voltage': v_centers.tolist(),
                    'dqdv': dqdv.tolist()
                }

    # Process discharge data
    if cycle_data['discharge']['voltage'] and cycle_data['discharge']['capacity']:
        v = np.array(cycle_data['discharge']['voltage'])
        q = np.array(cycle_data['discharge']['capacity'])

        # Sort by voltage (decreasing for discharge)
        sort_idx = np.argsort(v)[::-1]
        v_sorted = v[sort_idx]
        q_sorted = q[sort_idx]

        # Remove duplicates in voltage
        unique_idx = np.concatenate(([True], np.diff(v_sorted) < -1e-5))
        v_unique = v_sorted[unique_idx]
        q_unique = q_sorted[unique_idx]

        # Calculate dQ/dV
        if len(v_unique) > 3:
            dq = np.diff(q_unique)
            dv = np.diff(v_unique)
            dqdv = dq / dv
            v_centers = (v_unique[1:] + v_unique[:-1]) / 2

            # Apply smoothing if requested
            if smoothing and len(dqdv) > 10:
                window = min(11, len(dqdv) - 2 if len(dqdv) % 2 == 0 else len(dqdv) - 1)
                window = max(3, window - 1 if window % 2 == 0 else window)
                dqdv_smooth = savgol_filter(dqdv, window, 1)
                result['discharge'] = {
                    'voltage': v_centers.tolist(),
                    'dqdv': dqdv_smooth.tolist()
                }
            else:
                result['discharge'] = {
                    'voltage': v_centers.tolist(),
                    'dqdv': dqdv.tolist()
                }

    return result




def compute_dqdv(capacity, voltage, *, smooth=True, window_size=11, polyorder=3):
    """
    Return (voltage_midpoints, dQ/dV) for a single charge *or* discharge curve.

    Parameters
    ----------
    capacity : array-like
        Capacity values (mAh or mAh g-1).  Any iterable accepted.
    voltage : array-like
        Corresponding cell voltages (V).
    smooth : bool, default True
        Apply Savitzky–Golay smoothing to the raw derivative.
    window_size : int, default 11
        S-G window width; must be odd and <= len(dQ/dV).
    polyorder : int, default 3
        Polynomial order for Savitzky–Golay filter.

    Returns
    -------
    v_mid : np.ndarray
        Mid-point voltages (length = len(voltage) - 1).
    dqdv : np.ndarray
        dQ/dV values (same length as *v_mid*).

    Raises
    ------
    ValueError
        If fewer than three valid (non-NaN) points remain.
    """
    from scipy.signal import savgol_filter
    # -- 1. to numpy + drop NaNs with a single mask -------------------------
    capacity = np.asarray(capacity, dtype=float)
    voltage  = np.asarray(voltage, dtype=float)

    mask = (~np.isnan(capacity)) & (~np.isnan(voltage))
    capacity, voltage = capacity[mask], voltage[mask]

    if capacity.size < 3:
        raise ValueError("Not enough valid points to compute dQ/dV")

    # -- 2. ensure increasing capacity so dq > 0 ----------------------------
    if not np.all(np.diff(capacity) > 0):
        order = np.argsort(capacity)
        capacity, voltage = capacity[order], voltage[order]

    # -- 3. raw derivative --------------------------------------------------
    dq = np.diff(capacity)
    dv = np.diff(voltage)
    with np.errstate(divide="ignore", invalid="ignore"):
        dqdv = dq / dv

    v_mid = (voltage[:-1] + voltage[1:]) / 2  # align derivative to mid-points

    # -- 4. optional Savitzky–Golay smoothing -------------------------------
    if smooth and dqdv.size >= 5:  # need at least 5 pts for SG to make sense
        if window_size % 2 == 0:
            window_size += 1
        if window_size > dqdv.size:
            window_size = dqdv.size if dqdv.size % 2 else dqdv.size - 1
        if window_size >= polyorder + 2:       # SG requirement
            dqdv = savgol_filter(dqdv, window_size, polyorder)

    return v_mid, dqdv


# Append new function for Josh_request_Dq_dv

def josh_request_dq_dv(file_path: str, sheet_name: str = "Channel51_1", mass_g: float = 0.0015) -> pd.DataFrame:
    """Return smoothed dQ/dV DataFrame for the first cycle of *sheet_name*.

    Parameters
    ----------
    file_path : str
        Path to the Excel file containing the data.
    sheet_name : str, default "Channel51_1"
        Name of the sheet to parse.
    mass_g : float, default 0.0015
        Active material mass in grams used to normalise the capacity.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``"V"`` and ``"dQdV_sm"`` containing the
        voltage and smoothed differential capacity.
    """
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=20)
    hdr = next(
        i for i, row in raw.iterrows() if row.astype(str).str.contains("Cycle Index").any()
    )
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=hdr)
    df.columns = df.columns.str.strip()
    df1 = df[df["Cycle Index"] == 1].copy()

    q = df1["Charge Capacity (Ah)"].fillna(0).to_numpy() * 1e3 / mass_g
    v = df1["Voltage (V)"].to_numpy()

    mask = np.concatenate(([True], np.diff(q) > 0))
    v_charge, q_charge = v[mask], q[mask]

    dq_dv = np.gradient(q_charge, v_charge)
    dfdq = pd.DataFrame({"V": v_charge, "dQdV": dq_dv})
    dfdq["dQdV_sm"] = dfdq["dQdV"].rolling(51, center=True, min_periods=1).mean()
    dfdq["dQdV_sm"] = np.clip(dfdq["dQdV_sm"], 0, None)
    return dfdq[["V", "dQdV_sm"]]

