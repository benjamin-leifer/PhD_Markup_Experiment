"""
Electrochemical Impedance Spectroscopy (EIS) module for battery analysis.

This module provides functions for importing, analyzing, fitting and visualizing
EIS data from battery tests, leveraging the impedance.py package for equivalent
circuit modeling and fitting.

Required dependencies:
    - impedance.py (pip install impedance)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import warnings

try:
    from . import models, utils
except ImportError:  # pragma: no cover - allow running as script
    import importlib

    models = importlib.import_module("models")
    utils = importlib.import_module("utils")

# Import impedance.py for EIS analysis
try:
    import impedance
    from impedance.models.circuits import CustomCircuit
    from impedance.visualization import plot_nyquist, plot_bode
    from impedance.preprocessing import ignoreBelowX  # noqa: F401

    HAS_IMPEDANCE = True
except ImportError:
    HAS_IMPEDANCE = False
    warnings.warn(
        "impedance.py package not found. Install with 'pip install impedance' for full EIS functionality."
    )


# =============== EIS Data Parsing Functions ===============


def parse_eis_file(file_path):
    """
    Parse an EIS data file in various formats.

    Supported formats:
    - Text/CSV files with frequency, real, imaginary columns
    - Gamry .DTA files
    - BioLogic .z files
    - Autolab .dfr files

    Args:
        file_path (str): Path to the EIS data file

    Returns:
        dict: Dictionary containing frequency, Z_real, Z_imag, and metadata
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in [".csv", ".txt", ".dat"]:
        return parse_text_eis(file_path)
    elif file_ext == ".dta":
        return parse_gamry_eis(file_path)
    elif file_ext == ".z":
        return parse_biologic_eis(file_path)
    elif file_ext == ".dfr":
        return parse_autolab_eis(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def parse_text_eis(file_path):
    """
    Parse a text/CSV file with EIS data.

    Expected format: CSV or tab-delimited with headers for frequency, Z_real, Z_imag
    Alternative headers might be: freq/Hz, Re(Z)/Ohm, -Im(Z)/Ohm, etc.

    Args:
        file_path (str): Path to the text/CSV file

    Returns:
        dict: Dictionary with frequency, Z_real, Z_imag arrays, and metadata
    """
    # Try to determine the delimiter
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
        if "," in first_line:
            delimiter = ","
        elif "\t" in first_line:
            delimiter = "\t"
        else:
            delimiter = None  # Let pandas autodetect

    # Try to read with header
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
    except Exception:
        # Try reading with no header
        df = pd.read_csv(file_path, delimiter=delimiter, header=None)
        # Assign default column names
        df.columns = [f"Column_{i + 1}" for i in range(len(df.columns))]

    # Identify relevant columns
    freq_col = None
    real_col = None
    imag_col = None

    # Search for frequency column
    for col in df.columns:
        col_str = str(col).lower()
        if "freq" in col_str or "hz" in col_str:
            freq_col = col
            break

    # Search for real impedance column
    for col in df.columns:
        col_str = str(col).lower()
        if (
            "real" in col_str
            or "re(z)" in col_str
            or "z'" in col_str
            or "z_real" in col_str
            or "z real" in col_str
        ) and "imag" not in col_str:
            real_col = col
            break

    # Search for imaginary impedance column
    for col in df.columns:
        col_str = str(col).lower()
        if (
            "imag" in col_str
            or "im(z)" in col_str
            or 'z"' in col_str
            or "z_imag" in col_str
            or "z imag" in col_str
            or "-im" in col_str
        ):
            imag_col = col
            break

    # If we couldn't identify the columns, try assuming positions
    if freq_col is None and len(df.columns) >= 3:
        freq_col = df.columns[0]
    if real_col is None and len(df.columns) >= 3:
        real_col = df.columns[1]
    if imag_col is None and len(df.columns) >= 3:
        imag_col = df.columns[2]

    if freq_col is None or real_col is None or imag_col is None:
        raise ValueError(
            f"Could not identify frequency, real, and imaginary columns in file: {file_path}"
        )

    # Extract data
    frequency = df[freq_col].values
    z_real = df[real_col].values
    z_imag = df[imag_col].values

    # Extract metadata from filename and headers
    metadata = {"source_file": os.path.basename(file_path), "source_format": "Text/CSV"}

    # Check if imaginary values are already negated
    # In standard EIS notation, the imaginary part is plotted as -Z_imag vs Z_real
    # Some systems export already negated imaginary values
    imag_col_str = str(imag_col).lower()
    already_negated = "-im" in imag_col_str

    if not already_negated:
        # Negate the imaginary part for proper Nyquist plotting
        z_imag = -z_imag

    # Ensure frequency is in ascending order
    if not np.all(np.diff(frequency) >= 0):
        sort_idx = np.argsort(frequency)
        frequency = frequency[sort_idx]
        z_real = z_real[sort_idx]
        z_imag = z_imag[sort_idx]

    return {
        "frequency": frequency,
        "Z_real": z_real,
        "Z_imag": z_imag,
        "Z_mag": np.sqrt(z_real**2 + z_imag**2),
        "Z_phase": np.arctan2(z_imag, z_real) * (180 / np.pi),
        "metadata": metadata,
    }


def parse_gamry_eis(file_path):
    """
    Parse a Gamry .DTA file with EIS data.

    Args:
        file_path (str): Path to the Gamry .DTA file

    Returns:
        dict: Dictionary with frequency, Z_real, Z_imag arrays, and metadata
    """
    # Gamry .DTA files have a header section with metadata and a data section
    metadata = {}
    data_lines = []
    in_header = True

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if in_header and line.startswith("CURVE"):
                in_header = False
                continue

            if in_header:
                # Parse metadata
                if "=" in line:
                    key, value = line.split("=", 1)
                    metadata[key.strip()] = value.strip()
            else:
                # Parse data
                if line and not line.startswith("#"):
                    data_lines.append(line)

    # Convert data lines to arrays
    data = []
    for line in data_lines:
        try:
            values = [float(v) for v in line.split("\t")]
            data.append(values)
        except ValueError:
            continue

    if not data:
        raise ValueError(f"No data found in Gamry file: {file_path}")

    data_array = np.array(data)

    if data_array.shape[1] < 3:
        raise ValueError(f"Invalid data format in Gamry file: {file_path}")

    # Gamry typically uses ZCURVE format: freq, Zreal, Zimag, ...
    frequency = data_array[:, 0]
    z_real = data_array[:, 1]
    z_imag = data_array[:, 2]

    # Negate imaginary part if not already negated
    # Check sign convention from metadata if available
    if "ZCURVE" in metadata and "neg" not in metadata["ZCURVE"].lower():
        z_imag = -z_imag

    # Ensure frequency is in ascending order
    if not np.all(np.diff(frequency) >= 0):
        sort_idx = np.argsort(frequency)
        frequency = frequency[sort_idx]
        z_real = z_real[sort_idx]
        z_imag = z_imag[sort_idx]

    return {
        "frequency": frequency,
        "Z_real": z_real,
        "Z_imag": z_imag,
        "Z_mag": np.sqrt(z_real**2 + z_imag**2),
        "Z_phase": np.arctan2(z_imag, z_real) * (180 / np.pi),
        "metadata": {
            "source_file": os.path.basename(file_path),
            "source_format": "Gamry .DTA",
            **metadata,
        },
    }


def parse_biologic_eis(file_path):
    """
    Parse a BioLogic .z file with EIS data.

    Args:
        file_path (str): Path to the BioLogic .z file

    Returns:
        dict: Dictionary with frequency, Z_real, Z_imag arrays, and metadata
    """
    # BioLogic .z files are usually in a specific format
    # Try first as a simple text file with biologic formatting
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        # Find the data section
        header_lines = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("freq/Hz"):
                header_lines = i
                break

        # Parse metadata from header
        metadata = {
            "source_file": os.path.basename(file_path),
            "source_format": "BioLogic .z",
        }

        for i in range(min(header_lines, 20)):
            line = lines[i].strip()
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

        # Read data section
        df = pd.read_csv(file_path, skiprows=header_lines, sep="\t")

        # Identify columns
        freq_col = next((col for col in df.columns if "freq" in col.lower()), None)
        real_col = next((col for col in df.columns if "re(z)" in col.lower()), None)
        imag_col = next((col for col in df.columns if "im(z)" in col.lower()), None)

        if freq_col is None or real_col is None or imag_col is None:
            raise ValueError("Could not identify required columns in BioLogic file")

        frequency = df[freq_col].values
        z_real = df[real_col].values
        z_imag = df[imag_col].values

        # BioLogic typically uses negative imaginary convention directly
        # Check if values are already negated
        if "im(z)" in imag_col and "-" not in imag_col:
            z_imag = -z_imag

    except Exception as e:
        # If text parsing failed, try using a specialized library if available
        try:
            # Try importing galvani for BioLogic binary file support
            from galvani import BioLogic

            # Parse the binary file
            zfile = BioLogic.Zfile(file_path)

            # Extract frequency and impedance data
            frequency = zfile.frequency
            z_real = zfile.Re_Z
            z_imag = zfile.Im_Z

            # Ensure we're using the right sign convention
            z_imag = -z_imag

            metadata = {
                "source_file": os.path.basename(file_path),
                "source_format": "BioLogic .z (binary)",
            }

        except ImportError:
            raise ImportError(
                "Failed to parse BioLogic file. Consider installing 'galvani' package."
            ) from e
        except Exception as bin_error:
            raise ValueError(
                f"Could not parse BioLogic file: {e}. Binary parsing also failed: {bin_error}"
            )

    # Ensure frequency is in ascending order
    if not np.all(np.diff(frequency) >= 0):
        sort_idx = np.argsort(frequency)
        frequency = frequency[sort_idx]
        z_real = z_real[sort_idx]
        z_imag = z_imag[sort_idx]

    return {
        "frequency": frequency,
        "Z_real": z_real,
        "Z_imag": z_imag,
        "Z_mag": np.sqrt(z_real**2 + z_imag**2),
        "Z_phase": np.arctan2(z_imag, z_real) * (180 / np.pi),
        "metadata": metadata,
    }


def parse_autolab_eis(file_path):
    """
    Parse an Autolab .dfr file with EIS data.

    Args:
        file_path (str): Path to the Autolab .dfr file

    Returns:
        dict: Dictionary with frequency, Z_real, Z_imag arrays, and metadata
    """
    # Autolab .dfr files are XML-based, but we'll handle them as text for simplicity
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Extract frequency data
        freq_match = re.search(r"<Frequency>(.*?)</Frequency>", content, re.DOTALL)
        if not freq_match:
            raise ValueError("Could not find frequency data in Autolab file")

        freq_text = freq_match.group(1).strip()
        frequency = np.array([float(f) for f in freq_text.split(";") if f.strip()])

        # Extract real impedance data
        real_match = re.search(r"<Zreal>(.*?)</Zreal>", content, re.DOTALL)
        if not real_match:
            raise ValueError("Could not find real impedance data in Autolab file")

        real_text = real_match.group(1).strip()
        z_real = np.array([float(z) for z in real_text.split(";") if z.strip()])

        # Extract imaginary impedance data
        imag_match = re.search(r"<Zimag>(.*?)</Zimag>", content, re.DOTALL)
        if not imag_match:
            raise ValueError("Could not find imaginary impedance data in Autolab file")

        imag_text = imag_match.group(1).strip()
        z_imag = np.array([float(z) for z in imag_text.split(";") if z.strip()])

        # Autolab typically stores negative imaginary values directly (for Nyquist plotting)
        # Ensure our convention is consistent - invert if needed
        if np.median(z_imag) > 0:  # Most EIS data should have negative imaginary part
            z_imag = -z_imag

        # Extract basic metadata
        metadata = {
            "source_file": os.path.basename(file_path),
            "source_format": "Autolab .dfr",
        }

        # Extract procedure name if available
        proc_match = re.search(r"<ProcedureName>(.*?)</ProcedureName>", content)
        if proc_match:
            metadata["procedure"] = proc_match.group(1).strip()

    except Exception as e:
        raise ValueError(f"Error parsing Autolab file: {e}")

    # Ensure frequency is in ascending order
    if not np.all(np.diff(frequency) >= 0):
        sort_idx = np.argsort(frequency)
        frequency = frequency[sort_idx]
        z_real = z_real[sort_idx]
        z_imag = z_imag[sort_idx]

    return {
        "frequency": frequency,
        "Z_real": z_real,
        "Z_imag": z_imag,
        "Z_mag": np.sqrt(z_real**2 + z_imag**2),
        "Z_phase": np.arctan2(z_imag, z_real) * (180 / np.pi),
        "metadata": metadata,
    }


def import_eis_data(file_path, sample_id=None, metadata=None):
    """
    Import EIS data and optionally associate with a Sample in the database.

    Args:
        file_path (str): Path to the EIS data file
        sample_id (str, optional): ID of the Sample to associate with the EIS data
        metadata (dict, optional): Additional metadata to store with the EIS data

    Returns:
        dict: Imported EIS data with metadata
    """
    # Parse the EIS file
    eis_data = parse_eis_file(file_path)

    # Add user metadata if provided
    if metadata:
        eis_data["metadata"].update(metadata)

    # If a sample_id is provided, associate the EIS data with that sample
    if sample_id:
        sample = models.Sample.objects(id=sample_id).first()
        if not sample:
            raise ValueError(f"Sample with ID {sample_id} not found")

        # Create an EIS test result
        test_result = models.TestResult(
            sample=sample,
            tester=eis_data["metadata"].get("source_format", "EIS"),
            test_type="EIS",
            name=f"EIS_{os.path.basename(file_path)}",
            file_path=file_path,
            # Store EIS data as custom fields
            custom_data={
                "eis_data": {
                    "frequency": eis_data["frequency"].tolist(),
                    "Z_real": eis_data["Z_real"].tolist(),
                    "Z_imag": eis_data["Z_imag"].tolist(),
                },
                "eis_metadata": eis_data["metadata"],
            },
        )

        # Save the test result
        test_result.save()

        # Link to the sample
        sample.tests.append(test_result)
        sample.save()

        # Add the test_id to the returned data
        eis_data["test_id"] = str(test_result.id)

    return eis_data


def get_eis_data(test_id):
    """
    Retrieve EIS data from a stored test result.

    Args:
        test_id: ID of the TestResult containing EIS data

    Returns:
        dict: EIS data with frequency, Z_real, Z_imag arrays
    """
    test = models.TestResult.objects(id=test_id).first()
    if not test:
        raise ValueError(f"Test with ID {test_id} not found")

    if not hasattr(test, "custom_data") or "eis_data" not in test.custom_data:
        raise ValueError(f"Test {test_id} does not contain EIS data")

    eis_data = test.custom_data["eis_data"]
    eis_metadata = test.custom_data.get("eis_metadata", {})

    # Convert lists back to numpy arrays
    frequency = np.array(eis_data["frequency"])
    z_real = np.array(eis_data["Z_real"])
    z_imag = np.array(eis_data["Z_imag"])

    return {
        "frequency": frequency,
        "Z_real": z_real,
        "Z_imag": z_imag,
        "Z_mag": np.sqrt(z_real**2 + z_imag**2),
        "Z_phase": np.arctan2(z_imag, z_real) * (180 / np.pi),
        "metadata": eis_metadata,
        "test_id": str(test.id),
        "sample_name": utils.get_sample_name(test.sample),
    }


# =============== EIS Analysis and Fitting Functions ===============


def preprocess_eis_data(
    frequency, z_real, z_imag, f_min=None, f_max=None, inductive_filter=True
):
    """
    Preprocess EIS data for analysis by filtering frequency range and removing inductive effects.

    Args:
        frequency: Array of frequency values
        z_real: Array of real impedance values
        z_imag: Array of imaginary impedance values
        f_min: Minimum frequency to include (default: None = no minimum)
        f_max: Maximum frequency to include (default: None = no maximum)
        inductive_filter: Whether to filter out inductive behavior (positive imaginary) at high frequencies

    Returns:
        tuple: (frequency, z_real, z_imag) after preprocessing
    """
    # Create a mask for the frequency range
    mask = np.ones_like(frequency, dtype=bool)

    if f_min is not None:
        mask = mask & (frequency >= f_min)

    if f_max is not None:
        mask = mask & (frequency <= f_max)

    # Filter inductive effects (positive imaginary impedance at high frequencies)
    if inductive_filter:
        # Only keep points where Im(Z) is negative or very close to zero
        # This filters out inductive behavior which is often a measurement artifact
        mask = mask & (z_imag <= 0.01 * np.max(np.abs(z_imag)))

    # Apply the mask
    if not np.any(mask):
        raise ValueError("No data points remain after filtering")

    return frequency[mask], z_real[mask], z_imag[mask]


def validate_eis_data(frequency, z_real, z_imag):
    """
    Validate EIS data for quality and identify potential issues.

    Args:
        frequency: Array of frequency values
        z_real: Array of real impedance values
        z_imag: Array of imaginary impedance values

    Returns:
        dict: Validation results with quality metrics and warnings
    """
    results = {"quality_score": 0, "warnings": [], "metrics": {}}

    # Check for sufficient data points
    if len(frequency) < 10:
        results["warnings"].append("Insufficient data points for reliable analysis")

    # Check frequency range (at least 2 decades recommended)
    freq_range = frequency.max() / frequency.min()
    results["metrics"]["frequency_range_decades"] = np.log10(freq_range)

    if freq_range < 100:  # Less than 2 decades
        results["warnings"].append(
            f"Limited frequency range: {results['metrics']['frequency_range_decades']:.1f} decades"
        )

    # Check for noise in the data
    # Compute local derivatives and look for large fluctuations
    if len(frequency) > 5:
        z_mag = np.sqrt(z_real**2 + z_imag**2)
        log_freq = np.log10(frequency)
        log_z_mag = np.log10(z_mag)

        # Compute smoothness of the magnitude curve
        dz_smooth = np.gradient(log_z_mag, log_freq)
        dz2_smooth = np.gradient(dz_smooth, log_freq)

        # High second derivatives indicate sharp changes/noise
        max_d2z = np.max(np.abs(dz2_smooth))
        results["metrics"]["max_curvature"] = max_d2z

        if max_d2z > 5:
            results["warnings"].append("High noise detected in impedance data")

    # Check for Kramers-Kronig compliance if impedance.py is available
    if HAS_IMPEDANCE:
        try:
            from impedance.validation import linKK

            # Convert to impedance.py format
            z = z_real + 1j * z_imag

            # Run Lin-KK test
            kk_test = linKK(frequency, z, c=0.85, max_M=50)
            residuals = kk_test.get_residuals()

            # Calculate residual metrics
            avg_res = np.mean(np.abs(residuals))
            max_res = np.max(np.abs(residuals))

            results["metrics"]["avg_kk_residual"] = float(avg_res)
            results["metrics"]["max_kk_residual"] = float(max_res)

            if avg_res > 0.05:
                results["warnings"].append(
                    f"Poor Kramers-Kronig compliance (avg residual: {avg_res:.3f})"
                )

        except Exception as e:
            results["warnings"].append(
                f"Could not perform Kramers-Kronig validation: {str(e)}"
            )

    # Calculate quality score (0-100)
    # Start with perfect score and deduct for issues
    quality = 100

    # Deduct for limited frequency range
    if results["metrics"]["frequency_range_decades"] < 3:
        quality -= 20 * (3 - results["metrics"]["frequency_range_decades"]) / 3

    # Deduct for noise if detected
    if "max_curvature" in results["metrics"]:
        noise_penalty = min(30, max(0, (results["metrics"]["max_curvature"] - 2) * 5))
        quality -= noise_penalty

    # Deduct for KK compliance issues
    if "avg_kk_residual" in results["metrics"]:
        kk_penalty = min(40, max(0, results["metrics"]["avg_kk_residual"] * 500))
        quality -= kk_penalty

    # Deduct for insufficient points
    if len(frequency) < 20:
        quality -= 20 * (20 - len(frequency)) / 20

    results["quality_score"] = max(0, min(100, round(quality)))

    return results


def fit_circuit_model(
    frequency, z_real, z_imag, circuit_string, initial_guess=None, bounds=None
):
    """
    Fit EIS data to an equivalent circuit model using impedance.py.

    Args:
        frequency: Array of frequency values (Hz)
        z_real: Array of real impedance values (ohms)
        z_imag: Array of imaginary impedance values (ohms)
        circuit_string: Circuit description in impedance.py format
                        e.g., "R0-p(R1,C1)-p(R2,CPE2)-Wo3"
        initial_guess: Initial parameter values (if None, automatic estimation)
        bounds: Parameter bounds as (lower, upper) tuple of arrays

    Returns:
        dict: Fitting results including parameters, goodness of fit, and circuit elements
    """
    if not HAS_IMPEDANCE:
        raise ImportError(
            "impedance.py is required for circuit fitting. Install with 'pip install impedance'"
        )

    # Convert data to impedance.py format
    frequencies = frequency  # Already in Hz
    impedance = z_real + 1j * z_imag  # Complex impedance

    # Create the circuit model
    circuit = CustomCircuit(circuit_string, initial_guess=initial_guess)

    try:
        # Fit the circuit model to the data
        if bounds is not None:
            circuit.fit(frequencies, impedance, bounds=bounds)
        else:
            circuit.fit(frequencies, impedance)

        # Get the fitted parameters and their names
        param_names = circuit.get_param_names()
        param_values = circuit.parameters_

        # Calculate the goodness of fit
        z_fit = circuit.predict(frequencies)
        residuals = (impedance - z_fit) / np.abs(impedance)

        chi_squared = np.sum(np.abs(residuals) ** 2) / len(frequencies)
        r_squared = 1 - np.sum(np.abs(residuals) ** 2) / np.sum(
            np.abs((impedance - np.mean(impedance)) ** 2)
        )

        # Extract physical meaning of circuit elements
        elements = {}
        for i, name in enumerate(param_names):
            value = param_values[i]
            elements[name] = value

        # Create detailed result
        result = {
            "circuit_string": circuit_string,
            "parameters": dict(zip(param_names, param_values)),
            "elements": elements,
            "chi_squared": chi_squared,
            "r_squared": r_squared,
            "circuit_model": circuit,
            "fitted_impedance": z_fit,
        }

        return result

    except Exception as e:
        raise ValueError(f"Circuit fitting failed: {e}")


def fit_standard_circuits(frequency, z_real, z_imag, circuit_options=None):
    """
    Fit multiple standard circuit models and determine the best fit.

    Args:
        frequency: Array of frequency values
        z_real: Array of real impedance values
        z_imag: Array of imaginary impedance values
        circuit_options: List of circuit strings to try (default: standard battery models)

    Returns:
        dict: Results for all models, with the best model highlighted
    """
    if not HAS_IMPEDANCE:
        raise ImportError(
            "impedance.py is required for circuit fitting. Install with 'pip install impedance'"
        )

    # Default circuit options common for battery cells
    if circuit_options is None:
        circuit_options = [
            "R0",  # Simple resistor
            "R0-p(R1,C1)",  # Randles without diffusion
            "R0-p(R1,CPE1)",  # Randles with CPE
            "R0-p(R1,C1)-W2",  # Simple Randles with Warburg
            "R0-p(R1,CPE1)-W2",  # Randles with CPE and Warburg
            "R0-p(R1,C1)-p(R2,C2)",  # Two time constants
            "R0-p(R1,CPE1)-p(R2,CPE2)",  # Two time constants with CPEs
            "R0-p(R1,CPE1)-p(R2-W2,CPE2)",  # Complete battery model
        ]

    results = {}
    best_model = None
    best_r_squared = -np.inf

    for circuit_string in circuit_options:
        try:
            result = fit_circuit_model(frequency, z_real, z_imag, circuit_string)
            results[circuit_string] = result

            # Track the best model by R-squared value
            if result["r_squared"] > best_r_squared:
                best_r_squared = result["r_squared"]
                best_model = circuit_string
        except Exception as e:
            # If fitting fails for a particular model, log the error and continue
            print(f"Fitting failed for model {circuit_string}: {e}")
            results[circuit_string] = {"error": str(e)}

    # Add summary with best model
    summary = {
        "best_model": best_model,
        "best_r_squared": best_r_squared if best_model else None,
        "all_models": results,
    }

    return summary


def compute_characteristic_frequencies(frequency, z_real, z_imag):
    """
    Compute characteristic frequencies from EIS data (peaks, relaxation times).

    Args:
        frequency: Array of frequency values
        z_real: Array of real impedance values
        z_imag: Array of imaginary impedance values

    Returns:
        dict: Dictionary of characteristic frequencies and related metrics
    """
    # Calculate negative imaginary impedance for peak finding
    neg_z_imag = -z_imag

    # Find peaks in -Z" vs frequency
    from scipy.signal import find_peaks

    peaks, peak_properties = find_peaks(
        neg_z_imag, height=0.01 * max(neg_z_imag), prominence=0.05 * max(neg_z_imag)
    )

    char_frequencies = []

    if len(peaks) > 0:
        # For each peak, extract characteristic frequency and related parameters
        for i, peak_idx in enumerate(peaks):
            f_peak = frequency[peak_idx]
            z_imag_peak = neg_z_imag[peak_idx]
            z_real_peak = z_real[peak_idx]

            # Estimate time constant
            tau = 1 / (2 * np.pi * f_peak)

            char_frequencies.append(
                {
                    "frequency": float(f_peak),
                    "time_constant": float(tau),
                    "z_imag_peak": float(z_imag_peak),
                    "z_real_at_peak": float(z_real_peak),
                }
            )

    # Calculate frequency at phase minimum (often related to diffusion processes)
    phase = np.arctan2(z_imag, z_real) * (180 / np.pi)
    phase_min_idx = np.argmin(phase)
    f_phase_min = frequency[phase_min_idx]

    # Results dictionary
    result = {
        "characteristic_frequencies": char_frequencies,
        "peak_count": len(char_frequencies),
        "phase_minimum_frequency": float(f_phase_min),
    }

    # If impedance.py is available, estimate the distribution of relaxation times
    if HAS_IMPEDANCE and len(frequency) > 20:
        try:

            # Create frequencies and impedances in impedance.py format
            f = frequency
            z = z_real + 1j * z_imag

            # Perform DRT analysis
            frequencies_pred, gamma, tau, error, sigma = (
                impedance.models.circuits.drt.calculate_drt(
                    f,
                    z,
                    reg_method="ridge",
                    penalty_method="derivative",
                    M=1,
                    rbf_type="cubic",
                    sigma=0.05,
                )
            )

            # Find peaks in the DRT
            drt_peaks, _ = find_peaks(gamma, height=0.05 * max(gamma))

            # Add DRT results
            result["drt_analysis"] = {
                "tau": tau.tolist(),
                "gamma": gamma.tolist(),
                "peaks": [float(tau[p]) for p in drt_peaks],
                "error": float(error),
            }
        except Exception as e:
            print(f"DRT analysis failed: {e}")

    return result


def extract_physical_parameters(fit_result):
    """
    Extract physical parameters with physical meaning from circuit fit results.

    Args:
        fit_result: Result from fit_circuit_model

    Returns:
        dict: Physical parameters with descriptions
    """
    # Check for valid fit result
    if "parameters" not in fit_result:
        raise ValueError("Invalid fit result")

    # Extract model parameters
    params = fit_result["parameters"]
    circuit = fit_result["circuit_string"]

    # Initialize physical parameters dict
    physical_params = {}

    # Look for specific elements and extract their physical meaning

    # Series resistance (electrolyte resistance)
    if "R0" in params:
        physical_params["electrolyte_resistance"] = {
            "value": params["R0"],
            "unit": "ohm",
            "description": "Electrolyte/solution resistance",
        }

    # Charge transfer resistance
    for param in params:
        if param.startswith("R") and param != "R0":
            # This could be charge transfer resistance or other resistances
            physical_params[f"{param}_resistance"] = {
                "value": params[param],
                "unit": "ohm",
                "description": f"Resistance component ({param})",
            }

    # Capacitance (double layer)
    for param in params:
        if param.startswith("C"):
            physical_params[f"{param}_capacitance"] = {
                "value": params[param],
                "unit": "F",
                "description": f"Capacitance component ({param})",
            }

    # CPE parameters
    cpe_params = {}
    for param in params:
        if param.startswith("CPE") and param.endswith("_0"):
            base = param[:-2]  # Remove '_0'
            if f"{base}_1" in params:
                Q = params[param]
                n = params[f"{base}_1"]
                cpe_params[base] = (Q, n)

    # Convert CPE to effective capacitance where possible
    for cpe, (Q, n) in cpe_params.items():
        # Find associated resistance if available
        r_param = None
        for param in params:
            if param.startswith("R") and param != "R0":
                # Check if this R is paired with the CPE in a parallel element
                if f"p({param},{cpe})" in circuit or f"p({cpe},{param})" in circuit:
                    r_param = param
                    break

        # Calculate effective capacitance if resistance found
        if r_param is not None:
            R = params[r_param]
            # Brug formula: C_eff = (R^(1-n) * Q)^(1/n)
            C_eff = (R ** (1 - n) * Q) ** (1 / n)

            physical_params[f"{cpe}_effective_capacitance"] = {
                "value": C_eff,
                "unit": "F",
                "description": f"Effective capacitance calculated from CPE ({cpe})",
            }

        # Also add raw CPE parameters
        physical_params[f"{cpe}_Q"] = {
            "value": Q,
            "unit": "S*s^n",
            "description": f"CPE coefficient ({cpe})",
        }

        physical_params[f"{cpe}_n"] = {
            "value": n,
            "unit": "dimensionless",
            "description": f"CPE exponent ({cpe})",
        }

    # Warburg parameters
    for param in params:
        if param.startswith("W"):
            physical_params[f"{param}_warburg"] = {
                "value": params[param],
                "unit": "ohm/s^(1/2)",
                "description": f"Warburg coefficient ({param})",
            }

            # Estimate diffusion coefficient if it's a semi-infinite Warburg
            # For finite Warburg elements, the formula would need to be adjusted
            sigma = params[param]
            if sigma > 0:
                # D ≈ 1/(2 * (sigma*A)^2) * (R*T/n^2*F^2*C)^2
                # We don't have all parameters, so we just note the proportionality
                physical_params[f"{param}_diffusion_factor"] = {
                    "value": 1 / (2 * sigma**2),
                    "unit": "proportional to D",
                    "description": "Factor proportional to diffusion coefficient",
                }

    # Time constants
    for r_param in [p for p in params if p.startswith("R") and p != "R0"]:
        # Find capacitance or CPE paired with this resistance

        # Check for regular capacitance
        for c_param in [p for p in params if p.startswith("C")]:
            if (
                f"p({r_param},{c_param})" in circuit
                or f"p({c_param},{r_param})" in circuit
            ):
                # Calculate time constant
                tau = params[r_param] * params[c_param]
                physical_params[f"{r_param}_{c_param}_time_constant"] = {
                    "value": tau,
                    "unit": "s",
                    "description": f"Time constant for {r_param}-{c_param} pair",
                }

        # Check for CPE
        for cpe in cpe_params:
            if f"p({r_param},{cpe})" in circuit or f"p({cpe},{r_param})" in circuit:
                # For CPE, the characteristic frequency is calculated differently
                Q, n = cpe_params[cpe]
                R = params[r_param]

                # Characteristic frequency: f = 1/(2π * (R*Q)^(1/n))
                f_char = 1 / (2 * np.pi * (R * Q) ** (1 / n))
                tau_char = 1 / (2 * np.pi * f_char)

                physical_params[f"{r_param}_{cpe}_characteristic_frequency"] = {
                    "value": f_char,
                    "unit": "Hz",
                    "description": f"Characteristic frequency for {r_param}-{cpe} pair",
                }

                physical_params[f"{r_param}_{cpe}_time_constant"] = {
                    "value": tau_char,
                    "unit": "s",
                    "description": f"Characteristic time constant for {r_param}-{cpe} pair",
                }

    return physical_params


def compare_eis_spectra(test_ids, apply_normalization=False):
    """
    Compare multiple EIS spectra for the same sample or related samples.

    Args:
        test_ids: List of TestResult IDs containing EIS data
        apply_normalization: Whether to normalize impedance by area (if available)

    Returns:
        dict: Comparison results including trends and changes in key parameters
    """
    # Get EIS data for all test IDs
    eis_datasets = []

    for test_id in test_ids:
        try:
            data = get_eis_data(test_id)
            test = models.TestResult.objects(id=test_id).first()

            # Add metadata if available
            if test and hasattr(test, "date"):
                data["date"] = test.date

            eis_datasets.append(data)
        except Exception as e:
            print(f"Could not load EIS data for test {test_id}: {e}")

    if len(eis_datasets) < 2:
        raise ValueError("Need at least two valid EIS datasets for comparison")

    # Sort datasets by date if available
    if all("date" in data for data in eis_datasets):
        eis_datasets.sort(key=lambda x: x["date"])

    # Normalize impedance by area if requested and area information is available
    if apply_normalization:
        for i, data in enumerate(eis_datasets):
            test_id = data["test_id"]
            test = models.TestResult.objects(id=test_id).first()

            # Look for area information in test or sample
            area = None
            if test and hasattr(test, "custom_data") and "area" in test.custom_data:
                area = test.custom_data["area"]
            elif test and hasattr(test.sample, "area") and test.sample.area:
                area = test.sample.area

            if area and area > 0:
                # Normalize impedance by area (ohm -> ohm·cm²)
                data["Z_real"] = data["Z_real"] * area
                data["Z_imag"] = data["Z_imag"] * area
                data["Z_mag"] = data["Z_mag"] * area
                data["normalized"] = True
            else:
                data["normalized"] = False

    # Extract key parameters for each dataset
    key_params = []

    for data in eis_datasets:
        # Try to fit a Randles circuit as a common basis for comparison
        try:
            if HAS_IMPEDANCE:
                # Try basic Randles model
                randles_fit = fit_circuit_model(
                    data["frequency"],
                    data["Z_real"],
                    data["Z_imag"],
                    "R0-p(R1,CPE1)-W2",
                )

                params = extract_physical_parameters(randles_fit)
            else:
                # If impedance.py is not available, extract basic metrics
                params = {}

                # Extract high-frequency resistance (real axis intercept)
                hf_idx = np.argmax(data["frequency"])
                params["R_hf"] = {
                    "value": data["Z_real"][hf_idx],
                    "unit": "ohm" + ("·cm²" if data.get("normalized", False) else ""),
                }

                # Extract low-frequency resistance
                lf_idx = np.argmin(data["frequency"])
                params["R_lf"] = {
                    "value": data["Z_real"][lf_idx],
                    "unit": "ohm" + ("·cm²" if data.get("normalized", False) else ""),
                }

            # Extract characteristic frequencies
            char_freqs = compute_characteristic_frequencies(
                data["frequency"], data["Z_real"], data["Z_imag"]
            )

            # Store parameters
            key_params.append(
                {
                    "test_id": data["test_id"],
                    "sample_name": data.get("sample_name", "Unknown"),
                    "date": data.get("date", None),
                    "fitted_params": params if HAS_IMPEDANCE else {},
                    "characteristic_freqs": char_freqs,
                    "basic_metrics": {
                        "R_hf": (
                            params.get("R_hf", {}).get("value")
                            if not HAS_IMPEDANCE
                            else params.get("electrolyte_resistance", {}).get("value")
                        ),
                        "R_lf": (
                            params.get("R_lf", {}).get("value")
                            if not HAS_IMPEDANCE
                            else None
                        ),
                        "peak_count": char_freqs["peak_count"],
                    },
                }
            )

        except Exception as e:
            print(f"Error analyzing dataset {data['test_id']}: {e}")
            # Add basic data even if fitting fails
            key_params.append(
                {
                    "test_id": data["test_id"],
                    "sample_name": data.get("sample_name", "Unknown"),
                    "date": data.get("date", None),
                    "error": str(e),
                }
            )

    # Analyze changes over the series
    changes = {}

    # Only compute changes if we have parameters
    if len(key_params) >= 2 and all("basic_metrics" in p for p in key_params):
        # Look at electrolyte resistance trends
        r_hf_values = [
            p["basic_metrics"]["R_hf"]
            for p in key_params
            if p["basic_metrics"].get("R_hf") is not None
        ]

        if len(r_hf_values) >= 2:
            r_hf_change = (r_hf_values[-1] - r_hf_values[0]) / r_hf_values[0] * 100
            changes["R_hf_change_pct"] = r_hf_change

    # Return comparison results
    return {
        "datasets": eis_datasets,
        "parameters": key_params,
        "changes": changes,
        "normalized": apply_normalization,
    }


# =============== EIS Visualization Functions ===============


def plot_nyquist(
    eis_data, ax=None, label=None, include_fit=False, highlight_frequencies=False
):
    """
    Create a Nyquist plot (Z_real vs -Z_imag) from EIS data.

    Args:
        eis_data: Dictionary with Z_real and Z_imag arrays, or a tuple of (Z_real, Z_imag)
        ax: Matplotlib axis to plot on (if None, a new figure will be created)
        label: Label for the data series (if None, no label is used)
        include_fit: Whether to include circuit fit data if available in eis_data
        highlight_frequencies: Whether to highlight specific frequencies on the plot

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot
    """
    # Extract data based on input type
    if isinstance(eis_data, dict):
        z_real = eis_data["Z_real"]
        z_imag = eis_data["Z_imag"]
        frequency = eis_data.get("frequency", None)
        fitted_impedance = eis_data.get("fitted_impedance", None)
    elif isinstance(eis_data, tuple) and len(eis_data) >= 2:
        z_real = eis_data[0]
        z_imag = eis_data[1]
        frequency = eis_data[2] if len(eis_data) > 2 else None
        fitted_impedance = None
    else:
        raise ValueError(
            "eis_data must be a dictionary with Z_real and Z_imag keys, or a tuple of (Z_real, Z_imag)"
        )

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Plot Nyquist plot (-Z_imag vs Z_real)
    if label:
        ax.plot(z_real, -z_imag, "o-", label=label)
    else:
        ax.plot(z_real, -z_imag, "o-")

    # If impedance.py is available, we can use its enhanced plotting functions
    if HAS_IMPEDANCE and include_fit and fitted_impedance is not None:
        # Plot the fit
        z_fit_real = np.real(fitted_impedance)
        z_fit_imag = np.imag(fitted_impedance)
        ax.plot(z_fit_real, -z_fit_imag, "--", color="red", label="Fit")

    # Highlight specific frequencies if requested
    if highlight_frequencies and frequency is not None:
        # Define frequencies to highlight (in Hz)
        highlight_freqs = [
            0.01,
            0.1,
            1,
            10,
            100,
            1000,
            10000,
            100000,  # Standard frequencies
        ]

        # Find closest indices to highlight frequencies
        for freq in highlight_freqs:
            if frequency.min() <= freq <= frequency.max():
                idx = np.argmin(np.abs(frequency - freq))
                ax.plot(
                    z_real[idx],
                    -z_imag[idx],
                    "o",
                    markersize=8,
                    markerfacecolor="none",
                    markeredgecolor="red",
                )
                ax.annotate(
                    f"{freq} Hz",
                    (z_real[idx], -z_imag[idx]),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

    # Set plot properties
    ax.set_xlabel("Z$_{real}$ (Ω)")
    ax.set_ylabel("-Z$_{imag}$ (Ω)")
    ax.grid(True, alpha=0.3)

    # Equal aspect ratio for Nyquist plot
    ax.set_aspect("equal")

    # Add legend if label is provided
    if label or (include_fit and fitted_impedance is not None):
        ax.legend()

    return fig


def plot_bode(eis_data, ax=None, label=None, include_fit=False, phase_units="degrees"):
    """
    Create Bode plots (magnitude and phase vs frequency) from EIS data.

    Args:
        eis_data: Dictionary with frequency, Z_real, Z_imag arrays
        ax: List of two Matplotlib axes for magnitude and phase (if None, a new figure will be created)
        label: Label for the data series (if None, no label is used)
        include_fit: Whether to include circuit fit data if available in eis_data
        phase_units: Units for phase ('degrees' or 'radians')

    Returns:
        matplotlib.figure.Figure: The figure object containing the plots
    """
    # Extract data based on input type
    if isinstance(eis_data, dict):
        frequency = eis_data["frequency"]
        z_real = eis_data["Z_real"]
        z_imag = eis_data["Z_imag"]
        fitted_impedance = eis_data.get("fitted_impedance", None)
    else:
        raise ValueError(
            "eis_data must be a dictionary with frequency, Z_real, and Z_imag keys"
        )

    # Calculate impedance magnitude and phase
    z_mag = np.sqrt(z_real**2 + z_imag**2)
    z_phase = np.arctan2(z_imag, z_real)

    # Convert phase to degrees if requested
    if phase_units == "degrees":
        z_phase = z_phase * (180 / np.pi)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    else:
        fig = ax[0].figure

    # Plot Bode magnitude
    if label:
        ax[0].loglog(frequency, z_mag, "o-", label=label)
    else:
        ax[0].loglog(frequency, z_mag, "o-")

    # Plot Bode phase (negative by convention)
    if label:
        ax[1].semilogx(frequency, -z_phase, "o-", label=label)
    else:
        ax[1].semilogx(frequency, -z_phase, "o-")

    # If impedance.py is available and fit data is included
    if HAS_IMPEDANCE and include_fit and fitted_impedance is not None:
        # Calculate fit magnitude and phase
        z_fit_mag = np.abs(fitted_impedance)
        z_fit_phase = np.angle(fitted_impedance)

        # Convert phase to degrees if requested
        if phase_units == "degrees":
            z_fit_phase = z_fit_phase * (180 / np.pi)

        # Plot fit data
        ax[0].loglog(frequency, z_fit_mag, "--", color="red", label="Fit")
        ax[1].semilogx(frequency, -z_fit_phase, "--", color="red", label="Fit")

        # Add legends
        ax[0].legend(loc="best")
        ax[1].legend(loc="best")

    # Set plot properties
    ax[0].set_ylabel("|Z| (Ω)")
    ax[0].grid(True, alpha=0.3)

    phase_label = "-Phase (°)" if phase_units == "degrees" else "-Phase (rad)"
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel(phase_label)
    ax[1].grid(True, alpha=0.3)

    # Add legends if labels provided
    if label or (include_fit and fitted_impedance is not None):
        ax[0].legend()
        ax[1].legend()

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_drt(eis_data):
    """
    Plot Distribution of Relaxation Times (DRT) analysis of EIS data.

    Args:
        eis_data: Dictionary with frequency, Z_real, Z_imag arrays

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot
    """
    if not HAS_IMPEDANCE:
        raise ImportError(
            "impedance.py is required for DRT analysis. Install with 'pip install impedance'"
        )

    # Extract data
    frequency = eis_data["frequency"]
    z = eis_data["Z_real"] + 1j * eis_data["Z_imag"]

    try:
        # Perform DRT analysis
        freq_pred, gamma, tau, error, sigma = (
            impedance.models.circuits.drt.calculate_drt(
                frequency,
                z,
                reg_method="ridge",
                penalty_method="derivative",
                M=1,
                rbf_type="cubic",
                sigma=0.05,
            )
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot DRT
        ax.loglog(tau, gamma, "-", linewidth=2)

        # Format plot
        ax.set_xlabel("Time constant (s)")
        ax.set_ylabel("Distribution function γ (Ω/ln(s))")
        ax.set_title("Distribution of Relaxation Times (DRT)")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        # Find and annotate peaks
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(gamma, height=0.05 * max(gamma))

        for peak_idx in peaks:
            ax.plot(tau[peak_idx], gamma[peak_idx], "ro")
            ax.annotate(
                f"{tau[peak_idx]:.1e} s",
                xy=(tau[peak_idx], gamma[peak_idx]),
                xytext=(5, 5),
                textcoords="offset points",
            )

        return fig

    except Exception as e:
        raise ValueError(f"DRT analysis failed: {e}")


def generate_comprehensive_eis_report(eis_data, fit_results=None, filename=None):
    """
    Generate a comprehensive report with multiple EIS visualizations.

    Args:
        eis_data: Dictionary with EIS data
        fit_results: Optional dictionary with circuit fitting results
        filename: Output PDF filename (if None, returns the figure instead)

    Returns:
        matplotlib.figure.Figure or str: Figure object or path to saved PDF
    """
    # Create a multi-panel figure
    fig = plt.figure(figsize=(12, 10))

    # Define grid layout
    gs = fig.add_gridspec(2, 2)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Nyquist plot
    ax2 = fig.add_subplot(gs[0, 1])  # Complex plane with fits
    ax3 = fig.add_subplot(gs[1, 0])  # Bode magnitude
    ax4 = fig.add_subplot(gs[1, 1])  # Bode phase

    # Plot Nyquist
    plot_nyquist(
        {"Z_real": eis_data["Z_real"], "Z_imag": eis_data["Z_imag"]},
        ax=ax1,
        label="Data",
    )
    ax1.set_title("Nyquist Plot")

    # If fit results are provided, plot fitted data
    if fit_results is not None and "fitted_impedance" in fit_results:
        z_fit = fit_results["fitted_impedance"]
        ax1.plot(z_fit.real, -z_fit.imag, "--", color="red", label="Fit")
        ax1.legend()

        # Plot residuals in complex plane
        residuals = (eis_data["Z_real"] + 1j * eis_data["Z_imag"]) - z_fit
        ax2.plot(residuals.real, residuals.imag, "o", color="green")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Fit Residuals")
        ax2.set_xlabel("Re(Z_data - Z_fit) (Ω)")
        ax2.set_ylabel("Im(Z_data - Z_fit) (Ω)")

        # Add circuit model and parameters if available
        if "circuit_string" in fit_results:
            circuit_text = f"Model: {fit_results['circuit_string']}\n"
            circuit_text += f"R² = {fit_results['r_squared']:.4f}\n\n"

            # Add parameters
            for name, value in fit_results["parameters"].items():
                circuit_text += f"{name} = {value:.4e}\n"

            ax2.text(
                0.05,
                0.95,
                circuit_text,
                transform=ax2.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
    else:
        ax2.text(
            0.5,
            0.5,
            "No fit data available",
            transform=ax2.transAxes,
            ha="center",
            va="center",
        )
        ax2.set_title("Fit Information")

    # Plot Bode magnitude
    z_mag = np.sqrt(eis_data["Z_real"] ** 2 + eis_data["Z_imag"] ** 2)
    ax3.loglog(eis_data["frequency"], z_mag, "o-", label="Data")
    ax3.set_title("Bode Magnitude Plot")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("|Z| (Ω)")
    ax3.grid(True, which="both", linestyle="--", alpha=0.5)

    if fit_results is not None and "fitted_impedance" in fit_results:
        z_fit_mag = np.abs(fit_results["fitted_impedance"])
        ax3.loglog(eis_data["frequency"], z_fit_mag, "--", color="red", label="Fit")
        ax3.legend()

    # Plot Bode phase
    z_phase = np.arctan2(eis_data["Z_imag"], eis_data["Z_real"]) * (180 / np.pi)
    ax4.semilogx(eis_data["frequency"], -z_phase, "o-", label="Data")
    ax4.set_title("Bode Phase Plot")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("-Phase (°)")
    ax4.grid(True, which="both", linestyle="--", alpha=0.5)

    if fit_results is not None and "fitted_impedance" in fit_results:
        z_fit_phase = np.angle(fit_results["fitted_impedance"]) * (180 / np.pi)
        ax4.semilogx(
            eis_data["frequency"], -z_fit_phase, "--", color="red", label="Fit"
        )
        ax4.legend()

    # Add metadata and general information at the top
    metadata = eis_data.get("metadata", {})
    info_text = f"Sample: {eis_data.get('sample_name', 'Unknown')}\n"

    if "source_file" in metadata:
        info_text += f"Source: {metadata['source_file']}\n"

    if "date" in eis_data:
        info_text += f"Date: {eis_data['date'].strftime('%Y-%m-%d %H:%M:%S')}\n"

    plt.figtext(
        0.5,
        0.98,
        info_text,
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return filename

    return fig
