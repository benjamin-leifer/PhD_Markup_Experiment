#!/usr/bin/env python
"""
Example script demonstrating how to use the EIS module.

This script shows how to:
1. Import EIS data from files
2. Visualize EIS data with Nyquist and Bode plots
3. Fit equivalent circuit models to the data
4. Extract physical parameters from circuit fits
5. Generate comprehensive EIS reports
"""

import os
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to run the example directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Import package components
from battery_analysis import utils, models
from battery_analysis import eis


def main():
    """Main example function demonstrating EIS module functionality."""

    # Check if impedance.py is available
    if not eis.HAS_IMPEDANCE:
        logging.warning("impedance.py is not installed. Some functionality will be limited.")
        logging.warning("Install with: pip install impedance")

    # Connect to MongoDB
    logging.info("Connecting to MongoDB...")
    connected = utils.connect_to_database('battery_test_db')

    if not connected:
        logging.error("Failed to connect to database. Make sure MongoDB is running.")
        return

    # Create a sample to store EIS data
    sample = models.Sample(
        name="EIS_Demo_Sample",
        chemistry="LiNi0.8Mn0.1Co0.1O2/Graphite",
        form_factor="Coin cell",
        nominal_capacity=25.0,  # 25 mAh
        tags=["demo", "EIS"],
        area=1.54  # cm²
    )
    sample.save()

    # 1. Import or simulate EIS data
    logging.info("1. Creating simulated EIS data...")

    eis_data = simulate_eis_data()

    logging.info(f"- Generated EIS data with {len(eis_data['frequency'])} frequency points")
    logging.info(f"- Frequency range: {eis_data['frequency'].min():.2e} - {eis_data['frequency'].max():.2e} Hz")

    # Save EIS data to the database
    logging.info("- Storing EIS data in database...")

    test_data = eis.import_eis_data(
        "simulated_eis.csv",  # This is just a reference name since we're using simulated data
        sample_id=str(sample.id),
        metadata={"test_type": "EIS", "temperature": 25, "soc": 50}
    )

    test_id = test_data['test_id']
    logging.info(f"- Stored as test ID: {test_id}")

    # 2. Retrieve and validate EIS data from database
    logging.info("\n2. Retrieving and validating EIS data...")

    # Retrieve the data
    retrieved_data = eis.get_eis_data(test_id)

    # Validate the data
    validation_results = eis.validate_eis_data(
        retrieved_data['frequency'],
        retrieved_data['Z_real'],
        retrieved_data['Z_imag']
    )

    logging.info(f"- Data quality score: {validation_results['quality_score']}/100")

    if validation_results['warnings']:
        logging.info("- Validation warnings:")
        for warning in validation_results['warnings']:
            logging.info(f"  * {warning}")
    else:
        logging.info("- No validation warnings")

    # 3. Visualize with Nyquist and Bode plots
    logging.info("\n3. Creating EIS visualizations...")

    # Nyquist plot
    fig_nyquist = eis.plot_nyquist(
        retrieved_data,
        label="LiNMC Coin Cell",
        highlight_frequencies=True
    )
    fig_nyquist.savefig("eis_nyquist.png", dpi=300, bbox_inches='tight')
    logging.info("- Nyquist plot saved as eis_nyquist.png")

    # Bode plot
    fig_bode = eis.plot_bode(
        retrieved_data,
        label="LiNMC Coin Cell"
    )
    fig_bode.savefig("eis_bode.png", dpi=300, bbox_inches='tight')
    logging.info("- Bode plot saved as eis_bode.png")

    # 4. Fit EIS data to equivalent circuit models
    if eis.HAS_IMPEDANCE:
        logging.info("\n4. Fitting equivalent circuit models...")

        # Try multiple circuit models
        circuit_fit_results = eis.fit_standard_circuits(
            retrieved_data['frequency'],
            retrieved_data['Z_real'],
            retrieved_data['Z_imag']
        )

        # Get the best model
        best_model = circuit_fit_results['best_model']
        best_fit = circuit_fit_results['all_models'][best_model]

        logging.info(f"- Best circuit model: {best_model}")
        logging.info(f"- R² goodness of fit: {best_fit['r_squared']:.4f}")
        logging.info("- Circuit parameters:")

        for param, value in best_fit['parameters'].items():
            logging.info(f"  * {param}: {value:.4e}")

        # 5. Extract physical parameters
        logging.info("\n5. Extracting physical parameters...")

        physical_params = eis.extract_physical_parameters(best_fit)

        logging.info("Physical interpretation of circuit elements:")
        for name, param in physical_params.items():
            logging.info(f"  * {name}: {param['value']:.4e} {param['unit']} - {param['description']}")

        # 6. Generate comprehensive report
        logging.info("\n6. Generating comprehensive EIS report...")

        # Add fitted impedance to the data for plotting
        retrieved_data['fitted_impedance'] = best_fit['fitted_impedance']

        report_file = eis.generate_comprehensive_eis_report(
            retrieved_data,
            fit_results=best_fit,
            filename="eis_comprehensive_report.pdf"
        )

        logging.info(f"- Comprehensive report saved as {report_file}")

        # 7. Distribution of Relaxation Times (DRT) analysis
        try:
            logging.info("\n7. Performing DRT analysis...")

            fig_drt = eis.plot_drt(retrieved_data)
            fig_drt.savefig("eis_drt.png", dpi=300, bbox_inches='tight')
            logging.info("- DRT plot saved as eis_drt.png")

        except Exception as e:
            logging.warning(f"- DRT analysis failed: {e}")

    else:
        logging.info("\nSkipping circuit fitting and DRT analysis (impedance.py not installed)")

    # 8. Extract characteristic frequencies
    logging.info("\n8. Calculating characteristic frequencies...")

    char_freqs = eis.compute_characteristic_frequencies(
        retrieved_data['frequency'],
        retrieved_data['Z_real'],
        retrieved_data['Z_imag']
    )

    logging.info(f"- Detected {char_freqs['peak_count']} characteristic time constants:")

    for i, peak in enumerate(char_freqs['characteristic_frequencies']):
        logging.info(f"  * Peak {i + 1}: {peak['frequency']:.2e} Hz (τ = {peak['time_constant']:.2e} s)")

    logging.info(f"- Phase minimum frequency: {char_freqs['phase_minimum_frequency']:.2e} Hz")

    # 9. Compare multiple EIS spectra (simulating different tests)
    logging.info("\n9. Comparing multiple EIS spectra...")

    # Create a second EIS test with slightly different parameters
    # (simulating aging or different state of charge)
    second_eis_data = simulate_eis_data(
        rs_factor=1.2,  # 20% increase in series resistance
        rct_factor=1.5,  # 50% increase in charge transfer resistance
    )

    # Save the second test
    second_test_data = eis.import_eis_data(
        "simulated_eis_aged.csv",
        sample_id=str(sample.id),
        metadata={"test_type": "EIS", "temperature": 25, "soc": 50, "state": "Aged"}
    )

    # Compare the two tests
    comparison = eis.compare_eis_spectra([test_id, second_test_data['test_id']])

    logging.info("- EIS comparison results:")
    if 'changes' in comparison and 'R_hf_change_pct' in comparison['changes']:
        logging.info(f"  * Series resistance change: {comparison['changes']['R_hf_change_pct']:.1f}%")

    logging.info("\nEIS example completed!")


def simulate_eis_data(rs_factor=1.0, rct_factor=1.0):
    """
    Simulate EIS data for a typical lithium-ion battery.

    Simulates a Randles circuit with:
    - Rs (series resistance)
    - Rct (charge transfer resistance) in parallel with CPE (constant phase element)
    - W (Warburg element) for solid-state diffusion

    Args:
        rs_factor: Factor to multiply series resistance by
        rct_factor: Factor to multiply charge transfer resistance by

    Returns:
        dict: Dictionary with simulated EIS data
    """
    # Define frequency range (logarithmically spaced)
    frequency = np.logspace(-2, 5, 60)  # 0.01 Hz to 100 kHz

    # Define Randles circuit parameters
    rs = 5.0 * rs_factor  # Series resistance (ohm)
    rct = 15.0 * rct_factor  # Charge transfer resistance (ohm)
    cpe_q = 1e-4  # CPE parameter (F/s^(1-n))
    cpe_n = 0.8  # CPE exponent (1 for perfect capacitor)
    w = 10.0  # Warburg coefficient

    # Initialize impedance array
    z = np.zeros(len(frequency), dtype=complex)

    # Calculate impedance for each frequency
    for i, f in enumerate(frequency):
        omega = 2 * np.pi * f

        # CPE impedance: Z_CPE = 1/(Q*(jω)^n)
        z_cpe = 1 / (cpe_q * (1j * omega) ** cpe_n)

        # Warburg impedance: Z_W = W / sqrt(jω)
        z_w = w / np.sqrt(1j * omega)

        # Parallel combination of Rct and CPE: Z_parallel = 1/(1/Rct + 1/Z_CPE)
        z_parallel = 1 / (1 / rct + 1 / z_cpe)

        # Series combination of Rs, Z_parallel, and Z_W
        z[i] = rs + z_parallel + z_w

    # Extract real and imaginary parts
    z_real = np.real(z)
    z_imag = np.imag(z)

    # Add some random noise (1% of magnitude)
    z_mag = np.abs(z)
    noise_level = 0.01  # 1% noise

    z_real += np.random.normal(0, noise_level * z_mag)
    z_imag += np.random.normal(0, noise_level * z_mag)

    # Create data dictionary
    return {
        'frequency': frequency,
        'Z_real': z_real,
        'Z_imag': z_imag,
        'Z_mag': np.sqrt(z_real ** 2 + z_imag ** 2),
        'Z_phase': np.arctan2(z_imag, z_real) * (180 / np.pi),
        'metadata': {
            'source_file': 'simulated_eis.csv',
            'source_format': 'Simulated',
            'parameters': {
                'Rs': rs,
                'Rct': rct,
                'CPE_Q': cpe_q,
                'CPE_n': cpe_n,
                'W': w
            }
        }
    }


if __name__ == "__main__":
    main()