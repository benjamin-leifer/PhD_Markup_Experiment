#!/usr/bin/env python
"""
Example script demonstrating how to use the advanced analysis features.

This script shows how to:
1. Perform differential capacity analysis (dQ/dV)
2. Analyze capacity fade patterns and predict cycle life
3. Detect anomalies in cycling data
4. Compare and cluster multiple tests
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
from battery_analysis import advanced_analysis as aa


def main():
    """Main example function demonstrating advanced analysis features."""

    # Connect to MongoDB
    logging.info("Connecting to MongoDB...")
    connected = utils.connect_to_database('battery_test_db')

    if not connected:
        logging.error("Failed to connect to database. Make sure MongoDB is running.")
        return

    # For this example, we'll generate a simulated test with known characteristics
    logging.info("Creating a simulated test for demonstration...")

    # Create a sample
    sample = models.Sample(
        name="Demo_Advanced_Analysis",
        chemistry="NMC811/Graphite",
        form_factor="18650",
        nominal_capacity=3500,  # 3500 mAh
        tags=["demo", "advanced_analysis"]
    )
    sample.save()

    # Create a test with simulated cycle data showing capacity fade
    test = create_simulated_test(sample)

    # 1. Perform differential capacity analysis (dQ/dV)
    logging.info("1. Differential Capacity Analysis (dQ/dV)...")
    try:
        # Simulate voltage and capacity data for dQ/dV analysis
        voltage_data, capacity_data = simulate_voltage_capacity_data()

        # Perform differential capacity analysis
        v_centers, dq_dv = aa.differential_capacity_analysis(voltage_data, capacity_data)

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(v_centers, dq_dv)
        plt.xlabel('Voltage (V)')
        plt.ylabel('dQ/dV (mAh/V)')
        plt.title('Differential Capacity Analysis')
        plt.grid(True)
        plt.savefig('dqdv_analysis.png')
        plt.close()
        logging.info("- dQ/dV plot saved as dqdv_analysis.png")

        # Identify peaks in dQ/dV curve (phase transitions)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(dq_dv, height=0.1 * max(dq_dv), distance=50)

        logging.info(f"- Identified {len(peaks)} major phase transitions at voltages:")
        for peak_idx in peaks:
            logging.info(f"  * {v_centers[peak_idx]:.3f} V")

    except Exception as e:
        logging.error(f"Error in differential capacity analysis: {e}")

    # 2. Analyze capacity fade patterns and predict cycle life
    logging.info("\n2. Capacity Fade Analysis...")
    try:
        fade_analysis = aa.capacity_fade_analysis(test.id)

        logging.info(f"- Initial capacity: {fade_analysis['initial_capacity']:.2f} mAh")
        logging.info(f"- Final capacity: {fade_analysis['final_capacity']:.2f} mAh")
        logging.info(f"- Capacity retention: {fade_analysis['capacity_retention'] * 100:.2f}%")
        logging.info(f"- Fade rate: {fade_analysis['fade_rate_pct_per_cycle']:.4f}% per cycle")

        if fade_analysis['best_model']:
            best_model = fade_analysis['best_model']
            logging.info(f"- Best fit model: {fade_analysis['fade_models'][best_model]['name']}")
            logging.info(f"- R² value: {fade_analysis['fade_models'][best_model]['r_squared']:.4f}")

            eol_cycle = fade_analysis['predicted_eol_cycle']
            if eol_cycle:
                logging.info(f"- Predicted 80% capacity retention at cycle: {int(eol_cycle)}")
                logging.info(f"- Prediction confidence: {fade_analysis['confidence'] * 100:.1f}%")

    except Exception as e:
        logging.error(f"Error in capacity fade analysis: {e}")

    # 3. Detect anomalies in cycling data
    logging.info("\n3. Anomaly Detection...")
    try:
        # Add an anomaly to the test data
        add_anomaly_to_test(test)

        # Detect anomalies
        anomaly_results = aa.detect_anomalies(test.id, metric='discharge_capacity')

        logging.info(f"- Analyzed metric: {anomaly_results['metric_analyzed']}")
        logging.info(f"- Normal range: {anomaly_results['normal_range']['mean']:.2f} ± "
                     f"{anomaly_results['normal_range']['std_dev']:.2f}")
        logging.info(f"- Detected {anomaly_results['anomaly_count']} anomalies:")

        for anomaly in anomaly_results['anomalies']:
            logging.info(f"  * Cycle {anomaly['cycle']}: {anomaly['value']:.2f} mAh "
                         f"(detected via {anomaly['detection_method']}, "
                         f"significance: {anomaly['significance']:.2f})")

    except Exception as e:
        logging.error(f"Error in anomaly detection: {e}")

    # 4. Create multiple tests for comparison and clustering
    logging.info("\n4. Test Comparison and Clustering...")
    try:
        # Create several more tests with different characteristics
        test_ids = create_multiple_tests(sample)
        test_ids.append(str(test.id))  # Include our original test

        # Cluster tests based on performance metrics
        cluster_results = aa.cluster_tests(test_ids)

        logging.info(f"- Identified {cluster_results['n_clusters']} clusters of tests:")
        for cluster_id, tests in cluster_results['clusters'].items():
            logging.info(f"  * Cluster {cluster_id}: {len(tests)} tests")
            for t in tests:
                logging.info(f"    - {t['test_name']}")

        # Find tests similar to our reference test
        similar_tests = aa.find_similar_tests(test.id)

        logging.info(f"\n- Tests most similar to {test.name}:")
        for i, t in enumerate(similar_tests):
            logging.info(f"  {i + 1}. {t['test_name']} (similarity: {t['similarity']:.2f})")

    except Exception as e:
        logging.error(f"Error in test comparison: {e}")

    logging.info("\nAdvanced analysis examples completed!")


def create_simulated_test(sample, cycles=100):
    """Create a test with simulated cycle data."""

    # Create a new test
    test = models.TestResult(
        sample=sample,
        tester="Simulated",
        test_type="Cycling",
        name="Simulated_Advanced_Analysis",
        upper_cutoff_voltage=4.2,
        lower_cutoff_voltage=3.0,
        charge_rate=0.5,
        discharge_rate=1.0
    )

    # Simulate cycle data with realistic capacity fade
    # Using an exponential decay model: capacity = a * exp(-b * cycle) + c
    a = -700  # Initial capacity drop amount
    b = 0.01  # Decay rate
    c = 3500  # Final stable capacity

    for i in range(1, cycles + 1):
        # Calculate discharge capacity with some noise
        cycle_num = i
        true_capacity = c + a * np.exp(-b * cycle_num)
        noise = np.random.normal(0, 10)  # 10 mAh standard deviation of noise
        discharge_capacity = true_capacity + noise

        # Calculate charge capacity (slightly higher than discharge)
        charge_capacity = discharge_capacity / 0.99  # 99% coulombic efficiency

        # Calculate coulombic efficiency (with small variations)
        ce_noise = np.random.normal(0, 0.002)  # Small variations in CE
        coulombic_efficiency = 0.99 + ce_noise

        # Create cycle summary
        cycle = models.CycleSummary(
            cycle_index=cycle_num,
            charge_capacity=charge_capacity,
            discharge_capacity=discharge_capacity,
            coulombic_efficiency=coulombic_efficiency
        )

        # Add cycle to test
        test.cycles.append(cycle)

    # Calculate test metrics
    test.cycle_count = len(test.cycles)
    test.initial_capacity = test.cycles[0].discharge_capacity
    test.final_capacity = test.cycles[-1].discharge_capacity
    test.capacity_retention = test.final_capacity / test.initial_capacity
    test.avg_coulombic_eff = np.mean([c.coulombic_efficiency for c in test.cycles])

    # Save test
    test.save()

    # Update sample references
    sample.tests.append(test)
    sample.save()

    return test


def add_anomaly_to_test(test):
    """Add an anomalous cycle to a test."""
    if len(test.cycles) < 20:
        return

    # Choose a random cycle in the middle
    anomaly_idx = len(test.cycles) // 2

    # Get the cycle to modify
    cycle = test.cycles[anomaly_idx]

    # Change the discharge capacity to an anomalous value (30% drop)
    original_capacity = cycle.discharge_capacity
    cycle.discharge_capacity = original_capacity * 0.7

    # Update the test
    test.save()


def simulate_voltage_capacity_data():
    """Simulate voltage vs capacity data for a lithium-ion cell."""
    # Create voltage array from 3.0V to 4.2V
    voltage = np.linspace(3.0, 4.2, 1000)

    # Initialize capacity array
    capacity = np.zeros_like(voltage)

    # Add realistic phase transitions for a typical NMC or LCO battery
    # using Gaussian peaks in dQ/dV, which means sigmoid shapes in Q vs V
    transitions = [
        (3.45, 0.15, 200),  # (voltage center, width, intensity)
        (3.70, 0.1, 300),
        (3.95, 0.12, 250)
    ]

    # Create sigmoid shapes for capacity
    for v_center, width, intensity in transitions:
        # Sigmoid function centered at v_center
        capacity += intensity * (1 / (1 + np.exp(-(voltage - v_center) / (width / 5))))

    # Scale capacity to realistic range (0 to 3500 mAh)
    capacity = capacity / capacity.max() * 3500

    return voltage, capacity


def create_multiple_tests(sample):
    """Create multiple tests with different characteristics for clustering."""
    test_ids = []

    # Test 1: High capacity, good retention
    test1 = models.TestResult(
        sample=sample,
        tester="Simulated",
        test_type="Cycling",
        name="High_Cap_Good_Retention",
        cycle_count=100,
        initial_capacity=3600,
        final_capacity=3400,
        capacity_retention=3400 / 3600,
        avg_coulombic_eff=0.995
    )
    test1.save()
    test_ids.append(str(test1.id))

    # Test 2: High capacity, poor retention
    test2 = models.TestResult(
        sample=sample,
        tester="Simulated",
        test_type="Cycling",
        name="High_Cap_Poor_Retention",
        cycle_count=100,
        initial_capacity=3650,
        final_capacity=2800,
        capacity_retention=2800 / 3650,
        avg_coulombic_eff=0.988
    )
    test2.save()
    test_ids.append(str(test2.id))

    # Test 3: Low capacity, good retention
    test3 = models.TestResult(
        sample=sample,
        tester="Simulated",
        test_type="Cycling",
        name="Low_Cap_Good_Retention",
        cycle_count=100,
        initial_capacity=3100,
        final_capacity=2950,
        capacity_retention=2950 / 3100,
        avg_coulombic_eff=0.994
    )
    test3.save()
    test_ids.append(str(test3.id))

    # Test 4: Low capacity, poor retention
    test4 = models.TestResult(
        sample=sample,
        tester="Simulated",
        test_type="Cycling",
        name="Low_Cap_Poor_Retention",
        cycle_count=100,
        initial_capacity=3050,
        final_capacity=2250,
        capacity_retention=2250 / 3050,
        avg_coulombic_eff=0.985
    )
    test4.save()
    test_ids.append(str(test4.id))

    # Test 5: Medium capacity, very long cycle life
    test5 = models.TestResult(
        sample=sample,
        tester="Simulated",
        test_type="Cycling",
        name="Long_Cycle_Life",
        cycle_count=500,
        initial_capacity=3300,
        final_capacity=2900,
        capacity_retention=2900 / 3300,
        avg_coulombic_eff=0.998
    )
    test5.save()
    test_ids.append(str(test5.id))

    # Update sample references
    sample.tests.extend([test1, test2, test3, test4, test5])
    sample.save()

    return test_ids


if __name__ == "__main__":
    main()