"""
Analysis module for battery test data and detailed cycle analysis.
"""
# Simply import from the analysis_core.py file
from battery_analysis.analysis_core import (
    compute_metrics,
    create_test_result,
    update_sample_properties,
    compare_samples,
    get_cycle_data
)

# Import from cycle_data_analysis.py
try:
    from .cycle_data_analysis import (
        get_cycle_voltage_vs_capacity,
        calculate_differential_capacity,
        plot_cycle_voltage_capacity,
        plot_differential_capacity
    )
    from .protocol_detection import (
        is_last_cycle_complete,
        calculate_cycle_crates,
        summarize_protocol,
        get_or_create_protocol,
        detect_and_update_test_protocol,
    )
except ImportError:
    import logging
    logging.warning("Could not import cycle_data_analysis functions")

# Export all functions
__all__ = [
    'compute_metrics',
    'create_test_result',
    'update_sample_properties',
    'compare_samples',
    'get_cycle_data',
    'get_cycle_voltage_vs_capacity',
    'calculate_differential_capacity',
    'plot_cycle_voltage_capacity',
    'plot_differential_capacity',
    'is_last_cycle_complete',
    'calculate_cycle_crates',
    'summarize_protocol',
    'get_or_create_protocol',
    'detect_and_update_test_protocol'
]

