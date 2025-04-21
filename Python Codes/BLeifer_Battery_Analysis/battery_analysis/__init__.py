"""
Battery Test Data Analysis Package

A modular and extensible Python package for electrochemical (battery) test data
management, supporting data ingestion, analysis, and reporting for multiple file formats.

Main components:
- Data Models: MongoEngine document classes for samples and test results
- Data Parsers: Parsers for Arbin and BioLogic file formats
- Analysis: Functions to compute metrics and propagate properties
- Advanced Analysis: Advanced electrochemical techniques like dQ/dV, capacity fade modeling, and anomaly detection
- EIS: Electrochemical Impedance Spectroscopy analysis and modeling
- Reporting: PDF report generation for test results

Basic Usage:
    from battery_analysis import utils, models, analysis
    from battery_analysis.parsers import parse_file

    # Connect to MongoDB
    utils.connect_to_database('battery_db')

    # Create a sample
    sample = models.Sample(name="Sample_123", chemistry="NMC")
    sample.save()

    # Parse a test file
    cycles, metadata = parse_file('path/to/test_file.csv')

    # Create test result and attach to sample
    test = analysis.create_test_result(sample, cycles, tester="Arbin", metadata=metadata)

    # Generate a report
    from battery_analysis import report
    report.generate_report(test, filename="test_report.pdf")

    # Perform advanced analysis
    from battery_analysis import advanced_analysis
    fade_analysis = advanced_analysis.capacity_fade_analysis(test.id)
    anomalies = advanced_analysis.detect_anomalies(test.id)

    # Analyze EIS data
    from battery_analysis import eis
    eis_data = eis.import_eis_data('eis_file.csv', sample_id=sample.id)
    eis.plot_nyquist(eis_data, filename="nyquist.png")
"""

# Package version
__version__ = '0.1.0'

# Import main components for easier access
from . import models
from . import analysis
from . import report
from . import utils
from . import advanced_analysis
from . import eis

# Make parsers accessible
from .parsers import parse_file