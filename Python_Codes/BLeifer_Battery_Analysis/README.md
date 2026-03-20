# Battery Test Data Analysis Package

A modular and extensible Python package for electrochemical (battery) test data
management, supporting data ingestion, analysis, and reporting for multiple file formats.

## Features

- **Data Models**: MongoEngine document classes for Sample and TestResult data storage in MongoDB
- **Data Parsers**: Support for Arbin (.csv, .xlsx) and BioLogic (.mpr, .mpt) battery tester file formats
- **Analysis Logic**: Compute key metrics like capacity, retention, and coulombic efficiency
- **Inferred Property Propagation**: Automatically update parent samples based on child sample performance
- **Advanced Electrochemical Analysis**:
  - Differential capacity analysis (dQ/dV) for phase transition identification
  - Capacity fade modeling and cycle life prediction
  - Energy density and efficiency analysis
  - Anomaly detection in cycling data
  - Machine learning capabilities for test comparison and clustering
- **Electrochemical Impedance Spectroscopy (EIS)**:
  - Import and process EIS data from multiple formats (CSV, Gamry, BioLogic, Autolab)
  - Fit equivalent circuit models using impedance.py
  - Extract physical parameters from circuit fits (resistances, capacitances, time constants)
  - Generate publication-quality Nyquist and Bode plots
  - Distribution of Relaxation Times (DRT) analysis
- **Physics-Based Modeling with PyBAMM**:
  - Simulate battery behavior using physics-based models (SPM, DFN)
  - Compare simulations with experimental data
  - Fit model parameters to experimental measurements
  - Predict cycle life and degradation behavior
  - Explore parameter space for battery design optimization
- **Graphical User Interface**:
  - Intuitive data upload and management
  - Interactive plot visualization
  - MongoDB connection interface
  - Easy access to analysis and reporting features
- **Automated Reporting**: Generate PDF reports with performance metrics and visualizations
- **Modular Design**: Easily extendable to new file formats or analysis techniques

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/battery_analysis.git
cd battery_analysis

# Install the package
pip install .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"
```

### With Optional Features

```bash
# Install with BioLogic .mpr file support
pip install ".[biologic_mpr]"

# Install with EIS analysis capabilities
pip install ".[eis]"

# Install with PyBAMM modeling capabilities
pip install ".[pybamm]"

# Install all optional dependencies
pip install ".[all]"
```

The optional figure editing tools used in the GUI require a Qt binding
(`PyQt5` or `PySide2`). If these libraries are not installed the "Edit" button
on embedded plots will be hidden.

## Requirements

- Python 3.7+
- MongoDB (running locally or on a remote server)
- Dependencies: mongoengine, pandas, numpy, matplotlib, reportlab
- Advanced analysis features depend on `scipy` and `scikit-learn`.
Install them with:

```bash
pip install scipy scikit-learn
# or
pip install .[advanced]
```
You can also create an `advanced` extra in `setup.py` to bundle these packages.

## Usage

### Quick Start

```python
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
```

### Data Import

```python
# Scan a directory for all compatible files
file_list = utils.get_file_list('/path/to/data_directory')

# Process all files with automatic sample name extraction
# Use regex pattern to extract sample names from filenames
utils.batch_import_files(file_list, sample_name_pattern=r'cell_([A-Z0-9]+)_')
```

### Hierarchical Sample Management

```python
# Create a parent sample (e.g., a batch)
batch = models.Sample(name="Batch_A", chemistry="NMC")
batch.save()

# Create child samples that reference the parent
cell1 = models.Sample(name="Cell_A1", parent=batch, chemistry="NMC")
cell2 = models.Sample(name="Cell_A2", parent=batch, chemistry="NMC")
cell1.save()
cell2.save()

# When test results are added to child samples, the parent's
# properties will be automatically updated through property propagation
```

### Data Analysis

```python
# Get cycle data from a test
cycle_data = analysis.get_cycle_data(test_id)

# Compare multiple samples
comparison = analysis.compare_samples([sample1.id, sample2.id], metric='avg_capacity_retention')

# Update sample properties based on new test results
analysis.update_sample_properties(sample)
```

### Advanced Analysis

```python
# Perform differential capacity analysis (dQ/dV)
voltage, capacity = advanced_analysis.get_voltage_capacity_data(test_id, cycle_number=1)
v_centers, dq_dv = advanced_analysis.differential_capacity_analysis(voltage, capacity)

# Analyze capacity fade patterns and predict cycle life with custom EOL
fade_analysis = advanced_analysis.capacity_fade_analysis(test_id, eol_percent=75)
predicted_eol = fade_analysis['predicted_eol_cycle']  # Cycle where capacity reaches 75% of initial

# Detect anomalies in cycling data
anomalies = advanced_analysis.detect_anomalies(test_id, metric='discharge_capacity')

# Analyze energy efficiency and density
energy_metrics = advanced_analysis.energy_analysis(test_id)

# Cluster multiple tests based on performance metrics
clusters = advanced_analysis.cluster_tests([test1.id, test2.id, test3.id])

# Find tests similar to a reference test
similar_tests = advanced_analysis.find_similar_tests(reference_test.id)
```

### Electrochemical Impedance Spectroscopy

```python
# Import EIS data from a file
eis_data = eis.import_eis_data('my_eis_data.csv', sample_id=sample.id)

# Retrieve EIS data from database
eis_data = eis.get_eis_data(test_id)

# Create Nyquist plot
fig = eis.plot_nyquist(eis_data, highlight_frequencies=True)
fig.savefig('nyquist_plot.png')

# Fit equivalent circuit model (requires impedance.py)
circuit = "R0-p(R1,CPE1)-W2"  # Randles with CPE and Warburg
fit_result = eis.fit_circuit_model(
    eis_data['frequency'],
    eis_data['Z_real'],
    eis_data['Z_imag'],
    circuit
)

# Extract physical parameters from the fit
params = eis.extract_physical_parameters(fit_result)
print(f"Electrolyte resistance: {params['electrolyte_resistance']['value']} Ω")
print(f"Charge transfer resistance: {params['R1_resistance']['value']} Ω")

# Generate comprehensive EIS report
eis.generate_comprehensive_eis_report(eis_data, fit_result, "eis_report.pdf")
```

### Generating Reports

```python
# Generate a report for a single test
report.generate_report(test_result, filename="test_report.pdf")

# Generate a comparison report for multiple tests
report.generate_comparison_report([test1, test2, test3], filename="comparison_report.pdf")
```

## Directory Structure

```
battery_analysis/
├── __init__.py            # Package initialization
├── models.py              # MongoEngine document models
├── analysis.py            # Analysis functions
├── report.py              # PDF report generation
├── utils.py               # Utility functions
├── parsers/
│   ├── __init__.py        # Parser initialization
│   ├── arbin_parser.py    # Parser for Arbin files
│   └── biologic_parser.py # Parser for BioLogic files
├── examples/
│   └── example_usage.py   # Example script
└── tests/                 # Test scripts
```

## MongoDB Document Models

### Sample

The `Sample` document represents a physical battery sample or cell:

- **name**: Unique identifier for the sample
- **chemistry**: Battery chemistry (e.g., "Li-ion NMC")
- **form_factor**: Physical form (e.g., "18650", "pouch")
- **parent**: Reference to a parent sample (for hierarchical grouping)
- **tests**: List of references to TestResult documents
- **avg_initial_capacity**, **avg_final_capacity**, etc.: Aggregated metrics

### TestResult

The `TestResult` document represents a test performed on a sample:

- **sample**: Reference to the Sample
- **tester**: Equipment used (e.g., "Arbin", "BioLogic")
- **cycles**: List of embedded CycleSummary documents
- **cycle_count**, **initial_capacity**, etc.: Summary metrics

## Extending the Package

### Adding a New Parser

1. Create a new module in `parsers/` directory
2. Implement a parsing function with similar interface to existing parsers
3. Add the new format to `parsers/__init__.py`

### Adding New Analysis Metrics

1. Add new fields to the appropriate document models in `models.py`
2. Update `compute_metrics()` in `analysis.py` to calculate the new metric
3. Update `update_sample_properties()` to propagate the new metric

## Migration Notes

Use [`battery_analysis.utils.migrate_metadata_backfill`](battery_analysis/utils/migrate_metadata_backfill.py)
for the README backfill migration tasks. The utility performs two passes:

1. it finds `TestResult` documents missing `created_at` or `updated_at` and fills
   the missing timestamp fields deterministically from the test date, metadata,
   linked raw-file upload time, or finally the MongoDB object id timestamp; and
2. it scans `raw_data_files` and backfills missing `sample`, `operator`,
   `acquisition_device`, `tags`, and metadata keys when those values can be
   inferred from the linked `test_result`, stored file path, or existing
   metadata.

### How to run it

From `Python_Codes/BLeifer_Battery_Analysis` (or from any environment where the
package is installed in editable mode), use the module entry point:

```bash
cd Python_Codes/BLeifer_Battery_Analysis
python -m battery_analysis.utils.migrate_metadata_backfill --dry-run
python -m battery_analysis.utils.migrate_metadata_backfill
```

Add `--json` if you want machine-readable counts for shell scripts or CI logs:

```bash
python -m battery_analysis.utils.migrate_metadata_backfill --dry-run --json
```

### Dry-run behavior

`--dry-run` never saves any documents. It evaluates the same matching and
backfill logic as a real run, reports how many `TestResult` and `RawDataFile`
documents would change, and exits without writing updates.

### Verifying counts before and after

A practical verification flow is:

1. Run a dry-run before the migration to capture the candidate counts.
2. Run the migration without `--dry-run`.
3. Run the dry-run again. An idempotent, fully applied migration should now
   report `changed: 0` for both `test_results` and `raw_data_files`.

Example:

```bash
python -m battery_analysis.utils.migrate_metadata_backfill --dry-run --json
python -m battery_analysis.utils.migrate_metadata_backfill --json
python -m battery_analysis.utils.migrate_metadata_backfill --dry-run --json
```

### Idempotence

Yes. The migration is designed to be idempotent:

- missing `TestResult` timestamps are derived from stable document data, so the
  same document gets the same backfilled value every time; and
- `RawDataFile` updates only add missing derivable fields or normalize merged
  tags/metadata into the same final shape, so re-running the migration should
  not produce additional writes after the first successful apply.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This package was inspired by the need for standardized battery data management
- Thanks to all contributors and the battery research community
