# PhD_Markup_Experiment
This is my trial of notekeeping on Github. Let's see how it works out

## Installation

The repository includes the battery analysis package in
`Python_Codes/BLeifer_Battery_Analysis`. To install the required
dependencies, navigate to that folder and run:

```bash
pip install .
```

Additional optional features are available. Install them by passing one
of the extras:

```bash
pip install ".[biologic_mpr]"  # BioLogic .mpr file support
pip install ".[eis]"          # EIS tools
pip install ".[pybamm]"       # PyBAMM modeling
pip install ".[all]"          # everything
```
Advanced analysis functions depend on `scipy` and `scikit-learn`.
Install them with:

```bash
pip install scipy scikit-learn
# or
pip install .[advanced]
```
You can also create an `advanced` extra in `Python_Codes/BLeifer_Battery_Analysis/setup.py` to bundle these packages.

### Development setup

Install the dependencies needed for running the tests and style checks:

```bash
pip install -r requirements-dev.txt
pip install -e Python_Codes/BLeifer_Battery_Analysis
```

### Running tests

Make sure the runtime dependencies are installed before executing the test
suite. You can install them using the provided requirements files:

```bash
pip install -r requirements.txt        # core package requirements
pip install -r requirements-dev.txt    # testing and style tools
pip install -e Python_Codes/BLeifer_Battery_Analysis
```

Once the dependencies are available, run the tests from the repository root:

```bash
pytest -q
```

The package requires a running MongoDB instance for data storage.

## License

This project is licensed under the [MIT License](LICENSE).

## Dashboard

A lightweight Dash application provides a simple interface for monitoring battery tests.
The required dependencies, including `dash` and `dash-bootstrap-components`, are listed in
`requirements.txt`.

To run the dashboard locally, execute:

```bash
python -m dashboard
```

The app currently uses placeholder data but is structured for MongoDB integration
via the functions in `dashboard/data_access.py`.
