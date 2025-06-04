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

### Development setup

Install the dependencies needed for running the tests and style checks:

```bash
pip install -r requirements-dev.txt
```

The package requires a running MongoDB instance for data storage.

## License

This project is licensed under the [MIT License](LICENSE).
