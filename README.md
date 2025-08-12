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

Connection details are configured through environment variables:

* `MONGO_URI` – full MongoDB connection string. When provided it takes precedence.
* `MONGO_HOST` and `MONGO_PORT` – host and port values used when `MONGO_URI` is unset.
* `BATTERY_DB_NAME` – optional database name (defaults to `battery_test_db`).

Both the dashboard and the `update_cell_dataset_cli.py` script rely on these
variables when establishing database connections.

## Updating cell datasets

Use the ``update_cell_dataset_cli.py`` script to keep processed datasets in
sync with newly ingested ``TestResult`` documents.

- Refresh a single cell's dataset::

    python update_cell_dataset_cli.py --cell CN123

- Refresh datasets for all cells::

    python update_cell_dataset_cli.py --all

- Show the number of distinct cell codes without updating::

    python update_cell_dataset_cli.py --count

After new data are added for a cell, run ``update_cell_dataset_cli.py --cell
<CODE>`` to rebuild its dataset. This command can also be scheduled in a CI job
or a cron task to ensure datasets remain current.

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

### Ad hoc script workflow

Early experiments and one-off analyses can be dropped into
`dashboard/adhoc_scripts`. Each file should export a `layout()` function that
returns Dash components so the script can be loaded through the "Ad hoc
Analysis" tab.

Scripts that prove useful for regular work can be promoted to a dedicated tab in
the dashboard. Consider promoting a script when it:

* is stable and maintained,
* is useful to multiple users or sessions, and
* fits naturally alongside the existing dashboard views.

To promote a script:

1. Move or rename the file from `dashboard/adhoc_scripts` to the `dashboard`
   package, giving it a descriptive name such as `my_feature_tab.py`.
2. Ensure the module defines `layout()` and `register_callbacks()` functions.
3. Update `dashboard/app.py` to import the new module and call its
   `register_callbacks` function.
4. Remove the old ad hoc script if it is no longer needed.

Following this process keeps experimental ideas available while providing a
clear path for integrating mature tools into the main dashboard.
