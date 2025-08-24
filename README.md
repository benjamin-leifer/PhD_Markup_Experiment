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
pre-commit install
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
* `USE_MONGO_MOCK` – set to `1` to use an in-memory mock database (via
  `mongomock`) instead of connecting to a real MongoDB server. This is useful
  for running the test suite without MongoDB.

Both the dashboard and the `update_cell_dataset_cli.py` script rely on these
variables when establishing database connections.

### User configuration

Command line utilities such as `import_directory` and `import_watcher` also
consult an optional user configuration file. Create either
`~/.battery_analysis.toml` or `~/.battery_analysis.ini` to provide default values
for database connection and common command line flags. Example TOML file::

    db_uri = "mongodb://localhost:27017/battery_test_db"
    workers = 4
    include = ["*.csv"]
    exclude = ["*/archive/*"]
    debounce = 1.0
    depth = 2

Values supplied on the command line always override settings from the
configuration file.

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

## Normalizing capacity and impedance

Use the ``normalize_cli.py`` script to report capacity normalized by area and
impedance normalized by thickness for one or more cell codes::

    python normalize_cli.py CN123

The script prints a line for each supplied code with the available metrics.
These values are also displayed in the dashboard on the **Advanced Analysis**
tab when a sample is selected.

## Importing data from directories

Use the ``battery_analysis.utils.import_directory`` module to scan a directory
tree for supported files and import them into the database. The ``--include``
and ``--exclude`` options accept glob patterns that filter directories or
filenames::

    python -m battery_analysis.utils.import_directory data --include "*.csv" --exclude "*/archive/*"

Repeat the options to provide multiple patterns. Any file whose path matches an
``--exclude`` pattern is skipped, while ``--include`` restricts processing to
paths that match at least one provided pattern.

Each file processed by ``import_directory`` is archived to MongoDB's GridFS so
the original data can be retrieved later. Specify ``--no-archive`` to disable
this behaviour when storing the raw files is unnecessary.

Previous import runs are stored in ``ImportJob`` records. Use ``--status`` to
list recent jobs and their completion state::

    python -m battery_analysis.utils.import_directory --status

Provide a job identifier to show a single record. Interrupted jobs can be
resumed by passing the identifier with ``--resume``::

    python -m battery_analysis.utils.import_directory data --resume <JOB_ID>

## Downloading raw data files

Raw files saved to GridFS can be retrieved with the
``battery_analysis.utils.raw_file_cli`` module.

- Download directly by ``RawDataFile`` identifier::

    python -m battery_analysis.utils.raw_file_cli download <FILE_ID> --out data.bin

- Fetch the file associated with a ``TestResult``::

    python -m battery_analysis.utils.raw_file_cli by-test <TEST_ID> --out data.bin

Omit ``--out`` to stream the file contents to standard output.

## Searching test results

Use the ``battery_analysis.utils.search_tests`` module to look up
``TestResult`` documents using basic filters. Provide a sample name,
chemistry, or a date range (``YYYY-MM-DD:YYYY-MM-DD``)::

    # all tests for a particular sample
    python -m battery_analysis.utils.search_tests --sample S1

    # all NMC tests from the first half of 2024
    python -m battery_analysis.utils.search_tests --chemistry NMC --date-range 2024-01-01:2024-06-30

The command prints a table containing the matching test identifiers and a
few key metrics.

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

## Migration Notes

Existing databases should backfill the newly added `created_at` and
`updated_at` fields on `TestResult` documents. A simple approach is to run a
database update that sets both fields to the current time for documents where
they are missing.

Raw data files now support additional metadata. Existing `raw_data_files`
documents can optionally be updated to link to their corresponding `Sample`
via the new `sample` field and populate auxiliary information such as
`operator`, `acquisition_device`, and free-form `tags` or `metadata` as needed.
