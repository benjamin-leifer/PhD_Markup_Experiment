## Dashboard Module

Selecting a cell in the dashboard now prefers the combined cycle dataset. When
a cell code is chosen, the application first checks for the `default_dataset`
linked to the corresponding `Sample` and otherwise uses the
`get_cell_dataset` helper to retrieve a matching `CellDataset`. If no combined
dataset is available, the dashboard falls back to the raw `TestResult`
records, ensuring the views remain populated even for cells without a
precomputed dataset.

### MongoDB configuration

Runtime setup now loads the user configuration via
`battery_analysis.utils.config.load_config` and uses the resulting
`db_uri` to populate the `MONGO_URI` environment variable if it is not
already set. The dashboard first attempts to connect to a real MongoDB
instance using `MONGO_URI`, `MONGO_HOST` and `MONGO_PORT`. When that
connection fails (as in tests or remote deployments without a database)
it automatically falls back to an in-memory `mongomock` database so the
interface remains functional. The older `BATTERY_DB_*` variables are no
longer populated by default.

### User selection

The **Current User** dropdown now retrieves its options from
`battery_analysis.user_tracking`. The chosen username is stored in the
browser's local storage so that reloading the application restores the
previous selection automatically.

### Ad hoc scripts

Prototype analyses belong in the `adhoc_scripts` directory. Each script should
define a `layout()` function that returns Dash components. Files placed here are
automatically listed in the "Ad hoc Analysis" tab so they can be loaded without
modifying the core dashboard code.

### Promotion to core modules

When a script becomes stable and broadly useful, promote it to its own tab
module. A good candidate for promotion is one that is maintained, used across
projects, or fills a long-term need.

To promote a script:

1. Move it from `adhoc_scripts` into the main `dashboard` package and give it a
   descriptive filename ending in `_tab.py`.
2. Ensure the new module exposes `layout()` and `register_callbacks()` functions
   just like the existing tab modules.
3. Add an import for the module in `dashboard/app.py` and call its
   `register_callbacks` function.
4. Remove the old script from `adhoc_scripts` if no longer needed.

These steps provide a clear path from experimentation to fully integrated
dashboard features.

### Interactive tables

The **Running Tests** and **Upcoming Tests** views now use Dash's
`DataTable` component. Columns can be sorted by clicking the headers and
filtered using the built-in filter boxes. Each row retains the flag dropdown
for marking samples for review or retest, making the tables more powerful while
remaining familiar to existing users.

### Upload progress indicator

The Data Import tab now wraps the upload and metadata form in a loading
spinner. When a file is selected, a "Parsing file..." message appears until the
upload is processed so users receive immediate feedback during long-running
parsing operations.

### Toast notifications

Success and error messages are now delivered through Bootstrap toast
notifications. These messages appear in the upper-right corner of the screen
with an appropriate header and icon, then automatically dismiss after a short
interval.

