## Dashboard Module

Selecting a cell in the dashboard now prefers the combined cycle dataset. When
a cell code is chosen, the application first checks for the `default_dataset`
linked to the corresponding `Sample` and otherwise uses the
`get_cell_dataset` helper to retrieve a matching `CellDataset`. If no combined
dataset is available, the dashboard falls back to the raw `TestResult`
records, ensuring the views remain populated even for cells without a
precomputed dataset.

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

