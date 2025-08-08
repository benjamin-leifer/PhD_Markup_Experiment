## Dashboard Module

Selecting a cell in the dashboard now automatically loads its merged cycle
data. When a cell code is chosen, the application first looks for the
`default_dataset` linked to the corresponding `Sample`. If the sample does not
yet have a dataset, the new `get_cell_dataset` helper builds one on the fly by
aggregating all available `TestResult` records.

This behaviour ensures that plots and comparison views use the most complete
cycle information for each cell without requiring any manual preprocessing.

