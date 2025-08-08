## Dashboard Module

Selecting a cell in the dashboard now prefers the combined cycle dataset. When
a cell code is chosen, the application first checks for the `default_dataset`
linked to the corresponding `Sample` and otherwise uses the
`get_cell_dataset` helper to retrieve a matching `CellDataset`. If no combined
dataset is available, the dashboard falls back to the raw `TestResult`
records, ensuring the views remain populated even for cells without a
precomputed dataset.

