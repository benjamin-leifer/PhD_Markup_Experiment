# Normalization CLI

The repository includes a small helper script, `normalize_cli.py`, that prints
capacity and impedance normalized using the utilities in
`normalization_utils.py`.

Run the script with one or more cell codes:

```bash
python normalize_cli.py CELL_A CELL_B
```

Each line of output lists the cell code followed by any available normalized
values. Example:

```
CELL_A capacity=1.500 impedance=0.005
CELL_B sample not found
```

When running the dashboard, the same normalized values are shown on the
**Advanced Analysis** tab whenever a sample is selected.
