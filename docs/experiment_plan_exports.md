# Experiment Plan Exports

The DOE builder supports exporting experiment plans to CSV, PDF and HTML formats. See [sample_plan.pdf](sample_plan.pdf) for an example PDF export. HTML exports produce a sortable and filterable table and are written to `docs/doe_plans/` by default.

```bash
python -m battery_analysis.utils.doe_builder \
    --name demo --factors '{"A":[1,2]}' --html
```

The above command creates `docs/doe_plans/demo.html` which can be easily shared or hosted as part of the documentation site.
