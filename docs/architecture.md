# Architecture and Stage Models

This project models the manufacturing workflow of a battery cathode through a
series of MongoEngine document models. Each stage points to its precursor via a
`parent` field and shares metadata through a simple inheritance helper. The
chain mirrors the physical process and allows downstream data such as
`TestResult` documents to carry context from the original material.

```mermaid
graph TD
    CM[CathodeMaterial]
    S[Slurry]
    E[Electrode]
    C[Cell]
    T[TestResult]
    CM --> S --> E --> C --> T
```

### Model overview

| Stage           | Parent          | Example fields                      |
|-----------------|-----------------|-------------------------------------|
| CathodeMaterial | –               | `composition`, `manufacturer`       |
| Slurry          | CathodeMaterial | `solids_content`, `mixing_time`     |
| Electrode       | Slurry          | `loading`, `thickness`              |
| Cell            | Electrode       | `format`, `nominal_capacity`        |
| TestResult      | Cell            | `tester`, `test_type`               |

## Metadata inheritance

Every model exposes a `metadata` dictionary. The
[`inherit_metadata`](../Python_Codes/BLeifer_Battery_Analysis/battery_analysis/models/stages.py)
helper walks up the `parent` chain and merges these dictionaries so that child
values override those of their ancestors. Each model's `clean()` method invokes
this helper, so calling `save()` automatically fills in any missing metadata
from its ancestors. The `from_parent` constructors additionally pre-populate the
merged metadata when instantiating a child from its parent.

### Example propagation

| Stage            | Parent          | Metadata provided                    | Result after inheritance                                       |
|------------------|-----------------|--------------------------------------|----------------------------------------------------------------|
| CathodeMaterial  | –               | `{"material":"NMC"}`              | `{"material":"NMC"}`                                       |
| Slurry           | CathodeMaterial | `{"solids":45}`                   | `{"material":"NMC","solids":45}`                        |
| Electrode        | Slurry          | `{"loading":5}`                   | `{"material":"NMC","solids":45,"loading":5}`         |
| Cell             | Electrode       | `{"format":"2032"}`              | `{"material":"NMC","solids":45,"loading":5,"format":"2032"}` |
| TestResult       | Cell            | `{"tester":"Arbin"}`             | `{"material":"NMC","solids":45,"loading":5,"format":"2032","tester":"Arbin"}` |

## Creating entities

Create objects using the provided class methods so references and metadata are
wired correctly. Metadata values from ancestors are filled down automatically:

1. **CathodeMaterial** – starting point for metadata.
   ```python
   from battery_analysis.models import CathodeMaterial
   mat = CathodeMaterial(name="M1", metadata={"material": "NMC"})
   mat.save()
   ```
2. **Slurry** – derived from a material.
   ```python
   from battery_analysis.models import Slurry
   slurry = Slurry.from_parent(mat, solids_content=45)
   slurry.save()
   ```
3. **Electrode** – derived from slurry.
   ```python
   from battery_analysis.models import Electrode
   elec = Electrode.from_parent(slurry, loading=5)
   elec.save()
   ```
4. **Cell** – assembled from electrode.
   ```python
   from battery_analysis.models import Cell
   cell = Cell.from_parent(elec, format="2032")
   cell.save()
   ```
5. **TestResult** – references the cell. Saving triggers `clean()` which
   pulls metadata from the cell and all earlier stages.
   ```python
   from battery_analysis.models import TestResult
   test = TestResult(sample=..., parent=cell, tester="Arbin")
   test.save()  # metadata inherited automatically
   ```

Following this sequence ensures that each entity is created in the correct
order and that all downstream documents receive the accumulated metadata from
previous stages.
