# Model hierarchy

The manufacturing workflow for a cell is represented as a chain of document models:

1. **CathodeMaterial** – base powders or active material.
2. **Slurry** – mixture created from a `CathodeMaterial`.
3. **Electrode** – dried coating produced from a `Slurry`.
4. **Cell** – assembled cell using an `Electrode`.
5. **TestResult** – electrochemical data referencing the `Cell` under test.

Each model contains a `parent` field pointing to the previous stage. Metadata
stored in the `metadata` field is inherited down this chain using the
`inherit_metadata()` helper, which recursively merges metadata dictionaries so
that child values override those of their ancestors. Creation helper methods
(`from_parent`) automatically set the parent and pre-populate metadata.

## Creating objects with inheritance

Use the ``from_parent`` class methods when instantiating new stage objects. The
helper assigns the ``parent`` reference and immediately merges metadata from the
parent chain:

```python
from battery_analysis.models import CathodeMaterial, Slurry

mat = CathodeMaterial(name="M1", metadata={"material": "NMC"})
slurry = Slurry.from_parent(mat, metadata={"slurry": "S"})
```

When an object is saved or cleaned, :func:`inherit_metadata` recalculates the
merged metadata so any later changes to ancestors are reflected in descendants.
The utility can also be called directly on an object to retrieve the fully
resolved metadata dictionary:

```python
from battery_analysis.models import inherit_metadata

combined = inherit_metadata(slurry)
```

Future components should follow this pattern to maintain a consistent document
flow through the manufacturing process.
