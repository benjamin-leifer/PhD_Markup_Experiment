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

Future components should follow this pattern to maintain a consistent document
flow through the manufacturing process.
