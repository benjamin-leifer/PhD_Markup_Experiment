
from battery_analysis import models


def test_metadata_inheritance_chain():
    mat = models.CathodeMaterial(name="M1", metadata={"material": "NMC"})
    mat.clean()
    slurry = models.Slurry.from_parent(mat, metadata={"slurry": "S"})
    electrode = models.Electrode.from_parent(slurry, metadata={"electrode": "E"})
    cell = models.Cell.from_parent(electrode, metadata={"cell": "C"})
    sample = models.Sample(name="S1")
    test = models.TestResult.from_parent(
        cell, sample=sample, metadata={"test": "T"}, tester="Arbin"
    )
    test.clean()
    assert test.metadata == {
        "material": "NMC",
        "slurry": "S",
        "electrode": "E",
        "cell": "C",
        "test": "T",
    }
