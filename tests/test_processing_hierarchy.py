
from battery_analysis import models
import pytest
from mongoengine import ValidationError


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


def test_cell_code_fallback_from_name():
    sample = models.Sample(name="S1")
    tr = models.TestResult(sample=sample, tester="Arbin", name="run_CN9_data")
    tr.clean()
    assert tr.cell_code == "CN9"


def test_cell_code_fallback_missing_pattern():
    sample = models.Sample(name="S1")
    tr = models.TestResult(sample=sample, tester="Arbin", name="no_code_here")
    tr.clean()
    assert tr.cell_code is None


def test_child_metadata_overrides_parent():
    mat = models.CathodeMaterial(name="M1", metadata={"key": "parent"})
    slurry = models.Slurry.from_parent(mat, metadata={"key": "child"})
    assert slurry.metadata["key"] == "child"


def test_from_parent_sets_parent_chain():
    mat = models.CathodeMaterial(name="M1")
    slurry = models.Slurry.from_parent(mat)
    electrode = models.Electrode.from_parent(slurry)
    cell = models.Cell.from_parent(electrode)
    sample = models.Sample(name="S1")
    test = models.TestResult.from_parent(cell, sample=sample, tester="Arbin")
    test.clean()
    assert slurry.parent == mat
    assert electrode.parent == slurry
    assert cell.parent == electrode
    assert test.parent == cell


def test_missing_parent_raises_error():
    with pytest.raises(ValidationError, match="parent"):
        models.Slurry.from_parent(None).validate()
