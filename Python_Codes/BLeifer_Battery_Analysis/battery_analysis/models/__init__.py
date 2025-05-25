# battery_analysis/models/__init__.py
from mongoengine import Document, EmbeddedDocument, fields, CASCADE
from mongoengine.fields import FileField

# First import CycleSummary since it has no dependencies
from .cycle_summary import CycleSummary

# Next import Sample and TestResult
# The import order matters but the CLASS is what gets registered
from .sample import Sample
from .test_result import TestResult, CycleDetailData
from .raw_file import RawDataFile

# Print registration status
from mongoengine.base.common import _document_registry
print("Registered models after initial import:", _document_registry.keys())

# Now set up delete rules
from .finalize import add_delete_rules
add_delete_rules()

# For extra safety, print final registration state
print("Final registered models:", _document_registry.keys())

# Export the classes
__all__ = ["Sample", "TestResult", "CycleSummary", "RawDataFile", "CycleDetailData"]

from mongoengine.fields import LazyReferenceField, ReferenceField

for attr_name, field in Model._fields.items():
    if isinstance(field, (ReferenceField, LazyReferenceField)):
        column_name = field.db_field            # this is the actual DB key
    else:
        column_name = field.field               # for normal fields
    # …then use column_name when you wire up your rules…
