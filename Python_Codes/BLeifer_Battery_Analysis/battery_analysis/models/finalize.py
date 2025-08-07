# battery_analysis/models/finalize.py


def add_delete_rules():
    """Add delete rules after models are registered."""
    # Import the necessary components
    from mongoengine import CASCADE
    from mongoengine.base.common import _document_registry

    # Check if models are registered
    print("Models currently registered:", _document_registry.keys())

    # Import our models
    try:
        from .sample import Sample
        from .testresult import TestResult
    except ImportError:  # pragma: no cover - allow running as script
        import importlib

        Sample = importlib.import_module("sample").Sample
        TestResult = importlib.import_module("testresult").TestResult

    # Check registry again
    print("Models after import:", _document_registry.keys())

    try:
        # We need a direct way to add delete rules
        # This is a bit of a hack but it should work
        TestResult.sample.field.reverse_delete_rule = CASCADE
        print("Added CASCADE for TestResult.sample")
    except Exception as e:
        print(f"Error setting TestResult.sample rule: {e}")

    try:
        # Setup the other direction
        Sample.tests.field.field.reverse_delete_rule = CASCADE
        print("Added CASCADE for Sample.tests")
    except Exception as e:
        print(f"Error setting Sample.tests rule: {e}")

    return True
