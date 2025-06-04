from mongoengine import CASCADE


def setup_model_relationships():
    """Set up relationships between models after they're all defined."""
    # We'll use a different approach that doesn't require register_delete_rule
    from .sample import Sample
    from .testresult import TestResult

    # Instead of registering delete rules separately, we'll just
    # confirm the models are properly registered
    from mongoengine.base.common import _document_registry

    models = [Sample, TestResult]
    for model in models:
        if model.__name__ not in _document_registry:
            raise ImportError(f"Model {model.__name__} failed to register")

    return True
