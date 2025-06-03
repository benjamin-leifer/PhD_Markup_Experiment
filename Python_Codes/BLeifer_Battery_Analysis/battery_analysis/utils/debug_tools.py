import logging
from mongoengine import get_connection
from mongoengine.connection import get_db
from mongoengine.base.common import _document_registry

def debug_connection_status():
    print("\n🔍 [MongoEngine Debug Info]")

    # Check connection info
    try:
        conn = get_connection()
        print("✅ Connected to:", conn.address)
        print("🗃️  Databases:", conn.list_database_names())
    except Exception as e:
        print("❌ Failed to connect:", e)
        return

    # Check current DB
    try:
        db = get_db()
        print("📦 Current DB Name:", db.name)
        print("📂 Collections in DB:", db.list_collection_names())
    except Exception as e:
        print("❌ Failed to get DB info:", e)

    # Registered models
    print("📘 Registered Models:", list(_document_registry.keys()))

    # Check actual counts from each collection if registered
    try:
        from battery_analysis.models import Sample, TestResult
        print("📊 Sample count:", Sample.objects.count())
        print("📊 TestResult count:", TestResult.objects.count())
    except Exception as e:
        print("⚠️  Model query failed:", e)

    # Enable verbose query logging
    logger = logging.getLogger('mongoengine')
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        print("🧪 Enabled mongoengine debug logging")
    else:
        print("🧪 MongoEngine debug logging already active")

    print("🔍 End debug info\n")


def ensure_models_registered():
    """Ensure all models are registered with MongoEngine."""
    from battery_analysis.models import Sample, TestResult, CycleSummary
    from mongoengine import register_document

    for cls in [Sample, TestResult, CycleSummary]:
        try:
            register_document(cls)
        except:
            pass  # Already registered

    return True
