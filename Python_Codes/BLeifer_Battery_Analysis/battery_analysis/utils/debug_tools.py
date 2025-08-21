from mongoengine import get_connection
from mongoengine.connection import get_db
from mongoengine.base.common import _document_registry

from battery_analysis.utils.logging import get_logger

logger = get_logger(__name__)


def debug_connection_status():
    logger.info("\n🔍 [MongoEngine Debug Info]")

    # Check connection info
    try:
        conn = get_connection()
        logger.info("✅ Connected to: %s", conn.address)
        logger.info("🗃️  Databases: %s", conn.list_database_names())
    except Exception as e:
        logger.error("❌ Failed to connect: %s", e)
        return

    # Check current DB
    try:
        db = get_db()
        logger.info("📦 Current DB Name: %s", db.name)
        logger.info("📂 Collections in DB: %s", db.list_collection_names())
    except Exception as e:
        logger.error("❌ Failed to get DB info: %s", e)

    # Registered models
    logger.info("📘 Registered Models: %s", list(_document_registry.keys()))

    # Check actual counts from each collection if registered
    try:
        from battery_analysis.models import Sample, TestResult
        logger.info("📊 Sample count: %s", Sample.objects.count())
        logger.info("📊 TestResult count: %s", TestResult.objects.count())
    except Exception as e:
        logger.warning("⚠️  Model query failed: %s", e)

    # Enable verbose query logging
    mongo_logger = logging.getLogger('mongoengine')
    if not mongo_logger.hasHandlers():
        mongo_logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        mongo_logger.addHandler(handler)
        logger.info("🧪 Enabled mongoengine debug logging")
    else:
        logger.info("🧪 MongoEngine debug logging already active")

    logger.info("🔍 End debug info\n")


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
