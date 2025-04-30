"""
Utility functions for the battery analysis package.
"""
import logging
from mongoengine import connect
# ---------------------------------------------------------------------------
# Database helper (robust connector)
# ---------------------------------------------------------------------------
from .db import connect_with_fallback as connect_to_database


def connect_to_database(db_name="battery_test_db", host="localhost", port=27017):
    """
    Connect to MongoDB database.

    Args:
        db_name: Name of the database
        host: MongoDB host address
        port: MongoDB port

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        # Attempt to connect to the database
        conn = connect(db_name, host=host, port=port)

        # Test connection by accessing server info
        server_info = conn.server_info()

        logging.info(f"Connected to MongoDB {db_name} at {host}:{port}")
        logging.info(f"MongoDB version: {server_info.get('version', 'unknown')}")

        return True
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {str(e)}")
        return False


def debug_connection_status():
    """Print MongoDB connection status for debugging."""
    try:
        from mongoengine.connection import get_connection, get_db
        conn = get_connection()
        if conn:
            print(f"MongoDB connection exists: {conn}")
            db = get_db()
            print(f"Current database: {db.name}")
            return True
        else:
            print("No MongoDB connection exists")
            return False
    except Exception as e:
        print(f"Error checking MongoDB connection: {str(e)}")
        return False


# Create other utility functions as needed
def get_file_list(directory):
    """
    Get list of supported data files in a directory.

    Args:
        directory: Path to directory to search

    Returns:
        list: List of file paths
    """
    import os
    from battery_analysis.parsers import get_supported_formats

    formats = get_supported_formats()
    files = []

    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in formats:
                files.append(os.path.join(root, filename))

    return files