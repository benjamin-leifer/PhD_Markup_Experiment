# material.py
from database_object import DatabaseObject

class Material(DatabaseObject):
    db_name = "inventory"  # Set the database name at the class level
    collection_name = "materials"  # Set the collection name at the class level

    def __init__(self, name, lot_number, quantity, unit, manufacture_date=None):
        super().__init__(self.db_name, self.collection_name)  # Pass class-level attributes
        self.name = name  # Material name, e.g., "LiTFSI"
        self.lot_number = lot_number  # Lot number for tracking
        self.quantity = quantity  # Quantity available
        self.unit = unit  # Unit of measure, e.g., "g", "mL"
        self.manufacture_date = manufacture_date

    def to_dict(self):
        """Convert Material attributes to a dictionary for MongoDB storage."""
        return {
            "name": self.name,
            "lot_number": self.lot_number,
            "quantity": self.quantity,
            "unit": self.unit,
            "manufacture_date": self.manufacture_date
        }

    def from_dict(self, document):
        """Populate Material attributes from a MongoDB document."""
        self.name = document["name"]
        self.lot_number = document["lot_number"]
        self.quantity = document["quantity"]
        self.unit = document["unit"]
        self.manufacture_date = document["manufacture_date"]

    def update_quantity(self, amount_used):
        """Deduct or add quantity based on usage or restocking."""
        self.quantity -= amount_used
        self.save_to_db()  # Save updated quantity to the database
