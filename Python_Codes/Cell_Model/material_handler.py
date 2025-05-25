# material_handler.py
from material import Material
from database_object import DatabaseObject

class MaterialHandler:
    def __init__(self):
        self.materials = []

    def add_material(self, name, lot_number, quantity, unit, manufacture_date=None):
        """Add a new material and save it to the database."""
        material = Material(name=name, lot_number=lot_number, quantity=quantity, unit=unit, manufacture_date=manufacture_date)
        material.save_to_db()  # This should now work as expected
        self.materials.append(material)
        print(f"Added {name} to inventory.")

    def get_material(self, lot_number):
        """Retrieve a material by lot number."""
        return Material.load_from_db({"lot_number": lot_number})

    def view_inventory(self):
        for material in self.materials:
            print(f"{material.name} (Lot: {material.lot_number}) - {material.quantity} {material.unit}")


