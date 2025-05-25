# formulation.py
from datetime import datetime
from database_object import DatabaseObject
from material import Material


class Formulation(DatabaseObject):
    db_name = "inventory"
    collection_name = "formulations"

    def __init__(self, name, date=None, materials=None, formulation_id=None):
        super().__init__(self.db_name, self.collection_name)
        self.name = name
        self.date = date or datetime.now().strftime("%Y-%m-%d")
        self.materials = materials or {}
        self.formulation_id = formulation_id or self.generate_formulation_id()

    def generate_formulation_id(self):
        """Generate a unique formulation ID based on name and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{self.name}_{timestamp}"

    def to_dict(self):
        return {
            "name": self.name,
            "date": self.date,
            "materials": {material.lot_number: qty for material, qty in self.materials.items()},
            "formulation_id": self.formulation_id
        }

    def from_dict(self, document):
        self.name = document["name"]
        self.date = document["date"]
        self.formulation_id = document["formulation_id"]
        self.materials = {Material.load_from_db({"lot_number": lot}): qty for lot, qty in document["materials"].items()}

    def add_gui_components(self, root):
        from tkinter import Label
        Label(root, text=f"Formulation: {self.name}").pack()
        Label(root, text=f"Date: {self.date}").pack()
        Label(root, text=f"Formulation ID: {self.formulation_id}").pack()
        for material, qty in self.materials.items():
            Label(root, text=f"Material: {material.name} - {qty} {material.unit}").pack()

    def add_material(self, material, quantity):
        """Add a material to the formulation with a specific quantity."""
        self.materials[material] = quantity
        material.update_quantity(-quantity)  # Deduct quantity from inventory

    def track_usage(self):
        """Store a record of materials used in this formulation."""
        for material, quantity in self.materials.items():
            usage_record = {
                "formulation_id": self.formulation_id,
                "material_lot": material.lot_number,
                "quantity_used": quantity,
                "date": self.date
            }
            # You could add this to a "usage" collection or a dedicated log
            self.collection.insert_one(usage_record)
