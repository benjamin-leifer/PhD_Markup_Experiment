# formulation.py
from datetime import datetime
from database_object import DatabaseObject
from material import Material


class Formulation(DatabaseObject):
    db_name = "inventory"
    collection_name = "formulations"

    def __init__(self, name, date=None, materials=None, formulation_id=None):
        super().__init__(db_name="inventory", collection_name="formulations")
        self.name = name
        self.date = date or datetime.now().strftime("%Y-%m-%d")
        self.materials = materials or {}  # Dictionary with Material instances and quantities
        self.formulation_id = formulation_id or self.generate_formulation_id()

    def generate_formulation_id(self):
        """Generate a unique formulation ID based on name and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{self.name}_{timestamp}"

    def to_dict(self):
        # Convert materials to a dictionary of material lot numbers and quantities
        materials_dict = {material.lot_number: quantity for material, quantity in self.materials.items()}
        return {
            "name": self.name,
            "date": self.date,
            "materials": materials_dict,
            "formulation_id": self.formulation_id
        }

    def from_dict(self, document):
        """Load formulation details from a MongoDB document."""
        self.name = document["name"]
        self.date = document["date"]
        self.formulation_id = document["formulation_id"]

        # Load materials from MongoDB using the lot numbers provided in the document
        self.materials = {}
        for lot_number, qty in document["materials"].items():
            material = Material.load_from_db({"lot_number": lot_number})  # Load Material by lot_number
            if material:
                self.materials[material] = qty

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
