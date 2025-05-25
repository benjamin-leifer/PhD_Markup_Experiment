# electrolyte.py
from database_object import DatabaseObject
from formulation import Formulation
from database_object import DatabaseObject

class Electrolyte(DatabaseObject):
    db_name = "inventory"
    collection_name = "formulations"

    def __init__(self, formulation, **kwargs):
        super().__init__(db_name="electrochemical_cells", collection_name="electrolytes")
        self.formulation = formulation  # Instance of Formulation
        # You can add other properties here as needed, like conductivity, if not in Formulation

    def to_dict(self):
        # Save the formulation ID instead of the full formulation details to avoid redundancy
        return {
            "formulation_id": self.formulation.formulation_id,
            # Add other Electrolyte-specific fields if needed
        }

    def from_dict(self, document):
        # Load the formulation using the formulation ID from MongoDB
        self.formulation = Formulation.load_from_db({"formulation_id": document["formulation_id"]})
        # Populate other fields if they exist in Electrolyte

    def add_gui_components(self, root):
        from tkinter import Label
        Label(root, text=f"Electrolyte Formulation: {self.formulation.name}").pack()
        Label(root, text=f"Date: {self.formulation.date}").pack()
        for material, qty in self.formulation.materials.items():
            Label(root, text=f"{material.name} - {qty} {material.unit}").pack()
