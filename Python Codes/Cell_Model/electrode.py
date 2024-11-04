# electrode.py
from database_object import DatabaseObject
from tkinter import Tk, Label, Button, Entry

class Electrode(DatabaseObject):
    db_name = "inventory"
    collection_name = "formulations"

    def __init__(self, material, capacity, areal_capacity, surface_area, thickness):
        super().__init__(db_name="electrochemical_cells", collection_name="electrodes")
        self.material = material
        self.capacity = capacity
        self.areal_capacity = areal_capacity
        self.surface_area = surface_area
        self.thickness = thickness

    def to_dict(self):
        return {
            "material": self.material,
            "capacity": self.capacity,
            "areal_capacity": self.areal_capacity,
            "surface_area": self.surface_area,
            "thickness": self.thickness
        }

    def from_dict(self, document):
        self.material = document['material']
        self.capacity = document['capacity']
        self.areal_capacity = document['areal_capacity']
        self.surface_area = document['surface_area']
        self.thickness = document['thickness']

    def add_gui_components(self, root):
        from tkinter import Label
        Label(root, text=f"Material: {self.material}").pack()
        Label(root, text=f"Capacity: {self.capacity} mAh/g").pack()
        Label(root, text=f"Areal Capacity: {self.areal_capacity} mAh/cm²").pack()
        Label(root, text=f"Surface Area: {self.surface_area} cm²").pack()
        Label(root, text=f"Thickness: {self.thickness} cm").pack()
