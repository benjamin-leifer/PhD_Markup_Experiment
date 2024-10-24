# electrolyte.py
from database_object import DatabaseObject
from tkinter import Tk, Label, Button, Entry

class Electrolyte(DatabaseObject):
    def __init__(self, composition, conductivity, diffusion_coefficient, transference_number):
        super().__init__(db_name="electrochemical_cells", collection_name="electrolytes")
        self.composition = composition
        self.conductivity = conductivity
        self.diffusion_coefficient = diffusion_coefficient
        self.transference_number = transference_number

    def to_dict(self):
        return {
            "composition": self.composition,
            "conductivity": self.conductivity,
            "diffusion_coefficient": self.diffusion_coefficient,
            "transference_number": self.transference_number
        }

    def from_dict(self, document):
        self.composition = document['composition']
        self.conductivity = document['conductivity']
        self.diffusion_coefficient = document['diffusion_coefficient']
        self.transference_number = document['transference_number']

    def add_gui_components(self, root):
        Label(root, text=f"Composition: {self.composition}").pack()
        Label(root, text=f"Conductivity: {self.conductivity} S/cm").pack()
        Label(root, text=f"Diffusion Coefficient: {self.diffusion_coefficient} cm²/s").pack()
        Label(root, text=f"Transference Number: {self.transference_number}").pack()
