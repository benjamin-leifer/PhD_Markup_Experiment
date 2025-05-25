# cell.py
from database_object import DatabaseObject
from formulation import Formulation
from electrode import Electrode
from cycling_protocol import CyclingProtocol

class Cell(DatabaseObject):
    db_name = "inventory"
    collection_name = "formulations"

    def __init__(self, electrolyte_formulation, cathode_formulation, anode_formulation, separator, cycling_protocol):
        super().__init__(db_name="electrochemical_cells", collection_name="cells")
        self.electrolyte_formulation = electrolyte_formulation  # Formulation for the electrolyte
        self.cathode_formulation = cathode_formulation  # Formulation for the cathode
        self.anode_formulation = anode_formulation  # Formulation for the anode
        self.separator = separator  # Separator type
        self.cycling_protocol = cycling_protocol  # CyclingProtocol instance

    def to_dict(self):
        # Store formulation IDs instead of the full formulation details
        return {
            "electrolyte_formulation_id": self.electrolyte_formulation.formulation_id,
            "cathode_formulation_id": self.cathode_formulation.formulation_id,
            "anode_formulation_id": self.anode_formulation.formulation_id,
            "separator": self.separator,
            "cycling_protocol": self.cycling_protocol.to_dict()
        }

    def from_dict(self, document):
        # Load formulations and cycling protocol from their respective IDs
        self.electrolyte_formulation = Formulation.load_from_db({"formulation_id": document["electrolyte_formulation_id"]})
        self.cathode_formulation = Formulation.load_from_db({"formulation_id": document["cathode_formulation_id"]})
        self.anode_formulation = Formulation.load_from_db({"formulation_id": document["anode_formulation_id"]})
        self.separator = document["separator"]
        self.cycling_protocol = CyclingProtocol(**document["cycling_protocol"])

    def add_gui_components(self, root):
        from tkinter import Label
        Label(root, text=f"Electrolyte Formulation: {self.electrolyte_formulation.name}").pack()
        Label(root, text=f"Cathode Formulation: {self.cathode_formulation.name}").pack()
        Label(root, text=f"Anode Formulation: {self.anode_formulation.name}").pack()
        Label(root, text=f"Separator: {self.separator}").pack()
