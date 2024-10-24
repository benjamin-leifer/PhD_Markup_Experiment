# cell.py
from database_object import DatabaseObject
from tkinter import Tk, Label, Button, Entry
from electrolyte import Electrolyte
from electrode import Electrode
from cycling_protocol import CyclingProtocol

class Cell(DatabaseObject):
    def __init__(self, electrolyte, cathode, anode, separator, cycling_protocol):
        super().__init__(db_name="electrochemical_cells", collection_name="cells")
        self.electrolyte = electrolyte
        self.cathode = cathode
        self.anode = anode
        self.separator = separator
        self.cycling_protocol = cycling_protocol

    def to_dict(self):
        return {
            "electrolyte": self.electrolyte.to_dict(),
            "cathode": self.cathode.to_dict(),
            "anode": self.anode.to_dict(),
            "separator": self.separator,
            "cycling_protocol": self.cycling_protocol.to_dict()
        }

    def from_dict(self, document):
        self.electrolyte = Electrolyte.load_from_db({"_id": document["electrolyte"]})
        self.cathode = Electrode.load_from_db({"_id": document["cathode"]})
        self.anode = Electrode.load_from_db({"_id": document["anode"]})
        self.separator = document["separator"]
        self.cycling_protocol = CyclingProtocol(**document["cycling_protocol"])

    def add_gui_components(self, root):
        from tkinter import Label
        Label(root, text="Cell Properties:").pack()
        Label(root, text=f"Separator: {self.separator}").pack()

        Label(root, text="Cathode:").pack()
        self.cathode.add_gui_components(root)

        Label(root, text="Anode:").pack()
        self.anode.add_gui_components(root)

        Label(root, text="Electrolyte:").pack()
        self.electrolyte.add_gui_components(root)
