# main.py
from electrolyte import Electrolyte
from electrode import Electrode
from cell import Cell
from cycling_protocol import CyclingProtocol
from tkinter import Tk, Label, Button, Entry

def main():
    # Create Electrolyte
    electrolyte = Electrolyte(composition="DME", conductivity=10, diffusion_coefficient=1e-7, transference_number=0.4)

    # Create Cathode and Anode
    cathode = Electrode(material="NMC532", capacity=170, areal_capacity=1.774, surface_area=10, thickness=0.05)
    anode = Electrode(material="Graphite", capacity=350, areal_capacity=1.903, surface_area=10, thickness=0.04)

    # Create Cycling Protocol
    cycling_protocol = CyclingProtocol(current_density=1, voltage_limits=(2.5, 4.2), step_time=3600)

    # Create a Cell using the Electrolyte, Cathode, Anode, and Cycling Protocol
    cell = Cell(electrolyte=electrolyte, cathode=cathode, anode=anode, separator="Celgard 2320", cycling_protocol=cycling_protocol)

    # Save to MongoDB
    cell.save_to_db()

    # Create GUI for the Cell
    cell.create_gui()

if __name__ == "__main__":
    main()
