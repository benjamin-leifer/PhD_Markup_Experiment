# cycling_protocol.py
from tkinter import Tk, Label, Button, Entry

class CyclingProtocol:
    def __init__(self, current_density, voltage_limits, step_time):
        self.current_density = current_density  # mA/cm²
        self.voltage_limits = voltage_limits  # Tuple: (min_voltage, max_voltage)
        self.step_time = step_time  # Time for each step in seconds

    def to_dict(self):
        return {
            "current_density": self.current_density,
            "voltage_limits": self.voltage_limits,
            "step_time": self.step_time
        }

    def from_dict(self, document):
        self.current_density = document["current_density"]
        self.voltage_limits = document["voltage_limits"]
        self.step_time = document["step_time"]

    def display_protocol(self):
        return f"Current Density: {self.current_density} mA/cm², Voltage Limits: {self.voltage_limits}, Step Time: {self.step_time} s"
