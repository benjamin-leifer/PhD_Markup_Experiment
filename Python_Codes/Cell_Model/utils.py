# utils.py
from tkinter import Tk, Label, Button, Entry

def convert_units(from_unit, to_unit, value):
    """Utility to convert between units (e.g., mAh to Ah, cm² to m²)."""
    conversion_factors = {
        ('mAh', 'Ah'): 0.001,
        ('cm²', 'm²'): 0.0001
    }
    factor = conversion_factors.get((from_unit, to_unit), 1)
    return value * factor

def log_action(action):
    """Utility for logging actions."""
    print(f"[LOG] {action}")

def validate_value(value, min_value=None, max_value=None):
    """Check if a value is within a valid range."""
    if min_value is not None and value < min_value:
        raise ValueError(f"Value {value} is less than the minimum allowed {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"Value {value} exceeds the maximum allowed {max_value}")
    return True
