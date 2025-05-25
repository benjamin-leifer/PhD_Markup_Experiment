# inventory_gui.py
from tkinter import Tk, Label, Entry, Button
from material_handler import MaterialHandler
from database_object import DatabaseObject

def inventory_gui():
    handler = MaterialHandler()

    def add_material():
        name = name_entry.get()
        lot_number = lot_entry.get()
        quantity = float(quantity_entry.get())
        unit = unit_entry.get()
        handler.add_material(name, lot_number, quantity, unit)
        update_inventory()

    def update_inventory():
        for widget in inventory_frame.winfo_children():
            widget.destroy()
        for material in handler.materials:
            Label(inventory_frame, text=f"{material.name} (Lot: {material.lot_number}) - {material.quantity} {material.unit}").pack()

    root = Tk()
    root.title("Inventory Management")

    # Input fields
    Label(root, text="Material Name").pack()
    name_entry = Entry(root)
    name_entry.pack()

    Label(root, text="Lot Number").pack()
    lot_entry = Entry(root)
    lot_entry.pack()

    Label(root, text="Quantity").pack()
    quantity_entry = Entry(root)
    quantity_entry.pack()

    Label(root, text="Unit").pack()
    unit_entry = Entry(root)
    unit_entry.pack()

    Button(root, text="Add Material", command=add_material).pack()

    # Inventory display
    inventory_frame = Label(root)
    inventory_frame.pack()
    update_inventory()

    root.mainloop()
