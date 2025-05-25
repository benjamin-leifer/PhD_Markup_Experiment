# main.py
from material_handler import MaterialHandler
from formulation import Formulation

def main():
    # Initialize material handler
    handler = MaterialHandler()

    # Add materials to inventory
    handler.add_material(name="LiTFSI", lot_number="A123", quantity=1000, unit="g")
    handler.add_material(name="DME", lot_number="B456", quantity=500, unit="mL")

    # Create a formulation for an electrolyte using materials from inventory
    electrolyte_formulation = Formulation(name="Electrolyte Mix")
    electrolyte_formulation.add_material(handler.get_material("A123"), 5)  # 5g LiTFSI
    electrolyte_formulation.add_material(handler.get_material("B456"), 10)  # 10mL DME
    electrolyte_formulation.save_to_db()

    # Load the formulation from the database using the formulation ID
    loaded_formulation = Formulation.load_from_db({"formulation_id": electrolyte_formulation.formulation_id})
    if loaded_formulation:
        print(f"Loaded formulation: {loaded_formulation.name}")

if __name__ == "__main__":
    main()
