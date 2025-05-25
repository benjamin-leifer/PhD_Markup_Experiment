# data_handler.py
import pandas as pd
from material import Material
from formulation import Formulation
from database_object import DatabaseObject

class DataHandler:
    def load_materials_from_csv(self, filepath):
        data = pd.read_csv(filepath)
        materials = []
        for _, row in data.iterrows():
            material = Material(
                name=row['name'],
                lot_number=row['lot_number'],
                quantity=row['quantity'],
                unit=row['unit'],
                manufacture_date=row['manufacture_date']
            )
            material.save_to_db()
            materials.append(material)
        return materials

    def export_materials_to_csv(self, materials, filepath):
        data = [material.to_dict() for material in materials]
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Materials exported to {filepath}")

    def load_formulations_from_csv(self, filepath):
        data = pd.read_csv(filepath)
        formulations = []
        for _, row in data.iterrows():
            materials_dict = eval(row['materials'])  # Assuming materials are stored as a dictionary string
            formulation = Formulation(name=row['name'], date=row['date'], materials=materials_dict)
            formulation.save_to_db()
            formulations.append(formulation)
        return formulations
