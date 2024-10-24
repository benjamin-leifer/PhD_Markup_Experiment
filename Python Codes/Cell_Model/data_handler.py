# data_handler.py
import pandas as pd
from tkinter import Tk, Label, Button, Entry

class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_csv(self):
        """Load experimental data from a CSV file."""
        try:
            data = pd.read_csv(self.filepath)
            print(f"Data loaded successfully from {self.filepath}")
            return data
        except Exception as e:
            print(f"Error loading data from {self.filepath}: {e}")
            return None

    def clean_data(self, data):
        """Perform data cleaning operations (e.g., handling NaNs, duplicates)."""
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        return data

    def save_to_csv(self, data, output_filepath):
        """Save cleaned or processed data to a new CSV file."""
        data.to_csv(output_filepath, index=False)
        print(f"Data saved to {output_filepath}")
