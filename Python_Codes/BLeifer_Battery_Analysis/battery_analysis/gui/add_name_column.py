import pandas as pd
import os


def add_name_column(file_path, save_path=None):
    """
    Add a 'Name' column to an Excel file based on Cell Code.

    Args:
        file_path: Path to the input Excel file
        save_path: Path to save the updated file (if None, will add '_updated' to original filename)
    """
    # Set default save path if not provided
    if save_path is None:
        base, ext = os.path.splitext(file_path)
        save_path = f"{base}_updated{ext}"

    # Read the Excel file
    print(f"Reading file: {file_path}")
    df = pd.read_excel(file_path)

    # Check if 'Cell Code' column exists
    if 'Cell Code' not in df.columns:
        print(f"Warning: 'Cell Code' column not found. Available columns: {', '.join(df.columns)}")
        return False

    # Create counter dictionary to keep track of counts for each Cell Code
    code_counters = {}

    # Function to generate name based on Cell Code
    def generate_name(cell_code):
        if pd.isna(cell_code) or cell_code == '':
            return ''

        # Convert to string in case it's numeric
        cell_code = str(cell_code).strip()

        # Update counter for this code
        if cell_code not in code_counters:
            code_counters[cell_code] = 1
        else:
            code_counters[cell_code] += 1

        # Generate name with format XX0#
        return f"{cell_code}0{code_counters[cell_code]}"

    # Create the Name column
    df.insert(0, 'Name', df['Cell Code'].apply(generate_name))

    # Save the updated file
    print(f"Saving updated file to: {save_path}")
    df.to_excel(save_path, index=False)

    return True


# Example usage
if __name__ == "__main__":
    file_path = "Spring 2025 Cell List.xlsx"  # Change this to your file path if needed
    add_name_column(file_path)
    print("Done! Check the updated Excel file.")