import os
import shutil

# Define source directory and target directory
def copy_excel_files(source_directory, target_directory):
    # Ensure the target directory exists
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Walk through the directory structure
    for root, dirs, files in os.walk(source_directory):
        # Filter directories that contain 'BL-LL-D' in their name
        if 'BL-LL-D' in root:
            for file in files:
                # Check if the file is an Excel file
                if file.endswith(('.xls', '.xlsx')):
                    # Full path of the source file
                    source_file = os.path.join(root, file)

                    # Full path of the destination file
                    destination_file = os.path.join(target_directory, file)

                    # Copy the file
                    shutil.copy(source_file, destination_file)
                    print(f"Copied: {source_file} to {destination_file}")

# Example usage
source_directory = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\01"  # Replace with your source directory path
target_directory = r"C:\Users\benja\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\2025\01\BL-LL-Ds Combined"  # Replace with your target directory path

copy_excel_files(source_directory, target_directory)