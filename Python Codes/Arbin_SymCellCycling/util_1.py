import os
import shutil

def copy_ck_files(source_dir, dest_dir):
    """
    Copies all Excel files containing 'CK0X' in their filenames from source_dir and its subdirectories to dest_dir.

    Parameters
    ----------
    source_dir : str
        The directory to search for files.
    dest_dir : str
        The directory to copy the files to.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, _, files in os.walk(source_dir):
        for filename in files:
            if 'CM' in filename and filename.endswith('.xlsx'):
                source_file = os.path.join(root, filename)
                dest_file = os.path.join(dest_dir, filename)
                shutil.copy2(source_file, dest_file)
                print(f"Copied {filename} to {dest_dir}")

# Example usage
source_directory = r'C:\Users\leifer.be\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\11'
destination_directory = r'C:\Users\leifer.be\OneDrive - Northeastern University\Gallaway Group\Gallaway Extreme SSD Drive\Equipment Data\Lab Arbin\Li-Ion\Low Temp Li Ion\11\CM'

copy_ck_files(source_directory, destination_directory)