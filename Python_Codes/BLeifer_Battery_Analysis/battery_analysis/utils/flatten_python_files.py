import os
import shutil
import tkinter as tk
from tkinter import filedialog


def copy_to_flat_structure():
    """
    Copy all Python files from a directory and its subdirectories
    into a flat directory structure, adding the original path as a comment.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Prompt user to select source directory
    print("Please select the source directory containing your Python files...")
    source_dir = filedialog.askdirectory(title="Select Source Directory")
    if not source_dir:
        print("No source directory selected. Exiting.")
        return

    # Prompt user to select target directory
    print("Please select the target directory for the flattened files...")
    target_dir = filedialog.askdirectory(title="Select Target Directory")
    if not target_dir:
        print("No target directory selected. Exiting.")
        return

    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_count = 0

    # Walk through the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.py'):
                # Get the original file path
                original_path = os.path.join(root, file)

                # Create a relative path from the source directory
                rel_path = os.path.relpath(original_path, source_dir)

                # Create target file name (keeping the original filename)
                target_file = os.path.join(target_dir, file)

                # If there are duplicate filenames, add a suffix
                if os.path.exists(target_file):
                    base, ext = os.path.splitext(file)
                    # Replace directory separators with underscore for the suffix
                    suffix = rel_path.replace(file, '').replace('/', '_').replace('\\', '_').rstrip('_')
                    target_file = os.path.join(target_dir, f"{base}_{suffix}{ext}")

                # Read the content of the original file
                with open(original_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Write the content to the target file with the original path as a comment
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Original location: {rel_path}\n\n")
                    f.write(content)

                print(f"Copied: {rel_path} -> {os.path.basename(target_file)}")
                file_count += 1

    print(f"Done! Copied {file_count} Python files to {target_dir}")


if __name__ == "__main__":
    copy_to_flat_structure()