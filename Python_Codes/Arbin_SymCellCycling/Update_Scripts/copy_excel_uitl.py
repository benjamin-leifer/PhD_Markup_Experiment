import os
import shutil

def copy_excel_csv_files(source_dir, target_dir):
    """
    Recursively copy all .csv and .xlsx/.xls files from source_dir
    and its subdirectories into target_dir.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.csv', '.xlsx', '.xls')):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)

                # Avoid overwriting files with the same name
                base, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(target_path):
                    target_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
                    counter += 1

                shutil.copy2(source_path, target_path)
                print(f"Copied: {source_path} -> {target_path}")


if __name__ == "__main__":
    # Change these paths before running
    source_directory = r"C:\Users\benja\Downloads\Temp\C_10 Cycling\2025\08\C_10 Sept Update"
    target_directory = r"C:\Users\benja\Downloads\DQ_DV Work\Lab Arbin_DQ_DV_2025_07_15\07\Dq_DV"

    copy_excel_csv_files(source_directory, target_directory)
    print("Done!")
