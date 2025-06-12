import os
import shutil
import glob
import tkinter as tk
from tkinter import filedialog, messagebox

def gather_pdfs(source_dir: str, dest_dir: str, recursive: bool = False):
    """
    Copies all .pdf files from source_dir into dest_dir.
    If recursive is True, searches subdirectories as well.
    """
    os.makedirs(dest_dir, exist_ok=True)

    if recursive:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(dest_dir, file)
                    try:
                        shutil.copy2(src_path, dst_path)
                        print(f"Copied: {src_path} -> {dst_path}")
                    except Exception as e:
                        print(f"Failed to copy {src_path}: {e}")
    else:
        pattern = os.path.join(source_dir, "*.pdf")
        for src_path in glob.glob(pattern):
            file = os.path.basename(src_path)
            dst_path = os.path.join(dest_dir, file)
            try:
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {src_path} -> {dst_path}")
            except Exception as e:
                print(f"Failed to copy {src_path}: {e}")

class PDFGatherGUI(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        master.title("Gather all PDFs")
        master.resizable(False, False)
        self.grid(padx=10, pady=10)

        # Source folder
        tk.Label(self, text="Source Directory:").grid(row=0, column=0, sticky="w")
        self.src_var = tk.StringVar()
        tk.Entry(self, textvariable=self.src_var, width=50).grid(row=1, column=0, columnspan=2, padx=(0,10))
        tk.Button(self, text="Browse…", command=self.select_source).grid(row=1, column=2)

        # Destination folder
        tk.Label(self, text="Destination Folder:").grid(row=2, column=0, sticky="w", pady=(10,0))
        self.dest_var = tk.StringVar()
        tk.Entry(self, textvariable=self.dest_var, width=50).grid(row=3, column=0, columnspan=2, padx=(0,10))
        tk.Button(self, text="Browse…", command=self.select_dest).grid(row=3, column=2)

        # Recursive checkbox
        self.recursive_var = tk.BooleanVar()
        tk.Checkbutton(self, text="Include subdirectories", variable=self.recursive_var).grid(row=4, column=0, columnspan=2, pady=(10,0), sticky="w")

        # Gather button
        tk.Button(self, text="Gather PDFs", command=self.run_gather).grid(row=5, column=0, columnspan=3, pady=(15,0))

    def select_source(self):
        folder = filedialog.askdirectory(title="Select Source Directory")
        if folder:
            self.src_var.set(folder)

    def select_dest(self):
        folder = filedialog.askdirectory(title="Select Destination Folder")
        if folder:
            self.dest_var.set(folder)

    def run_gather(self):
        src = self.src_var.get().strip()
        dst = self.dest_var.get().strip()
        rec = self.recursive_var.get()

        if not src or not os.path.isdir(src):
            messagebox.showerror("Error", "Please select a valid source directory.")
            return

        if not dst:
            messagebox.showerror("Error", "Please select a destination folder.")
            return

        try:
            gather_pdfs(src, dst, rec)
            messagebox.showinfo("Done", "All PDFs have been copied.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    PDFGatherGUI(root)
    root.mainloop()
