import subprocess
from pathlib import Path

workdir = Path(r"C:\Users\benja\PycharmProjects\PhD_Markup_Experiment\Python_Codes\Visualizations\Proposal\EIS_Prop")
texfile = workdir / "eis_circuit.tex"

PDFLATEX = r"C:\Users\benja\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"

# 1) Compile to PDF
subprocess.run(
    [PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", texfile.name],
    cwd=str(workdir),
    check=True
)

pdf = workdir / texfile.with_suffix(".pdf").name
png = workdir / texfile.with_suffix(".png").name

# 2) Convert PDF -> PNG (choose ONE converter)

# --- Option A: Inkscape (recommended if installed) ---
# inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"
# subprocess.run([inkscape_path, str(pdf), "--export-type=png", "--export-dpi=600", f"--export-filename={png}"],
#                cwd=str(workdir), check=True)

#--- Option B: Poppler pdftocairo (if available on PATH) ---
subprocess.run(["pdftocairo", "-png", "-r", "600", str(pdf), str(workdir / texfile.stem)],
               cwd=str(workdir), check=True)
# pdftocairo outputs eis_circuit-1.png; rename it:
(workdir / f"{texfile.stem}-1.png").replace(png)

# --- Option C: ImageMagick (if installed) ---
# subprocess.run(["magick", "-density", "600", str(pdf), "-quality", "100", str(png)],
#                cwd=str(workdir), check=True)

print(f"PDF: {pdf}")
print(f"PNG: {png}")
