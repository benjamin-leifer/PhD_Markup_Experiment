import pandas as pd
from pathlib import Path


# ========= USER SETTINGS (EDIT THESE) =========
INPUT_FILE = r"C:\Users\benja\Downloads\DRT EIS Stair 1\11\EIS Formation Stair\HZ01\BL-LL-HZ01_RT_EIS_Stair_Formation_Stair_03_SPEIS_C02.mpt"  # or .txt/.asc
OUTPUT_DIR = r"C:\Users\benja\Downloads\DRT EIS Stair 1\11\EIS Formation Stair\HZ01\HZ01_RT_EIS_split"  # will be created if it doesn't exist
# =============================================


def get_header_lines(path: Path) -> int:
    """
    Parse 'Nb header lines : N' from the EC-Lab ASCII header.
    """
    with path.open("r", encoding="cp1252") as f:
        for line in f:
            if "Nb header lines" in line:
                # Example line: 'Nb header lines : 65'
                try:
                    return int(line.split(":")[1].strip())
                except Exception:
                    pass
    raise RuntimeError("Could not find 'Nb header lines' in header.")


def main():
    input_path = Path(INPUT_FILE)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_header = get_header_lines(input_path)
    print(f"Detected {n_header} header lines.")

    # Read the data section
    df = pd.read_csv(
        input_path,
        sep=r"\s+",
        skiprows=n_header-1,
        engine="python",
        encoding="cp1252",      # <-- important fix
    )

    # Sanity check column names
    required_cols = [
        "freq/Hz",
        "Re(Z)/Ohm",
        "-Im(Z)/Ohm",
        "<Ewe>/V",
        "cycle",
    ]

    print("Columns in file:")
    print(df.columns.tolist())

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    # Keep only rows with actual frequency points
    df = df[df["freq/Hz"] > 0].copy()

    # Drop any obviously bad rows
    df = df.dropna(subset=["freq/Hz", "Re(Z)/Ohm", "-Im(Z)/Ohm", "<Ewe>/V", "cycle"])

    # Convert -Im(Z) to Im(Z) (DRT usually expects Z = Z' + j Z'')
    df["Z_im_Ohm"] = -df["-Im(Z)/Ohm"]

    summary_rows = []

    base_stem = input_path.stem

    for cycle, group in df.groupby("cycle"):
        # Sort by frequency (just to be tidy for pydrt)
        group = group.sort_values("freq/Hz")

        if group["freq/Hz"].nunique() < 5:
            # Probably not a valid EIS spectrum; skip
            continue

        ewe_mean = group["<Ewe>/V"].mean()
        ewe_std = group["<Ewe>/V"].std()
        ewe_mV = int(round(ewe_mean * 1000))

        out_df = pd.DataFrame({
            "freq_Hz": group["freq/Hz"].values,
            "Z_real_Ohm": group["Re(Z)/Ohm"].values,
            "Z_imag_Ohm": group["Z_im_Ohm"].values,
        })

        # Example filename: BL-LL-HY01_..._cycle041_2300mV.csv
        out_name = f"{base_stem}_cycle{int(cycle):03d}_{ewe_mV:+05d}mV.csv"
        out_path = out_dir / out_name

        out_df.to_csv(out_path, index=False)

        summary_rows.append({
            "cycle": int(cycle),
            "Ewe_mean_V": ewe_mean,
            "Ewe_std_V": ewe_std,
            "Ewe_mean_mV": ewe_mV,
            "n_points": len(out_df),
            "freq_min_Hz": out_df["freq_Hz"].min(),
            "freq_max_Hz": out_df["freq_Hz"].max(),
            "outfile": out_name,
        })

        print(f"Wrote {out_path}  (cycle {cycle}, ~{ewe_mean:.4f} V)")

    # Save summary mapping cycle → potential and file
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("cycle")
        summary_path = out_dir / f"{base_stem}_summary_by_cycle.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary written to: {summary_path}")
    else:
        print("No valid cycles found (check column names and freq>0 filtering).")


if __name__ == "__main__":
    main()
