#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot dQ/dV plotter for Biologic .mpt files
----------------------------------------------
• Finds the *actual* header row by scanning for “Ewe”, “Ecell”, or “Voltage”
• Uses that row’s tokens as column names, then streams the rest in chunks
• Works for charge or discharge on any cycle
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
# ── put this right after the imports and BEFORE main() ──



# ────────────────────────── USER SETTINGS ────────────────────────────────
DATA_DIR   = Path(r"C:\Users\benja\Downloads\DQ_DV Work\Lab Arbin_DQ_DV_2025_07_15\07\Dq_DV")            # folder with the .mpt files
FILES      = [
    "BL-LL-FZ01_RT_C_20_Charge_02_CP_C04.mpt",
    #"BL-LL-FY01_RT_C_20_Charge_02_CP_C04.mpt",
    #"BL-LL-FX01_RT_C_20_Charge_02_CP_C02.mpt",
    "BL-LL-FW01_RT_C_20_Charge_02_CP_C01.mpt",
    "BL-LL-GA01_RT_C_20_Charge_02_CP_C02.mpt",
]
CYCLE      = 0            # cycle index
CHARGE     = True         # True → charge, False → discharge
WIN_PRE    = 201           # rolling window (capacity) before derivative
WIN_POST   = 51            # rolling window on dQ/dV after derivative
CHUNK_SIZE = 25_000       # rows per streamed chunk
ENCODING   = "cp1252"     # EC-Lab default

# ────────────────────────── INTERNAL HELPERS ─────────────────────────────
# -------------------------------------------------------------------------
#  REPLACEMENT: robust header locator
# -------------------------------------------------------------------------
def find_header_row(fp: Path, *, enc="cp1252", max_scan=1200) -> tuple[int, list[str]]:
    """
    Return (row_index, column_names_list).

    Strategy
    --------
    1. If the second non-blank line matches "Nb header lines : N",
       use `N-1` (because lines are 0-indexed).
    2. Otherwise, scan until we hit a line that contains BOTH
       "time/s" **and** "Ewe" (or "Ecell" or "Voltage").
    """
    with open(fp, "r", encoding=enc, errors="ignore") as f:
        lines = []
        for i, raw in enumerate(f):
            if raw.strip():
                lines.append((i, raw.rstrip("\n")))
            if len(lines) >= 2:
                break

    # ── 1) fast path via explicit count ────────────────────────────
    if len(lines) >= 2 and lines[1][1].lower().startswith("nb header lines"):
        try:
            n_hdr = int(re.search(r":\s*(\d+)", lines[1][1]).group(1))
            header_idx = n_hdr - 1          # zero-based
            # read just that row to get the column tokens
            with open(fp, "r", encoding=enc, errors="ignore") as f:
                for i, line in enumerate(f):
                    if i == header_idx:
                        cols = line.rstrip("\n").split("\t")
                        return header_idx, cols
        except (AttributeError, ValueError):
            pass  # fall through to scan if the number is malformed

    # ── 2) fallback scan ───────────────────────────────────────────
    pat_v  = re.compile(r"\b(Ewe|Ecell|Voltage)\b", re.I)
    pat_t  = re.compile(r"\btime/s\b", re.I)
    with open(fp, "r", encoding=enc, errors="ignore") as f:
        for idx, line in enumerate(f):
            if idx > max_scan:
                break
            if pat_v.search(line) and pat_t.search(line):
                cols = line.rstrip("\n").split("\t")
                if len(cols) > 3:
                    return idx, cols

    raise RuntimeError(f"Column header row not found in first {max_scan} lines of {fp.name}")
def debug_one(fp, cycle_idx=1, charge=True):
    hdr_idx, hdr_cols = find_header_row(fp)
    print(f"\nHeader row = {hdr_idx}")
    print("Header tokens:", hdr_cols)

    v_col, q_col, cyc_col = detect_cols(hdr_cols, charge=charge)
    print(f"Picked columns → V: '{v_col}',  Q: '{q_col}',  cycle: {cyc_col}")

    rows_total = 0
    for ch in stream_numeric(fp, hdr_idx, hdr_cols, usecols=[v_col, q_col, cyc_col]):
        if cyc_col:
            ch = ch[ch[cyc_col] == cycle_idx]
        rows_total += len(ch)
        if rows_total and rows_total < 10:
            print("\nFirst few lines after filters:")
            print(ch.head())

    print(f"\nTotal rows kept: {rows_total}")
    if not rows_total:
        print("→ All rows filtered out – check cycle number or current-sign logic.")
        return

    # build DataFrame for derivative and print min/max
    df = pd.concat(stream_numeric(fp, hdr_idx, hdr_cols, usecols=[v_col, q_col, cyc_col]),
                   ignore_index=True)
    if cyc_col:
        df = df[df[cyc_col] == cycle_idx]

    df = df.drop_duplicates(subset=v_col)
    df.columns = ["Voltage", "Capacity", *df.columns[2:]]  # standardise first 2
    print("\nVoltage range :", df['Voltage'].min(), "→", df['Voltage'].max())
    print("Capacity range:", df['Capacity'].min(), "→", df['Capacity'].max())
    print(df[['Voltage', 'Capacity']].head(), "...\n")
    return df

# ----------------- RUN THE DEBUG on one file -----------------



def stream_numeric(fp: Path, header_idx: int, colnames: list[str], usecols=None):
    """Yield DataFrame chunks with *numeric* columns only."""
    reader = pd.read_csv(
        fp,
        sep="\t",  # <- use TAB, not arbitrary whitespace
        header=None,
        names=colnames,
        skiprows=range(header_idx + 1),
        encoding=ENCODING,
        chunksize=CHUNK_SIZE,
        engine="python",
    )
    for chunk in reader:
        chunk = chunk.dropna(how="all")     # toss all-blank rows
        yield chunk.apply(pd.to_numeric, errors="ignore")


# -------------------------------------------------------------------------
#  BETTER column detector  – distinguishes half-cycle capacity columns
# -------------------------------------------------------------------------
def detect_cols(cols: list[str], charge=True):
    """
    Return (Vcol, Qcol, cycle_col|None).

    * Voltage:  <Ewe/V>  or  Ecell(V)  or  Voltage
    * Capacity: Q charge/mA.h   or   Q discharge/mA.h   (half-cycle specific)
    """
    def match(pats):
        for pat in pats:
            for c in cols:
                if re.search(pat, c, re.I):
                    return c
        return None

    v_col = match([r"<Ewe", r"\bEwe", r"\bEcell", r"\bVoltage"])
    if not v_col:
        raise KeyError(f"No voltage column found in headers: {cols}")

    if charge:
        q_col = match([r"\bQ\s*charge/mA\.?h", r"\bQ.*charge"])
    else:
        q_col = match([r"\bQ\s*discharge/mA\.?h", r"\bQ.*discharge"])
    if not q_col:
        raise KeyError(f"No capacity column found in headers: {cols}")

    cyc_col = match([r"\bcycle.*number", r"\bcycle.*index", r"\bNs"])
    return v_col, q_col, cyc_col



def load_vq(fp: Path, cycle: int, charge: bool):
    hdr_idx, hdr_cols = find_header_row(fp)
    v_col, q_col, cyc_col = detect_cols(hdr_cols, charge=charge)

    chunks = []
    for ch in stream_numeric(fp, hdr_idx, hdr_cols, usecols=[v_col, q_col, cyc_col]):
        #if cyc_col:
        #ch = ch[ch[cyc_col] == cycle]
        chunks.append(ch[[v_col, q_col]])

    df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=v_col)
    df.columns = ["Voltage", "Capacity"]    # standardise
    return df.astype(float)


def dqdv(df: pd.DataFrame, win_pre=31, win_post=7):
    v = df["Voltage"].to_numpy()
    q = df["Capacity"].to_numpy() / 1000.0  # mAh → Ah

    order = np.argsort(v)
    v, q = v[order], q[order]

    q_s = pd.Series(q).rolling(win_pre, center=True, min_periods=1).mean().to_numpy()
    dq  = np.diff(q_s)
    dv  = np.diff(v)
    v_m = 0.5 * (v[:-1] + v[1:])

    with np.errstate(divide="ignore", invalid="ignore"):
        y = dq / dv
    y[~np.isfinite(y)] = np.nan
    y = pd.Series(y).rolling(win_post, center=True, min_periods=1).mean().to_numpy()
    return v_m, y

# ── put this right after the imports and BEFORE main() ──
def debug_one(fp, cycle_idx=1, charge=True):
    hdr_idx, hdr_cols = find_header_row(fp)
    print(f"\nHeader row = {hdr_idx}")
    print("Header tokens:", hdr_cols)

    v_col, q_col, cyc_col = detect_cols(hdr_cols, charge=charge)
    print(f"Picked columns → V: '{v_col}',  Q: '{q_col}',  cycle: {cyc_col}")

    rows_total = 0
    for ch in stream_numeric(fp, hdr_idx, hdr_cols, usecols=[v_col, q_col, cyc_col]):
        if cyc_col:
            ch = ch[ch[cyc_col] == cycle_idx]
        rows_total += len(ch)
        if rows_total and rows_total < 10:
            print("\nFirst few lines after filters:")
            print(ch.head())

    print(f"\nTotal rows kept: {rows_total}")
    if not rows_total:
        print("→ All rows filtered out – check cycle number or current-sign logic.")
        return

    # build DataFrame for derivative and print min/max
    df = pd.concat(stream_numeric(fp, hdr_idx, hdr_cols, usecols=[v_col, q_col, cyc_col]),
                   ignore_index=True)
    if cyc_col:
        df = df[df[cyc_col] == cycle_idx]

    df = df.drop_duplicates(subset=v_col)
    df.columns = ["Voltage", "Capacity", *df.columns[2:]]  # standardise first 2
    print("\nVoltage range :", df['Voltage'].min(), "→", df['Voltage'].max())
    print("Capacity range:", df['Capacity'].min(), "→", df['Capacity'].max())
    print(df[['Voltage', 'Capacity']].head(), "...\n")
    return df

# ----------------- RUN THE DEBUG on one file -----------------
test_fp = Path(r"C:\Users\benja\Downloads\DQ_DV Work\Lab Arbin_DQ_DV_2025_07_15\07\Dq_DV\BL-LL-FZ01_RT_C_20_Charge_02_CP_C04.mpt")
debug_one(test_fp, cycle_idx=1, charge=True)


# ────────────────────────── MAIN ─────────────────────────────────────────
def main():
    plt.figure(figsize=(7, 4))
    for fn in FILES:
        fp = DATA_DIR / fn
        print("Processing", fp.name)
        df_vq = load_vq(fp, CYCLE, CHARGE)
        v_mid, y = dqdv(df_vq, WIN_PRE, WIN_POST)

        label = re.search(r"LL-([A-Z]{2}\d{2})", fn).group(1)  # e.g. FZ01
        plt.plot(v_mid, y, lw=1.2, alpha=0.9, label=label)

    seg = "Charge" if CHARGE else "Discharge"
    plt.xlabel("Voltage (V)")
    plt.ylabel("dQ/dV (Ah V$^{-1}$)")
    plt.title(f"dQ/dV – {seg} Cycle {CYCLE}")
    plt.legend(fontsize="x-small", ncol=3)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
    main()
