"""
EIS fitting with impedance.py
Circuit: Rs + (Rsei || CPE1) + (Rct + W) || CPE2

Install:
    pip install impedance matplotlib numpy pandas

Circuit string elements:
    R0       → Rs    (ohmic)
    R1       → Rsei  (SEI resistance)
    CPE1     → [Q1, n1]  (SEI CPE)
    R2       → Rct   (charge-transfer)
    W1       → Aw    (semi-infinite Warburg inside CT parallel)
    CPE2     → [Q2, n2]  (CT CPE)

Constraint enforced via bounds: n2 > 0.5
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from impedance.models.circuits import CustomCircuit
from impedance.preprocessing import ignoreBelowX
from pathlib import Path

# Label sizing handle
Scale = 3
LABEL_FS = 16*Scale
TICK_FS = 14*Scale
LEGEND_FS = 12*Scale
LINE_W = 2.0*Scale
MARKER_S = 20*Scale
TICK_LEN = 6.0*Scale
TICK_W = 1.2*Scale


def save_ax_legend_png(ax, out_png: str, *, ncol: int = 2, fontsize: int = 9, dpi: int = 300) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    # De-dupe while preserving order
    seen = set()
    H, L = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        H.append(h)
        L.append(l)

    fig = plt.figure(figsize=(6.0, 0.6 + 0.35 * (len(L) / max(1, ncol))))
    axl = fig.add_subplot(111)
    axl.axis("off")
    axl.legend(H, L, ncol=ncol, frameon=False, loc="center", fontsize=fontsize,
               handlelength=2.2, handletextpad=0.6, columnspacing=1.0)
    fig.savefig(out_png, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for fitting/params
# ─────────────────────────────────────────────────────────────────────────────

def _norm_param_name(n: str) -> str:
    return str(n).replace("_", "-")

def _get_param(d: dict, *keys: str) -> float:
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"Missing param (tried: {keys}). Available: {list(d.keys())}")


def _cell_color(path: str) -> str:
    name = str(Path(path).name).upper()
    if "CX02" in name:
        return "black"
    if "EF02" in name:
        return "green"
    return "tab:blue"


def _cell_label(path: str) -> str:
    name = str(Path(path).name).upper()
    if "CX02" in name:
        return "LP"
    if "EF02" in name:
        return "Exp"
    return Path(path).stem


def fit_dataset(filepath: str, cycle: int | None, circuit_str: str, initial_guess: list[float], bounds) -> dict:
    df = parse_mpt_header_safe(filepath)
    spec, cycle_id, v_sel = select_cycle(df, cycle=cycle)

    freq = spec["freq/Hz"].values
    Z_re = spec["Re(Z)/Ohm"].values
    Z_neg_im = spec["-Im(Z)/Ohm"].values
    Z = Z_re - 1j * Z_neg_im

    # Pre-processing
    freq, Z = ignoreBelowX(freq, Z)
    mask = freq <= 2e5
    freq, Z = freq[mask], Z[mask]

    model = CustomCircuit(circuit_str, initial_guess=initial_guess)
    model.fit(freq, Z, bounds=bounds, weight_by_modulus=True)

    names = model.get_param_names()
    if names and isinstance(names[0], (list, tuple)):
        names = [n for sub in names for n in sub]
    values = model.parameters_
    errors = model.conf_

    param_dict = {_norm_param_name(n): v for n, v in zip(names, values)}
    error_dict = {_norm_param_name(n): e for n, e in zip(names, errors)}

    # Derived parameters
    Rs = _get_param(param_dict, "R0")
    Rsei = _get_param(param_dict, "R1")
    Q1 = _get_param(param_dict, "CPE1-0", "CPE1_0")
    n1 = _get_param(param_dict, "CPE1-1", "CPE1_1")
    Rct = _get_param(param_dict, "R2")
    Q2 = _get_param(param_dict, "CPE2-0", "CPE2_0")
    n2 = _get_param(param_dict, "CPE2-1", "CPE2_1")
    Z0 = _get_param(param_dict, "Ws1-0", "Ws1_0", "Ws1")
    tau = _get_param(param_dict, "Ws1-1", "Ws1_1")
    Aw = Z0 / np.sqrt(2 * np.pi * tau)

    f_sei = (1.0 / (Rsei * Q1)) ** (1.0 / n1) / (2 * np.pi)
    f_ct = (1.0 / (Rct * Q2)) ** (1.0 / n2) / (2 * np.pi)

    f_dense = np.logspace(np.log10(freq.min()), np.log10(freq.max()), 600)
    Z_dense = model.predict(f_dense)
    Z_fit = model.predict(freq)

    res_re = (Z.real - Z_fit.real) / np.abs(Z) * 100
    res_im = (Z.imag - Z_fit.imag) / np.abs(Z) * 100

    return dict(
        filepath=filepath,
        freq=freq,
        Z=Z,
        f_dense=f_dense,
        Z_dense=Z_dense,
        res_re=res_re,
        res_im=res_im,
        cycle_id=cycle_id,
        v_sel=v_sel,
        model=model,
        params=param_dict,
        errors=error_dict,
        derived=dict(Rs=Rs, Rsei=Rsei, Rct=Rct, Q1=Q1, n1=n1, Q2=Q2, n2=n2, Aw=Aw, f_sei=f_sei, f_ct=f_ct),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PARSER
# ─────────────────────────────────────────────────────────────────────────────

def get_header_lines(path: Path) -> int:
    """
    Parse 'Nb header lines : N' from the EC-Lab ASCII header.
    """
    with path.open("r", encoding="cp1252") as f:
        for line in f:
            if "Nb header lines" in line:
                try:
                    return int(line.split(":")[1].strip())
                except Exception:
                    pass
    raise RuntimeError("Could not find 'Nb header lines' in header.")


def parse_mpt(filepath: str) -> pd.DataFrame:
    """Read an EC-Lab ASCII .mpt file into a tidy DataFrame."""
    with open(filepath, "r", encoding="cp1252") as fh:
        lines = fh.readlines()

    nb_header = None
    for ln in lines:
        if ln.startswith("Nb header lines"):
            nb_header = int(ln.split(":")[1].strip())
            break
    if nb_header is None:
        raise ValueError("Cannot find 'Nb header lines' in file.")

    data_block = lines[nb_header:]
    col_names  = [c.strip() for c in data_block[0].split("\t")]
    rows       = [ln.strip().split("\t") for ln in data_block[1:] if ln.strip()]

    df = pd.DataFrame(rows, columns=col_names)
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except Exception:
            pass
    return df


def parse_mpt_header_safe(filepath: str) -> pd.DataFrame:
    """Robust .mpt parser (handles varied 'Nb header lines' formatting)."""
    path = Path(filepath)
    with path.open("r", encoding="cp1252") as fh:
        lines = fh.readlines()

    nb_header = get_header_lines(path)

    # Prefer the header line containing 'cycle number' AND 'freq/Hz'
    header_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        if "cycle number" in low and "freq/hz" in low and "\t" in line:
            header_idx = i
            break
    # Fallback: just 'cycle number' with tabs
    if header_idx is None:
        for i, line in enumerate(lines):
            low = line.lower()
            if "cycle number" in low and "\t" in line:
                header_idx = i
                break
    if header_idx is None:
        start = nb_header
        while start < len(lines) and not lines[start].strip():
            start += 1
        header_idx = start
    if header_idx >= len(lines):
        raise ValueError("No header row found after header lines.")

    header = [c.strip() for c in lines[header_idx].strip().split("\t")]
    data_lines = [ln for ln in lines[header_idx + 1:] if ln.strip()]
    rows = []
    ncols = len(header)
    for ln in data_lines:
        parts = ln.strip().split("\t")
        if len(parts) != ncols:
            continue
        rows.append(parts)

    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except Exception:
            pass
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CYCLE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, contains: list[str]) -> str:
    for col in df.columns:
        col_l = str(col).lower()
        if all(token in col_l for token in contains):
            return col
    raise KeyError(f"No column matches tokens={contains}. Columns={list(df.columns)}")


def select_cycle(df: pd.DataFrame,
                 cycle: int | None = None,
                 fraction: float = 0.5) -> tuple[pd.DataFrame, int, float]:
    """
    Select a specific cycle by number, or auto-select the cycle whose mean
    voltage is closest to `fraction` of the full window (default = mid-SOC).
    """
    cycle_col = _find_col(df, ["cycle", "number"])
    v_col = _find_col(df, ["ewe", "/v"])
    cycle_v = df.groupby(cycle_col)[v_col].mean()

    if cycle is not None:
        if cycle not in cycle_v.index:
            raise ValueError(
                f"Cycle {cycle} not found. Available: {sorted(cycle_v.index.tolist())}"
            )
        v_sel = float(cycle_v[cycle])
        spec  = df[df[cycle_col] == cycle].copy()
        print(f"Selected cycle : {int(cycle)}  |  Ewe = {v_sel:.4f} V")
    else:
        v_lo, v_hi = cycle_v.min(), cycle_v.max()
        v_target   = v_lo + fraction * (v_hi - v_lo)
        cycle      = int((cycle_v - v_target).abs().idxmin())
        v_sel      = float(cycle_v[cycle])
        spec       = df[df[cycle_col] == cycle].copy()
        print(f"Voltage window : {v_lo:.4f} – {v_hi:.4f} V")
        print(f"Auto-selected  : cycle {cycle}  (Ewe = {v_sel:.4f} V, fraction={fraction})")

    return spec, cycle, v_sel


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():

    FILEPATH ="BL-LL-CV01_EIS_STAIR_RT_05_CP_C03_01_PEIS_C04.mpt"
    #FILEPATH = "BL-LL-CX02_EIS_STAIR_RT_01_PEIS_C02.mpt"
    FILEPATH_2 = "BL-LL-EF02_EIS_STAIR_RT_01_PEIS_C03.mpt"
    CYCLE = 7 # set to None to auto-select mid-SOC
    CYCLE_2 = 4  # set to None to auto-select mid-SOC
    NYQUIST_OFFSET = 20.0  # Ω offset for second dataset (y-axis)
    PLOT_SECOND_CELL = False  # set False to plot only FILEPATH

    # ── Define circuit ────────────────────────────────────────────────────
    circuit_str = "R0-p(R1,CPE1)-p(R2-Ws1,CPE2)"

    initial_guess = [
        20.0,  # R0   Rs
        5.0,   # R1   Rsei
        1e-4,  # CPE1-0  Q1
        0.80,  # CPE1-1  n1
        30.0,  # R2   Rct
        10.0,  # Ws1-0  Z0  (Ohm)
        1.0,  # Ws1-1  tau (s)
        5e-4,  # CPE2-0  Q2
        0.80,  # CPE2-1  n2
    ]

    bounds = (
        [0.0, 0.0, 1e-9, 0.50, 0.0, 0.0, 1e-6, 1e-9, 0.51],
        [50.0, 50.0, 1e-1, 1.00, 300.0, 500.0, 1e6, 1e-1, 1.00],
    )

    # ── Fit both datasets ────────────────────────────────────────────────
    data_1 = fit_dataset(FILEPATH, CYCLE, circuit_str, initial_guess, bounds)
    data_2 = None
    if PLOT_SECOND_CELL:
        data_2 = fit_dataset(FILEPATH_2, CYCLE_2, circuit_str, initial_guess, bounds)

    print(f"Points for fit (1) : {len(data_1['freq'])}")
    print(f"Freq range     (1) : {data_1['freq'].min():.4f} – {data_1['freq'].max():.1f} Hz\n")
    if data_2 is not None:
        print(f"Points for fit (2) : {len(data_2['freq'])}")
        print(f"Freq range     (2) : {data_2['freq'].min():.4f} – {data_2['freq'].max():.1f} Hz\n")

    # ── Print results ─────────────────────────────────────────────────────
    print(data_1["model"])  # impedance.py summary for dataset 1
    if data_2 is not None:
        print(data_2["model"])  # impedance.py summary for dataset 2

    print(f"  f_SEI (characteristic) [1] = {data_1['derived']['f_sei']:.2f} Hz")
    print(f"  f_CT  (characteristic) [1] = {data_1['derived']['f_ct']:.4f} Hz\n")
    if data_2 is not None:
        print(f"  f_SEI (characteristic) [2] = {data_2['derived']['f_sei']:.2f} Hz")
        print(f"  f_CT  (characteristic) [2] = {data_2['derived']['f_ct']:.4f} Hz\n")

    # ── Nyquist-only plot ────────────────────────────────────────────────
    fig, ax_ny = plt.subplots(figsize=(7.5, 6.5))
    if data_2 is not None:
        fig.suptitle(
            f"EIS — Cycle {data_1['cycle_id']} vs {data_2['cycle_id']}  |  "
            f"Ewe = {data_1['v_sel']:.3f} V & {data_2['v_sel']:.3f} V",
            fontsize=12, fontweight="bold",
        )

    # ── Nyquist ──────────────────────────────────────────────────────────
    c1 = _cell_color(data_1["filepath"])
    l1 = _cell_label(data_1["filepath"])

    ax_ny.plot(data_1["Z_dense"].real, -data_1["Z_dense"].imag, color=c1, lw=LINE_W, label=f"{l1} Fit")

    x1 = data_1["Z"].real
    y1 = -data_1["Z"].imag

    ax_ny.scatter(
        x1, y1,
        s=MARKER_S,
        facecolors=c1,
        edgecolors=c1,
        linewidths=0.8,
        zorder=5,
        label=f"{l1} Data",
    )

    if data_2 is not None:
        c2 = _cell_color(data_2["filepath"])
        l2 = _cell_label(data_2["filepath"])

        ax_ny.plot(data_2["Z_dense"].real, -data_2["Z_dense"].imag + NYQUIST_OFFSET, color=c2, lw=LINE_W, label=f"{l2} Fit (offset)")

        x2 = data_2["Z"].real
        y2 = -data_2["Z"].imag + NYQUIST_OFFSET

        ax_ny.scatter(
            x2, y2,
            s=MARKER_S,
            facecolors=c2,
            edgecolors=c2,
            linewidths=0.8,
            zorder=5,
            label=f"{l2} Data (offset)",
        )

    ax_ny.set_xlabel("Re(Z) / Ω", fontsize=LABEL_FS)
    ax_ny.set_ylabel("−Im(Z) / Ω", fontsize=LABEL_FS)
    ax_ny.set_aspect("equal", adjustable="box")
    try:
        ax_ny.set_box_aspect(1)
    except Exception:
        pass
    ax_ny.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=TICK_FS,
        length=TICK_LEN,
        width=TICK_W,
    )
    # No legend on main plot; export separately for slides

    plt.tight_layout()
    plt.savefig("eis_impedancepy_fit.png", dpi=150, bbox_inches="tight")
    save_ax_legend_png(ax_ny, "eis_impedancepy_fit_legend.png", ncol=2, fontsize=LEGEND_FS, dpi=300)
    print("Plot saved → eis_impedancepy_fit.png")
    plt.show()


if __name__ == "__main__":
    main()

