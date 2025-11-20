import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# USER PARAMETERS
# ---------------------------------------------------------------------------
ROOT_DIRS = {
    "HZ01": r"C:\Users\benja\Downloads\DRT EIS Stair 1\11\EIS Formation Stair\HZ01\HZ01_RT_EIS_split",
    "HY01": r"C:\Users\benja\Downloads\DRT EIS Stair 1\11\EIS Formation Stair\HY01\HY01_RT_EIS_split",
}

CHANNEL_PATTERNS = {
    "HZ01": "SPEIS_C02_cycle",
    "HY01": "SPEIS_C01_cycle",
}

CYCLE_STEP = 8
POT_MIN_V = 2.3
POT_MAX_V = 3.5
POT_BIN_WIDTH = 0.005  # ~5 mV tolerance

# Fixed DRT parameters (from Auto_DRT_t1)
N_TAU = 200
LAMBDA = 0.05
TAU_MIN = 1e-5
TAU_MAX = 1e2


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def parse_cycle_and_potential(path):
    fname = os.path.basename(path)
    m = re.search(r"cycle(\d+)_\+(\d+)mV", fname)
    if not m:
        return None, None
    cycle = int(m.group(1))
    pot_V = int(m.group(2)) / 1000
    return cycle, pot_V


def scan_root_dir_for_spectra(root_dir, label, cycle_step, pot_min_v, pot_max_v, channel_pattern):
    pattern = os.path.join(root_dir, f"*{channel_pattern}*.csv")
    file_paths = sorted(glob.glob(pattern))
    spectra = []
    for fp in file_paths:
        cycle, pot_v = parse_cycle_and_potential(fp)
        if cycle is None or pot_v is None:
            continue
        if not (pot_min_v <= pot_v <= pot_max_v):
            continue
        if (cycle - 1) % cycle_step != 0:
            continue
        spectra.append({"label": label, "cycle": cycle, "potential_V": pot_v, "path": fp})
    return spectra


def build_common_potential_map(root_dirs, channel_patterns, cycle_step, pot_min_v, pot_max_v, pot_bin_width):
    all_spectra = {}
    for label, root_dir in root_dirs.items():
        spectra = scan_root_dir_for_spectra(
            root_dir=root_dir,
            label=label,
            cycle_step=cycle_step,
            pot_min_v=pot_min_v,
            pot_max_v=pot_max_v,
            channel_pattern=channel_patterns[label],
        )
        all_spectra[label] = spectra

    pot_map = {}
    for label, spectra in all_spectra.items():
        for rec in spectra:
            pot_key = round(rec["potential_V"] / pot_bin_width) * pot_bin_width
            if pot_key not in pot_map:
                pot_map[pot_key] = {}
            pot_map[pot_key][label] = rec

    labels = list(root_dirs.keys())
    common_potentials = sorted([p for p, d in pot_map.items() if all(lbl in d for lbl in labels)])
    common_map = {p: pot_map[p] for p in common_potentials}
    return common_map, common_potentials


# ---------------------------------------------------------------------------
# FIXED NONNEGATIVE DRT IMPLEMENTATION
# ---------------------------------------------------------------------------
def compute_drt_for_file(file_path):
    """
    Fixed-regularization DRT (non-negative) adapted from Auto_DRT_t1.
    """
    df = pd.read_csv(file_path)
    freq = df["freq_Hz"].to_numpy()
    Z = df["Z_real_Ohm"].to_numpy() + 1j * df["Z_imag_Ohm"].to_numpy()

    mask = np.isfinite(freq) & np.isfinite(Z.real) & np.isfinite(Z.imag)
    f = freq[mask]
    Z = Z[mask]
    sort_idx = np.argsort(f)[::-1]
    f, Z = f[sort_idx], Z[sort_idx]

    omega = 2 * np.pi * f
    tau = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), N_TAU)

    n_omega = len(omega)
    n_tau = len(tau)

    A_real = np.zeros((n_omega, n_tau))
    A_imag = np.zeros((n_omega, n_tau))
    for i, w in enumerate(omega):
        denom = 1.0 + (w * tau) ** 2
        A_real[i, :] = 1.0 / denom
        A_imag[i, :] = -(w * tau) / denom

    A = np.vstack([
        np.hstack([np.ones((n_omega, 1)), A_real]),
        np.hstack([np.zeros((n_omega, 1)), A_imag]),
    ])
    b = np.concatenate([Z.real, Z.imag])

    L = np.eye(n_tau + 1)
    L[0, 0] = 0
    AtA = A.T @ A + (LAMBDA ** 2) * (L.T @ L)
    Atb = A.T @ b

    x = np.linalg.solve(AtA, Atb)
    gamma = x[1:]
    gamma = np.clip(gamma, 0, None)  # enforce nonnegative DRT
    return tau, gamma


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def plot_drt_comparison(common_map, common_potentials, root_dirs):
    if not common_potentials:
        raise ValueError("No common potentials found.")

    cell_labels = list(root_dirs.keys())
    styles = {cell_labels[0]: "-", cell_labels[1]: "--"}
    cmap = plt.get_cmap("viridis")
    vmin, vmax = min(common_potentials), max(common_potentials)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 7))
    for pot in common_potentials:
        color = cmap(norm(pot))
        for cell in cell_labels:
            tau, gamma = compute_drt_for_file(common_map[pot][cell]["path"])
            ax.plot(tau, gamma, styles[cell], color=color, linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau$ / s", fontsize=12)
    ax.set_ylabel(r"$\gamma(\tau)$ / $\Omega$", fontsize=12)
    ax.set_title("DRT Comparison (Nonnegative) — HZ01 vs HY01, 2.7–3.5 V")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Potential / V")

    legend = [
        Line2D([0], [0], color="k", linestyle=styles[cell_labels[0]], label=cell_labels[0]),
        Line2D([0], [0], color="k", linestyle=styles[cell_labels[1]], label=cell_labels[1]),
    ]
    ax.legend(handles=legend, title="Cell", loc="best")

    ax.grid(True, which="both", linestyle=":")
    fig.tight_layout()
    plt.show()
# Add a cell_label argument to the function
def plot_drt_single_cell(common_map, common_potentials, cell_label):
    if not common_potentials:
        raise ValueError("No common potentials found.")

    style = "-"  # or "--" as needed
    cmap = plt.get_cmap("viridis")
    vmin, vmax = min(common_potentials), max(common_potentials)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 7))
    for pot in common_potentials:
        color = cmap(norm(pot))
        tau, gamma = compute_drt_for_file(common_map[pot][cell_label]["path"])
        ax.plot(tau, gamma, style, color=color, linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau$ / s", fontsize=12)
    ax.set_ylabel(r"$\gamma(\tau)$ / $\Omega$", fontsize=12)
    ax.set_title(f"DRT (Nonnegative) — {cell_label}")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Potential / V")

    ax.legend([cell_label], title="Cell", loc="best")
    ax.grid(True, which="both", linestyle=":")
    fig.tight_layout()
    plt.show()

# In main(), call with the desired cell label, e.g.:
# plot_drt_single_cell(common_map, common_potentials, "HZ01")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    common_map, common_potentials = build_common_potential_map(
        ROOT_DIRS,
        CHANNEL_PATTERNS,
        CYCLE_STEP,
        POT_MIN_V,
        POT_MAX_V,
        POT_BIN_WIDTH,
    )

    print("Common potentials:")
    for pot in common_potentials:
        print(f"  {pot:.3f} V")
        for lbl in ROOT_DIRS:
            print(f"    {lbl}: {common_map[pot][lbl]['path']}")

    plot_drt_comparison(common_map, common_potentials, ROOT_DIRS)
    #plot_drt_single_cell(common_map, common_potentials, "HZ01")


if __name__ == "__main__":
    main()
