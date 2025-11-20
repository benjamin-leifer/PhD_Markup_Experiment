"""
Auto_DRT_t1.py

Compute and plot DRT spectra from EIS CSV files produced by SPEIS_handler.py.

Expected input CSV columns:
    - "freq_Hz"      : frequency in Hz
    - "Z_real_Ohm"   : real part of impedance (Ohm)
    - "Z_imag_Ohm"   : imaginary part of impedance (Ohm) (negative for capacitive loops)

The script:
    1. Builds a file list of matching CSVs in ROOT_DIR.
    2. Prints the full list to the console.
    3. Computes a Tikhonov-regularized DRT for every Nth spectrum
       (e.g., cycles 1, 5, 10, ... via STEP_EVERY_N).
    4. Plots overlaid DRT spectra (tau vs gamma) in a style similar to slide 5.
"""

import os
import glob
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# USER CONFIGURATION
# -----------------------------------------------------------------------------

# Folder containing the *_SPEIS_C01_cycleXXX_±XXXXmV.csv files from SPEIS_handler
ROOT_DIR = r"C:\Users\benja\Downloads\DRT EIS Stair 1\11\EIS Formation Stair\HY01\HY01_RT_EIS_split"  # <-- EDIT THIS

# Glob pattern for the EIS CSV files you want to include
FILE_GLOB = "*_SPEIS_C01_cycle*.csv"

# Take every Nth spectrum (0-based indices: 0, N, 2N, ...)
# For cycles 1, 5, 10, ... set STEP_EVERY_N = 5
STEP_EVERY_N = 4

# Optional hard cap on number of spectra to plot (set to None for no cap)
MAX_SPECTRA = 14  # e.g., 10 or None

# DRT inversion parameters
N_TAU = 200          # number of tau points in the DRT grid
REG_LAMBDA = 0.05    # Tikhonov regularization strength (tune as needed)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def find_eis_files(root_dir: str, pattern: str) -> List[str]:
    """
    Return a sorted list of EIS CSV files matching the glob pattern.
    """
    search_pattern = os.path.join(root_dir, pattern)
    file_list = sorted(glob.glob(search_pattern))
    return file_list


def parse_potential_from_name(filename: str) -> str:
    """
    Extract a label like '+2.30 V' from a filename containing '..._+2300mV.csv'.
    Falls back to basename if no match.
    """
    base = os.path.basename(filename)
    m = re.search(r'([+-]\d+)mV', base)
    if m:
        mV = int(m.group(1))
        return f"{mV / 1000.0:.2f} V"
    return base


def compute_drt(
    freq_hz: np.ndarray,
    z_real: np.ndarray,
    z_imag: np.ndarray,
    n_tau: int = N_TAU,
    reg_lambda: float = REG_LAMBDA,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Basic Tikhonov-regularized DRT inversion.

    Parameters
    ----------
    freq_hz : array
        Frequencies in Hz (positive).
    z_real, z_imag : array
        Real and imaginary parts of Z(ω) in Ohm.
    n_tau : int
        Number of tau points in the DRT grid.
    reg_lambda : float
        Regularization parameter.

    Returns
    -------
    tau : array
        Relaxation times (s) on a log-spaced grid.
    gamma : array
        DRT amplitude γ(τ) in Ohm.
    R_inf : float
        Estimated high-frequency resistance.
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    z_real = np.asarray(z_real, dtype=float)
    z_imag = np.asarray(z_imag, dtype=float)

    # Clean up inputs
    mask = (
        np.isfinite(freq_hz)
        & np.isfinite(z_real)
        & np.isfinite(z_imag)
        & (freq_hz > 0)
    )
    freq_hz = freq_hz[mask]
    z_real = z_real[mask]
    z_imag = z_imag[mask]

    if freq_hz.size < 5:
        raise ValueError("Not enough valid points to compute DRT.")

    omega = 2 * np.pi * freq_hz
    Z = z_real + 1j * z_imag

    # Tau grid from the frequency range (slightly extended)
    f_min = freq_hz.min()
    f_max = freq_hz.max()
    tau_min = 0.1 / (2 * np.pi * f_max)
    tau_max = 10.0 / (2 * np.pi * f_min)
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)

    # Kernel: Z(ω) ≈ R_inf + ∑ γ(τ_k) / (1 + j ω τ_k) Δln τ
    K = 1.0 / (1.0 + 1j * omega[:, None] * tau[None, :])

    # Estimate R_inf as minimum Re(Z)
    R_inf = float(np.min(z_real))
    Z_prime = Z - R_inf

    # Build real-valued least-squares system
    A_real = K.real
    A_imag = K.imag
    b_real = Z_prime.real
    b_imag = Z_prime.imag

    A = np.vstack([A_real, A_imag])
    b = np.hstack([b_real, b_imag])

    # Scale to improve conditioning
    scale = np.max(np.abs(b)) or 1.0
    A_scaled = A / scale
    b_scaled = b / scale

    # Ridge regression (zero-order Tikhonov)
    AtA = A_scaled.T @ A_scaled + (reg_lambda ** 2) * np.eye(n_tau)
    Atb = A_scaled.T @ b_scaled

    gamma = np.linalg.solve(AtA, Atb)
    gamma = np.clip(gamma, 0, None)  # non-negativity constraint (optional)

    return tau, gamma, R_inf


def plot_drt_overlay(file_list: List[str], indices_to_plot: List[int]) -> None:
    """
    Compute and plot DRT spectra for the files in `file_list` at the
    positions given in `indices_to_plot`.

    The plot style is loosely inspired by the DRT plots on slide 5:
    - x-axis: relaxation time τ (log-scale)
    - y-axis: γ(τ) (Ohm)
    - overlaid curves for multiple potentials
    - shaded regions annotated as 'SEI', 'Charge transfer', 'Diffusion', etc.
    """
    if not file_list or not indices_to_plot:
        print("No files/indices to plot.")
        return

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Color map for multiple spectra
    cmap = plt.get_cmap("viridis")
    num_to_plot = len(indices_to_plot)

    all_taus = []
    all_gammas = []
    y_max = 0.0

    for rank, idx in enumerate(indices_to_plot):
        file_path = file_list[idx]
        print(f"Processing [{idx}] {file_path}")
        df = pd.read_csv(file_path)

        if not {"freq_Hz", "Z_real_Ohm", "Z_imag_Ohm"}.issubset(df.columns):
            raise KeyError(
                f"File '{file_path}' is missing one of the required columns: "
                "'freq_Hz', 'Z_real_Ohm', 'Z_imag_Ohm'"
            )

        freq_hz = df["freq_Hz"].to_numpy()
        z_real = df["Z_real_Ohm"].to_numpy()
        z_imag = df["Z_imag_Ohm"].to_numpy()

        tau, gamma, R_inf = compute_drt(freq_hz, z_real, z_imag)

        all_taus.append(tau)
        all_gammas.append(gamma)
        y_max = max(y_max, float(np.max(gamma)))

        label = parse_potential_from_name(file_path)
        color = cmap(rank / max(1, num_to_plot - 1))

        ax.semilogx(
            tau,
            gamma,
            label=label,
            linewidth=2.5,
            alpha=0.9,
            color=color,
        )

    if not all_taus:
        print("No spectra successfully processed — skipping plot.")
        return

    # Axis labels and styling
    ax.set_xlabel(r"Relaxation time, $\tau$ (s)", fontsize=12)
    ax.set_ylabel(r"DRT amplitude, $\gamma(\tau)$ (Ω)", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(title="EIS step potential", fontsize=9)
    ax.set_title("DRT spectra from SPEIS stair-step EIS", fontsize=13)

    # Shaded regions (approximate, adjust to taste)
    tau_min_global = min(t.min() for t in all_taus)
    tau_max_global = max(t.max() for t in all_taus)
    y_top = y_max * 1.05

    # Define nominal tau windows (s)
    regions = [
        (1e-6, 1e-4, "SEI"),
        (1e-4, 1e-2, "Charge transfer"),
        (1e-2, 1e+1, "Diffusion"),
    ]

    for (t1, t2, label) in regions:
        # Clip to actual tau range
        t1c = max(t1, tau_min_global)
        t2c = min(t2, tau_max_global)
        if t1c >= t2c:
            continue

        ax.axvspan(t1c, t2c, color="grey", alpha=0.08, zorder=0)

        x_text = 10 ** ((np.log10(t1c) + np.log10(t2c)) / 2.0)
        ax.text(
            x_text,
            0.9 * y_top,
            label,
            ha="center",
            va="top",
            fontsize=9,
            rotation=0,
        )

    ax.set_ylim(bottom=0, top=y_top)
    fig.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main() -> None:
    # Build and print the full file list
    file_list = find_eis_files(ROOT_DIR, FILE_GLOB)

    if not file_list:
        print(f"No files found in '{ROOT_DIR}' matching pattern '{FILE_GLOB}'.")
        return

    print("\nDetected EIS CSV files:")
    for idx, fp in enumerate(file_list):
        print(f"  [{idx:02d}] {fp}")

    # Choose every STEP_EVERY_N-th spectrum: indices 0, STEP, 2*STEP, ...
    indices_to_plot = list(range(0, len(file_list), STEP_EVERY_N))

    # Apply optional MAX_SPECTRA cap
    if MAX_SPECTRA is not None:
        indices_to_plot = indices_to_plot[:MAX_SPECTRA]

    print(f"\nWill plot spectra at file indices (0-based): {indices_to_plot}")
    print(f"STEP_EVERY_N = {STEP_EVERY_N}, MAX_SPECTRA = {MAX_SPECTRA}\n")

    # Plot DRT overlay for those selected files
    plot_drt_overlay(file_list, indices_to_plot)


if __name__ == "__main__":
    main()
