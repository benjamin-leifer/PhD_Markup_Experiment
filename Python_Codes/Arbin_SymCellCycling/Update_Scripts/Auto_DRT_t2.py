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
    "JB02": r"C:\Users\benja\Downloads\EIS Formation Stair\JB02\DRTools_output",
    "IZ06":r"C:\Users\benja\Downloads\EIS Formation Stair\IZ06\DRTools_output",
}

CHANNEL_PATTERNS = {
    "HZ01": "SPEIS_C02_cycle",
    "HY01": "SPEIS_C01_cycle",
    "JB02": "SPEIS_C02_cycle",
    "IZ06": "SPEIS_C02_cycle"
}

CYCLE_STEP = 8
POT_MIN_V = 2.0
POT_MAX_V = 4.1
POT_BIN_WIDTH = 0.005  # ~5 mV tolerance

# Fixed DRT parameters (from Auto_DRT_t1)
N_TAU = 200
LAMBDA = 0.01
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


# python
def build_common_potential_map(root_dirs, channel_patterns, cycle_step, pot_min_v, pot_max_v, pot_bin_width):
    # (unchanged scanning code above)
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

    # return the full pot_map as well as the common_map and list
    return pot_map, common_map, common_potentials


def plot_drt_single_cell_stepwise(pot_map, cell_label, pot_step=0.05):
    """
    Plot DRT for `cell_label` at potentials spaced by ~pot_step (V).
    Accepts the full `pot_map` (all binned potentials); selects potentials
    present for `cell_label` spaced by at least `pot_step`.
    """
    pots = sorted([p for p, d in pot_map.items() if cell_label in d])
    if not pots:
        raise ValueError(f"No potentials found for cell {cell_label}.")

    selected = [pots[0]]
    for p in pots[1:]:
        if p - selected[-1] >= pot_step - 1e-9:
            selected.append(p)

    cmap = plt.get_cmap("viridis")
    vmin, vmax = min(selected), max(selected)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 7))
    for pot in selected:
        color = cmap(norm(pot))
        tau, gamma = compute_drt_for_file(pot_map[pot][cell_label]["path"])
        ax.plot(tau, gamma, "-", color=color, linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel(r"\$\\tau\$ / s", fontsize=12)
    ax.set_ylabel(r"\$\\gamma(\\tau)\$ / \$\\Omega\$", fontsize=12)
    ax.set_title(f"DRT (Nonnegative) — {cell_label} every {int(pot_step*1000)} mV")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Potential / V")

    ax.legend([cell_label], title="Cell", loc="best")
    ax.grid(True, which="both", linestyle=":")
    fig.tight_layout()
    plt.show()



# ---------------------------------------------------------------------------
# FIXED NONNEGATIVE DRT IMPLEMENTATION
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# IMPROVED NONNEGATIVE DRT (NNLS + smoothness + auto-tau)
# ---------------------------------------------------------------------------

def _second_derivative_matrix(n: int) -> np.ndarray:
    """
    2nd-difference operator on gamma grid (n points).
    Shape: (n-2, n). Penalizes curvature -> smooth gamma in log(tau).
    """
    D2 = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        D2[i, i]     = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0
    return D2


def _estimate_r_inf(Z: np.ndarray, hf_points: int = 5) -> float:
    """
    Estimate ohmic offset R_inf from high-frequency real impedance.
    Robust choice: median of first `hf_points` points (after sorting f desc).
    """
    hf_points = max(1, min(hf_points, Z.size))
    return float(np.median(Z.real[:hf_points]))


def _auto_tau_grid(omega: np.ndarray, n_tau: int, pad_decades: float = 1.0) -> np.ndarray:
    """
    Choose tau grid based on omega range:
      tau_min ~ 1/omega_max, tau_max ~ 1/omega_min, padded by `pad_decades`.
    """
    w_max = float(np.max(omega))
    w_min = float(np.min(omega))
    # Guard against degenerate inputs
    w_max = max(w_max, 1e-30)
    w_min = max(w_min, 1e-30)

    tau_min = (1.0 / w_max) / (10.0 ** pad_decades)
    tau_max = (1.0 / w_min) * (10.0 ** pad_decades)

    # Ensure ordering
    tau_min = max(tau_min, 1e-30)
    tau_max = max(tau_max, tau_min * 10.0)

    return np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)


def _solve_nonnegative_least_squares(A: np.ndarray, b: np.ndarray, max_iter: int = 5000) -> np.ndarray:
    """
    Solve min ||Ax - b||_2 with x >= 0.
    Uses scipy.optimize.lsq_linear if available; otherwise a simple projected GD fallback.
    """
    try:
        from scipy.optimize import lsq_linear
        res = lsq_linear(A, b, bounds=(0.0, np.inf), method="trf")
        return res.x
    except Exception:
        # Fallback: projected gradient descent (slower, but dependency-free)
        x = np.zeros(A.shape[1], dtype=float)

        # Lipschitz constant for gradient of ||Ax-b||^2 is 2*||A^T A||_2
        # We'll approximate with a few power iterations for ||A||_2, then L ~ 2*||A||_2^2
        v = np.random.default_rng(0).normal(size=A.shape[1])
        v /= (np.linalg.norm(v) + 1e-30)
        for _ in range(20):
            v = A.T @ (A @ v)
            v /= (np.linalg.norm(v) + 1e-30)
        sigma_sq = float(v @ (A.T @ (A @ v)))
        L = max(2.0 * sigma_sq, 1e-12)
        step = 1.0 / L

        for _ in range(max_iter):
            grad = 2.0 * (A.T @ (A @ x - b))
            x_new = x - step * grad
            x_new = np.clip(x_new, 0.0, None)
            # simple convergence check
            if np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-12) < 1e-6:
                x = x_new
                break
            x = x_new

        return x


def compute_drt_for_file(
    file_path: str,
    *,
    n_tau: int = None,
    lam: float = None,
    tau_pad_decades: float = 1.0,
    hf_points_rinf: int = 5,
    weight_mode: str = "1_over_absZ",  # "unity" or "1_over_absZ"
):
    """
    Nonnegative DRT via constrained least squares on an augmented system:

      min_{gamma >= 0} || W [K_real; K_imag] gamma - W [Re(Z)-R_inf; Im(Z)] ||^2
                       + || lam * D2 * gamma ||^2

    - gamma constrained with NNLS (no post-hoc clipping).
    - D2 is 2nd-difference smoothness operator (curvature penalty).
    - tau grid derived from spectrum frequency range (auto).
    - R_inf estimated from high-frequency Re(Z) and removed before inversion.
    """
    if n_tau is None:
        n_tau = N_TAU  # uses your global default :contentReference[oaicite:1]{index=1}
    if lam is None:
        lam = LAMBDA   # uses your global default :contentReference[oaicite:2]{index=2}

    df = pd.read_csv(file_path)
    freq = df["freq_Hz"].to_numpy()
    Z = df["Z_real_Ohm"].to_numpy() + 1j * df["Z_imag_Ohm"].to_numpy()

    mask = np.isfinite(freq) & np.isfinite(Z.real) & np.isfinite(Z.imag)
    f = freq[mask]
    Z = Z[mask]
    if f.size < 5:
        raise ValueError(f"Not enough valid EIS points in {file_path}")

    # Sort high->low frequency (matches your prior logic) :contentReference[oaicite:3]{index=3}
    sort_idx = np.argsort(f)[::-1]
    f, Z = f[sort_idx], Z[sort_idx]
    omega = 2 * np.pi * f

    # Auto tau grid from omega range (instead of fixed TAU_MIN/MAX) :contentReference[oaicite:4]{index=4}
    tau = _auto_tau_grid(omega, n_tau=n_tau, pad_decades=tau_pad_decades)

    # Estimate and remove R_inf (intercept handling without unconstrained variable)
    R_inf = _estimate_r_inf(Z, hf_points=hf_points_rinf)
    Z_re = Z.real - R_inf
    Z_im = Z.imag

    # Build kernels (vectorized; no python loops)
    WT = np.outer(omega, tau)                   # shape (n_omega, n_tau)
    denom = 1.0 + WT**2
    K_real = 1.0 / denom
    K_imag = -(WT) / denom

    # Stack system
    A0 = np.vstack([K_real, K_imag])            # (2*n_omega, n_tau)
    b0 = np.concatenate([Z_re, Z_im])           # (2*n_omega,)

    # Optional weighting
    if weight_mode == "1_over_absZ":
        w = 1.0 / np.clip(np.abs(Z), 1e-12, None)   # length n_omega
    elif weight_mode == "unity":
        w = np.ones_like(Z_re)
    else:
        raise ValueError("weight_mode must be 'unity' or '1_over_absZ'")

    # Apply weights to real and imag blocks
    A0[:len(omega), :] *= w[:, None]
    b0[:len(omega)]    *= w
    A0[len(omega):, :] *= w[:, None]
    b0[len(omega):]    *= w

    # Smoothness regularization (2nd derivative) instead of L = I :contentReference[oaicite:5]{index=5}
    D2 = _second_derivative_matrix(n_tau)       # (n_tau-2, n_tau)
    A_aug = np.vstack([A0, lam * D2])
    b_aug = np.concatenate([b0, np.zeros(D2.shape[0])])

    # Solve NNLS
    gamma = _solve_nonnegative_least_squares(A_aug, b_aug)

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
    pot_map, common_map, common_potentials = build_common_potential_map(
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

    # Use the full pot_map when plotting single-cell stepwise (so non-common potentials are included)
    plot_drt_single_cell_stepwise(pot_map, "JB02", pot_step=0.1)

if __name__ == "__main__":
    main()
