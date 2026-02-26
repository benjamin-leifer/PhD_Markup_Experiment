"""
EIS .mpt Parser and Equivalent Circuit Fitting
EC-Lab staircase EIS (BL-LL-CX02_EIS_STAIR_RT_07 format)

Equivalent circuit: Rs + (Rct || CPE) + Warburg (Randles + CPE)
    - Rs  : ohmic/electrolyte resistance
    - Rct : charge-transfer resistance
    - CPE : constant phase element (Q, n)   replaces ideal capacitor
    - W   : semi-infinite Warburg (diffusion at low freq)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
import warnings
warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
# 1.  FILE PARSER
# ────────────────────────────────────────────────────────────────────────────

def parse_mpt(filepath: str) -> pd.DataFrame:
    """Parse an EC-Lab ASCII .mpt file into a DataFrame."""
    with open(filepath, "r", encoding="latin-1") as fh:
        lines = fh.readlines()

    # Locate header length
    nb_header = None
    for ln in lines:
        if ln.startswith("Nb header lines"):
            nb_header = int(ln.split(":")[1].strip())
            break
    if nb_header is None:
        raise ValueError("Cannot find 'Nb header lines' in file.")

    data_block = lines[nb_header:]
    col_names = [c.strip() for c in data_block[0].split("\t")]

    rows = []
    for ln in data_block[1:]:
        if ln.strip():
            rows.append(ln.strip().split("\t"))

    df = pd.DataFrame(rows, columns=col_names)
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except Exception:
            pass
    return df


# ────────────────────────────────────────────────────────────────────────────
# 2.  SELECT MID-SOC CYCLE
# ────────────────────────────────────────────────────────────────────────────

def select_mid_soc(df: pd.DataFrame, fraction: float = 0.5) -> tuple[pd.DataFrame, float, float]:
    """
    Pick the cycle whose mean <Ewe> is closest to fraction of the
    full voltage window (default 0.5 → mid-SOC).

    Returns: (spectrum_df, cycle_id, mean_voltage)
    """
    cycle_v = df.groupby("cycle number")["<Ewe>/V"].mean()
    v_lo, v_hi = cycle_v.min(), cycle_v.max()
    v_target = v_lo + fraction * (v_hi - v_lo)

    best_cycle = (cycle_v - v_target).abs().idxmin()
    v_sel = cycle_v[best_cycle]

    spec = df[df["cycle number"] == best_cycle].copy()

    print(f"Voltage window : {v_lo:.4f} – {v_hi:.4f} V")
    print(f"Target voltage : {v_target:.4f} V  (fraction = {fraction})")
    print(f"Selected cycle : {int(best_cycle)}  (Ewe = {v_sel:.4f} V)\n")
    return spec, best_cycle, v_sel


# ────────────────────────────────────────────────────────────────────────────
# 3.  EQUIVALENT CIRCUIT MODEL
# ────────────────────────────────────────────────────────────────────────────
#
#   Z(ω) = Rs  +  Rct / (1 + Rct·Q·(jω)^n)  +  Aw / sqrt(jω)
#
#   Parameters: Rs, Rct, Q, n, Aw
# ────────────────────────────────────────────────────────────────────────────

def Z_model(omega, Rs, Rct, Q, n, Aw):
    jw = 1j * omega
    Z_rct_cpe = Rct / (1.0 + Rct * Q * jw**n)   # Rct ∥ CPE
    Z_W       = Aw / np.sqrt(jw)                  # Warburg
    return Rs + Z_rct_cpe + Z_W


def cost_fn(params, omega, Z_meas):
    """Modulus-weighted sum of squared residuals (real + imag)."""
    Rs, Rct, Q, n, Aw = params
    if not (0 < n <= 1):
        return 1e12
    Z_calc = Z_model(omega, Rs, Rct, Q, n, Aw)
    delta  = Z_calc - Z_meas
    w      = 1.0 / (np.abs(Z_meas)**2 + 1e-30)   # modulus weighting
    return float(np.sum(w * (delta.real**2 + delta.imag**2)))


def fit_eis(freq, Z_re, Z_neg_im):
    """
    Fit Randles+CPE model.
    Inputs  : freq [Hz], Z_re [Ω], Z_neg_im [Ω]  (= -Im(Z), column as-is)
    Returns : (params_dict, Z_fit_complex at input freqs)
    """
    omega  = 2 * np.pi * freq
    Z_meas = Z_re - 1j * Z_neg_im        # reconstruct complex Z

    # ── coarse initial guess from data ──────────────────────────────────────
    Rs0  = float(Z_re[np.argmax(freq)])   # HF intercept
    Rct0 = float(Z_re.max() - Rs0)        # approximate semicircle width
    Q0   = 1e-4
    n0   = 0.8
    Aw0  = 5.0

    # ── global optimisation (avoids local minima) ────────────────────────────
    bounds = [
        (max(0, Rs0 * 0.5),  Rs0  * 3 + 50),   # Rs
        (0.1,                 Rct0 * 5 + 200),   # Rct
        (1e-7,  1e-1),                            # Q
        (0.5,   1.0),                             # n
        (0.0,   500.0),                           # Aw
    ]

    print("Running global optimisation (DE) …")
    res_de = differential_evolution(
        cost_fn, bounds,
        args=(omega, Z_meas),
        seed=0, maxiter=2000, tol=1e-10,
        popsize=20, mutation=(0.5, 1.5), recombination=0.9,
        updating="deferred", workers=1,
    )

    # ── local polish ─────────────────────────────────────────────────────────
    res_loc = minimize(
        cost_fn, res_de.x,
        args=(omega, Z_meas),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 50000},
    )
    p = res_loc.x
    Rs, Rct, Q, n, Aw = p

    # ── derived quantities ────────────────────────────────────────────────────
    omega_c = (1.0 / (Rct * Q)) ** (1.0 / n)    # CPE characteristic freq
    f_c     = omega_c / (2 * np.pi)

    params = dict(Rs=Rs, Rct=Rct, Q=Q, n=n, Aw=Aw,
                  f_characteristic_Hz=f_c,
                  cost=res_loc.fun)

    Z_fit = Z_model(omega, Rs, Rct, Q, n, Aw)
    return params, Z_fit, Z_meas


# ────────────────────────────────────────────────────────────────────────────
# 4.  PLOTTING
# ────────────────────────────────────────────────────────────────────────────

def plot_results(freq, Z_re, Z_neg_im, Z_meas, Z_fit,
                 params, cycle_id, v_sel, omega_fit=None, Z_fit_dense=None):

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        f"EIS — Cycle {int(cycle_id)} (mid-SOC)   Ewe = {v_sel:.3f} V",
        fontsize=13, fontweight="bold"
    )

    ax1 = fig.add_subplot(1, 2, 1)                     # Nyquist
    ax2 = fig.add_subplot(2, 2, 2)                     # Bode |Z|
    ax3 = fig.add_subplot(2, 2, 4, sharex=ax2)        # Bode phase

    # ── Nyquist ──────────────────────────────────────────────────────────────
    sc = ax1.scatter(Z_re, Z_neg_im,
                     c=np.log10(freq), cmap="viridis_r",
                     s=40, zorder=5, label="Data")
    if Z_fit_dense is not None and omega_fit is not None:
        ax1.plot(Z_fit_dense.real, -Z_fit_dense.imag,
                 "r-", lw=2, label="Fit")
    ax1.plot(Z_fit.real, -Z_fit.imag, "r.", ms=6, alpha=0.4)

    cb = plt.colorbar(sc, ax=ax1, pad=0.01)
    cb.set_label("log₁₀(f / Hz)", fontsize=9)

    # Mark Rs, Rs+Rct
    Rs, Rct = params["Rs"], params["Rct"]
    ax1.axvline(Rs,       color="gray",  ls="--", lw=1, alpha=0.7, label=f"Rs = {Rs:.3f} Ω")
    ax1.axvline(Rs + Rct, color="olive", ls="--", lw=1, alpha=0.7,
                label=f"Rs+Rct = {Rs+Rct:.3f} Ω")

    ax1.set_xlabel("Re(Z) / Ω",  fontsize=11)
    ax1.set_ylabel("−Im(Z) / Ω", fontsize=11)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)

    # Parameter box
    txt = (f"Rs  = {Rs:.4f} Ω\n"
           f"Rct = {Rct:.4f} Ω\n"
           f"Q   = {params['Q']:.3e} S·sⁿ\n"
           f"n   = {params['n']:.4f}\n"
           f"Aw  = {params['Aw']:.4f} Ω·s⁻⁰·⁵\n"
           f"fc  = {params['f_characteristic_Hz']:.3f} Hz")
    ax1.text(0.98, 0.97, txt, transform=ax1.transAxes, fontsize=8,
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.9))

    # ── Bode |Z| ─────────────────────────────────────────────────────────────
    ax2.semilogx(freq, np.abs(Z_meas), "o", ms=4, color="steelblue", label="Data")
    if Z_fit_dense is not None:
        f_dense = omega_fit / (2 * np.pi)
        ax2.semilogx(f_dense, np.abs(Z_fit_dense), "r-", lw=1.8, label="Fit")
    ax2.set_ylabel("|Z| / Ω", fontsize=10)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ── Bode phase ───────────────────────────────────────────────────────────
    ax3.semilogx(freq, np.angle(Z_meas, deg=True), "s", ms=4, color="darkorange", label="Data")
    if Z_fit_dense is not None:
        ax3.semilogx(f_dense, np.angle(Z_fit_dense, deg=True), "r-", lw=1.8, label="Fit")
    ax3.set_xlabel("Frequency / Hz", fontsize=10)
    ax3.set_ylabel("Phase / °",      fontsize=10)
    ax3.grid(True, which="both", alpha=0.25)
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("eis_midSOC_fit.png", dpi=150, bbox_inches="tight")
    print("Plot saved → eis_midSOC_fit.png")
    plt.show()


# ────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    FILEPATH = "BL-LL-CX02_EIS_STAIR_RT_07_PEIS_C02.mpt"   # ← change as needed

    # ── Load & select ─────────────────────────────────────────────────────────
    df = parse_mpt(FILEPATH)
    spec, cycle_id, v_sel = select_mid_soc(df, fraction=0.5)

    # ── Extract columns ───────────────────────────────────────────────────────
    freq     = spec["freq/Hz"].values
    Z_re     = spec["Re(Z)/Ohm"].values
    Z_neg_im = spec["-Im(Z)/Ohm"].values       # already -Im(Z), use directly for Nyquist

    # ── Quality filter ────────────────────────────────────────────────────────
    # Keep frequencies where data looks physically meaningful:
    #   • Re(Z) > 0  (causal)
    #   • frequency between 50 mHz and 200 kHz
    mask = (
        (freq  >= 5e-2)  &
        (freq  <= 2e5)   &
        (Z_re  >  0)
    )
    freq, Z_re, Z_neg_im = freq[mask], Z_re[mask], Z_neg_im[mask]

    print(f"Points used for fit : {mask.sum()}")
    print(f"Frequency range     : {freq.min():.4f} – {freq.max():.0f} Hz\n")

    # ── Fit ───────────────────────────────────────────────────────────────────
    params, Z_fit, Z_meas = fit_eis(freq, Z_re, Z_neg_im)

    print("\n══════════════ Fit Results ══════════════")
    print(f"  Rs   (ohmic)         = {params['Rs']:.5f}  Ω")
    print(f"  Rct  (charge-trans.) = {params['Rct']:.5f}  Ω")
    print(f"  Q    (CPE prefactor) = {params['Q']:.4e}  S·sⁿ")
    print(f"  n    (CPE exponent)  = {params['n']:.5f}  (1=cap, 0.5=Warb)")
    print(f"  Aw   (Warburg)       = {params['Aw']:.5f}  Ω·s⁻⁰·⁵")
    print(f"  fc   (CPE char. f)   = {params['f_characteristic_Hz']:.4f}  Hz")
    print(f"  Cost (weighted SSR)  = {params['cost']:.4e}")
    print("═════════════════════════════════════════\n")

    # ── Dense fit curve for smooth plot ──────────────────────────────────────
    omega_fit   = 2 * np.pi * np.logspace(
                      np.log10(freq.min()), np.log10(freq.max()), 600)
    Z_fit_dense = Z_model(omega_fit,
                          params["Rs"], params["Rct"],
                          params["Q"],  params["n"], params["Aw"])

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(freq, Z_re, Z_neg_im, Z_meas, Z_fit,
                 params, cycle_id, v_sel, omega_fit, Z_fit_dense)