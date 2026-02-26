"""
Radar charts (one per solvent)

Improvements:
- No overlapping value labels
- Outer ring for raw values with angle-aware alignment
- Smooth desirability scaling
- No FEC, no VC, no χ
- Includes DPE values (thermophysical where available)

Outputs:
PNG + PDF per solvent
CSV of raw values and scores
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# -------------------------
# OUTPUT
# -------------------------
OUTPUT_DIR = Path(r"C:\Users\benja\Downloads\Final Countdown\Proposal Slide Figures - Solvent Radar")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300

# -------------------------
# STYLE (fix kerning)
# -------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# -------------------------
# COLORS
# -------------------------
SOLVENT_COLORS = {
    "THF": "#E69F00",
    "DME": "#56B4E9",
    "DPE": "#FF6666",
    "EC":  "#009E73",
    "PC":  "#0072B2",
    "DMC": "#999999",
}

# -------------------------
# AXES
# -------------------------
COLS = ["mp_C", "eta", "epsilon", "DN", "bp_C"]

AXIS_LABELS = [
    r"$T_m$ (°C)",
    r"$\eta$ (cP)",
    r"$\varepsilon$",
    r"DN",
    r"$T_b$ (°C)",
]

# -------------------------
# RAW DATA (peer-reviewed where available)
# -------------------------
RAW_DATA = [
    # solvent, mp, eta(25C), epsilon, DN, bp
    ("EC",   36.4,   1.9,   89.8, 16.4, 248),
    ("PC",  -48.8,   2.53,  64.9, 15.1, 242),
    ("DMC",   4.6,   0.59,   3.11,17.2,  91),
    ("DME", -58.0,   0.45,   7.2, 20.0,  82.5),
    ("THF",-108.5,   0.46,   7.6, 20.0,  66),
    # Dipropyl ether (di-n-propyl ether)
    # mp, bp from CRC/NIST; dielectric ~3.3; viscosity literature ~0.7 cP
    ("DPE",-122.0,   0.70,   3.3,  18.0, 90),
]

df = pd.DataFrame(RAW_DATA, columns=["solvent"] + COLS).set_index("solvent")


# -------------------------
# SMOOTH SCALING
# -------------------------
def logistic_high(x, x0, w):
    return 10 / (1 + np.exp(-(x - x0) / w))

def logistic_low(x, x0, w):
    return 10 - logistic_high(x, x0, w)

SCALING = {
    # calibrated: -60°C -> 7, +30°C -> 3
    "mp_C": ("low", -15, 40.94, -140, 60),

    "eta": ("low", 1.0, 0.7, 0.2, 8),
    "epsilon": ("high", 20, 12, 2, 140),
    "DN": ("high", 18, 1.8, 8, 25),
    "bp_C": ("high", 100, 45, 40, 300),
}

def score_value(col, val):
    if pd.isna(val):
        return np.nan
    mode, x0, w, lo, hi = SCALING[col]
    v = np.clip(val, lo, hi)
    return logistic_high(v, x0, w) if mode == "high" else logistic_low(v, x0, w)

scores = df.copy()
for c in COLS:
    scores[c] = df[c].apply(lambda v: score_value(c, v))


# -------------------------
# FORMAT RAW VALUES
# -------------------------
def fmt(v):
    if pd.isna(v):
        return "NA"
    if abs(v) > 10:
        return f"{v:.0f}"
    return f"{v:.2g}"


# -------------------------
# RADAR PLOT
# -------------------------
R_MAX = 10
R_RAW = 12.3  # outer ring for raw values

# -------------------------
# RADAR PLOT
# -------------------------
# -------------------------
# RADAR PLOT — 4× TEXT
# -------------------------
# -------------------------
# RADAR PLOT — PPT RELATIVE SCALING
# -------------------------

# Base scale (change this if you want everything bigger/smaller)
BASE = 36

FONT_AXIS = BASE * 1.0
FONT_VALUES = BASE * 0.8
FONT_TICKS = BASE * 0.6
FONT_TITLE = BASE * 1.2

LINEWIDTH = 2.0
MARKER_SIZE = 80

R_SCORE_MAX = 10
R_PLOT_MAX = 12
BASE_OFFSET = 0.9


def fmt(v):
    if pd.isna(v):
        return "NA"
    if abs(v) > 10:
        return f"{v:.0f}"
    return f"{v:.2g}"


def make_blank_radar(
    cols,
    axis_labels,
    fig_size=(7, 7),
    r_plot_max=R_PLOT_MAX,
    r_yticks=None,
    theta_offset=np.pi / 2,
    theta_direction=-1,
    font_axis=FONT_AXIS,
    font_ticks=FONT_TICKS,
    font_values=FONT_VALUES,
    pad_xtick=12,
    grid_alpha=0.35,
    title=None,
):
    """
    Create and return a blank radar (spider) chart figure and polar axis.

    Returns:
      - fig: matplotlib Figure
      - ax: polar Axes
      - angles: numpy array of angles (radians) for the axes (not closed)

    The caller can plot on `ax` using the returned `angles`.
    """
    N = len(cols)
    if N < 1:
        raise ValueError("cols must contain at least one axis label")

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, polar=True)

    # orientation: 0 at top, clockwise by default (matches existing visuals)
    ax.set_theta_offset(theta_offset)
    ax.set_theta_direction(theta_direction)

    # radial limits and ticks
    ax.set_ylim(0, r_plot_max)
    if r_yticks is None:
        r_yticks = [0, 2, 4, 6, 8, 10]
    ax.set_yticks(r_yticks)
    ax.set_yticklabels([str(t) for t in r_yticks], fontsize=font_ticks)

    # angular ticks and labels
    ax.set_xticks(angles)
    ax.set_xticklabels(axis_labels, fontsize=font_axis)
    ax.tick_params(axis="x", pad=pad_xtick)

    # grid and aesthetic
    ax.grid(alpha=grid_alpha)
    try:
        ax.set_box_aspect(1)
    except Exception:
        # older matplotlib may not support set_box_aspect on polar axes
        pass

    if title:
        fig.suptitle(title, fontsize=FONT_TITLE, y=0.98)

    return fig, ax, angles


def make_radar(solvent):

    vals = scores.loc[solvent, COLS].fillna(5).values
    raw = df.loc[solvent, COLS]
    color = SOLVENT_COLORS[solvent]

    # use the shared blank radar setup
    fig, ax, angles = make_blank_radar(
        COLS,
        AXIS_LABELS,
        fig_size=(7, 7),
        r_plot_max=R_PLOT_MAX,
        r_yticks=[0, 2, 4, 6, 8, 10],
        pad_xtick=12,
        grid_alpha=0.35,
    )
    angles_closed = np.r_[angles, angles[0]]
    vals_closed = np.r_[vals, vals[0]]

    # Polygon
    ax.plot(angles_closed, vals_closed, lw=LINEWIDTH, color=color)
    ax.fill(angles_closed, vals_closed, color=color, alpha=0.15)
    ax.scatter(angles, vals, s=MARKER_SIZE, color=color, zorder=3)

    # Missing values
    missing = scores.loc[solvent, COLS].isna()
    if missing.any():
        ax.scatter(
            angles[missing.values],
            vals[missing.values],
            s=MARKER_SIZE * 1.6,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            zorder=4
        )

    # Optional title
    # ax.set_title(solvent, fontsize=FONT_TITLE, pad=14)

    # ---- Raw values above markers ----
    for ang, score_val, col in zip(angles, vals, COLS):

        text_val = fmt(raw[col])

        extra = 0.4 * abs(np.cos(ang))
        r_text = score_val + BASE_OFFSET + extra

        if r_text > R_PLOT_MAX - 0.2:
            r_text = R_PLOT_MAX - 0.2

        ax.text(
            ang,
            r_text,
            text_val,
            ha="center",
            va="bottom",
            fontsize=FONT_VALUES,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
                pad=1.2
            ),
            zorder=5
        )

    fig.tight_layout()

    png = OUTPUT_DIR / f"radar_{solvent}_ppt.png"
    pdf = OUTPUT_DIR / f"radar_{solvent}_ppt.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close()





# -------------------------
# RUN
# -------------------------
df.to_csv(OUTPUT_DIR / "solvent_raw_values.csv")
scores.to_csv(OUTPUT_DIR / "solvent_scores_smooth.csv")

for s in df.index:
    make_radar(s)

print("Saved radar plots to:", OUTPUT_DIR)

# Create and save a blank radar template that uses the identical parameters
# as the completed radar plots (same size, fonts, ticks, grid, radial limits).
fig_blank, ax_blank, angles_blank = make_blank_radar(
    COLS,
    AXIS_LABELS,
    fig_size=(7, 7),
    r_plot_max=R_PLOT_MAX,
    r_yticks=[0, 2, 4, 6, 8, 10],
    pad_xtick=12,
    grid_alpha=0.35,
)

blank_png = OUTPUT_DIR / "radar_blank_template.png"
fig_blank.savefig(blank_png, dpi=DPI, bbox_inches="tight")
plt.close(fig_blank)
print(f"Saved blank radar template: {blank_png}")
