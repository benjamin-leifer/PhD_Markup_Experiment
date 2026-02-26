from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap


# ============================================================
# ONE KNOB
# ============================================================
FONT_BASE = 30  # <- set to 30 (or change here)

SCALE = FONT_BASE / 15.0  # baseline the script was designed around


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def fsz(mult: float) -> int:
    """Font sizes: scale around FONT_BASE, but keep sane."""
    return int(round(FONT_BASE * mult))


def diamond_s(base_s=170.0) -> float:
    """Marker area (points^2): scale gently with font (NOT linearly)."""
    return base_s * (SCALE ** 1.10)


# =====================
# Color scheme
# =====================
COL = {
    "text": "#2F3437",
    "grid": "#BFC5CC",

    "THF": "#55BBD6",   # Aim 1
    "VC":  "#3B6EA8",   # Aim 2 electrochem
    "FEC": "#9BB7E0",   # Aim 2 solv/interphase
    "DPE": "#D0A0A0",   # Aim 3

    "neutral_fill": "#E6E8EB",
    "neutral_edge": "#9AA3AD",
}


def lighten(hex_color: str, frac: float = 0.6) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * frac)
    g = int(g + (255 - g) * frac)
    b = int(b + (255 - b) * frac)
    return f"#{r:02X}{g:02X}{b:02X}"


STYLE = {
    "A1":     {"edge": COL["THF"], "fill": lighten(COL["THF"], 0.65)},
    "A2_VC":  {"edge": COL["VC"],  "fill": lighten(COL["VC"],  0.72)},
    "A2_FEC": {"edge": COL["FEC"], "fill": lighten(COL["FEC"], 0.60)},
    "A3":     {"edge": COL["DPE"], "fill": lighten(COL["DPE"], 0.55)},
    "NEUT":   {"edge": COL["neutral_edge"], "fill": COL["neutral_fill"]},
}


# =====================
# Timeline definition
# =====================
LANES = [
    ("Aim 1  (THF)  Failure modes", [
        ("Perf",   0, 4,  "A1",    None),
        ("Damage", 2, 6,  "A1",    "//"),
        ("Solv",   3, 8,  "A1",    "."),
    ]),
    ("Aim 2  (VC/FEC)  Carbonate packages", [
        ("RT electrochem",   0, 6,  "A2_VC",  "////"),
        ("Solv+Interphase",  4, 11, "A2_FEC", "."),
        ("Low-T retest",     10, 12, "A2_VC", None),
    ]),
    ("Aim 3  (DPE/DTD)  Beyond carbonate strategies", [
        ("Cofactor",          4, 8,  "A3", None),
        ("Sulfites/DTD",      6, 12, "A3", "."),
        ("−51 °C validation", 12, 18, "A3", None),
    ]),
    ("Outputs", [
        ("Paper 1", 9, 14, "NEUT", None),
        ("Paper 2", 12, 18, "NEUT", None),
    ]),
]

MILESTONES = [
    (4,  "A1",    "Aim 1: Baseline performance map complete"),
    (6,  "A1",    "Aim 1: Graphite damage diagnosed"),
    (8,  "A1",    "Aim 1: Solvation baseline complete"),

    (6,  "A2_VC", "Aim 2: RT electrochem screen complete"),
    (11, "A2_FEC","Aim 2: Solvation/interphase quantified"),
    (12, "A2_VC", "Aim 2: Best carbonate package selected"),

    (8,  "A3",    "Aim 3: Cofactor screen complete"),
    (12, "A3",    "Aim 3: Non-carbonate mechanism confirmed"),
    (18, "A3",    "Aim 3: Final −51 °C validation complete"),
]


# Right-side key (what you called “the key”)
CODES_KEY = [
    ("Perf",   "Performance"),
    ("Damage", "Graphite damage"),
    ("Solv",   "Solvation / interphase"),
]
HATCH_KEY = [
    ("////", "Already underway"),
    (".",    "New / untested"),
]


# =====================
# Derived layout params (this is what fixes spacing at FONT_BASE=30)
# =====================
# Figure sizes: scale moderately with font
EXEC_FIG_W = 16.8 * (0.75 + 0.25 * SCALE)
EXEC_FIG_H = 7.6  * (0.70 + 0.30 * SCALE)

AIM_FIG_W  = 15.5 * (0.75 + 0.25 * SCALE)
AIM_FIG_H  = 3.2  * (0.75 + 0.25 * SCALE)

KEY_FIG_W  = 12.8 * (0.75 + 0.25 * SCALE)
KEY_FIG_H  = 7.4  * (0.70 + 0.30 * SCALE)

# Bar geometry in data coords: make bars a bit taller when font is huge
LANE_BAND_H = 0.85 * (0.85 + 0.15 * SCALE)
MINI_H      = 0.22 * (0.85 + 0.15 * SCALE)
MINI_GAP    = 0.06 * (0.85 + 0.15 * SCALE)

# Fonts (don’t let everything go full 30+ everywhere)
FS_LANE   = fsz(1.00)
FS_BAR    = fsz(0.70)
FS_Q      = fsz(0.85)
FS_TICK   = fsz(0.80)
FS_AXIS   = fsz(0.90)
FS_MNUM   = fsz(0.55)
FS_LEG_T  = fsz(0.95)
FS_LEG    = fsz(0.72)
FS_KEY_T  = fsz(1.05)

# Wrapping for milestone key
WRAP_W = int(clamp(68 * 15 / FONT_BASE, 34, 68))


# =====================
# Utilities
# =====================
def set_rc():
    plt.rcParams.update({
        "font.family": ["Calibri", "Arial", "DejaVu Sans"],
        "font.size": FONT_BASE,
    })


def save_all_formats(fig, basepath: Path, dpi: int = 300):
    save_kwargs = dict(bbox_inches="tight", pad_inches=0.30)  # <- KEY FIX

    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(str(basepath) + "_editable.svg", **save_kwargs)

    plt.rcParams["svg.fonttype"] = "path"
    fig.savefig(str(basepath) + "_path.svg", **save_kwargs)

    fig.savefig(str(basepath) + ".png", dpi=dpi, **save_kwargs)



def setup_axis(ax):
    ax.set_xlim(0, 18)
    ax.set_yticks([])
    ax.set_xlabel("Month", fontsize=FS_AXIS)

    # quarter grid
    for x in [0, 3, 6, 9, 12, 15, 18]:
        ax.axvline(x, linewidth=1.2, alpha=0.35, color=COL["grid"], zorder=0)

    ax.set_xticks([0, 3, 6, 9, 12, 15, 18])
    ax.tick_params(axis="x", labelsize=FS_TICK)

    # Q labels at top
    quarter_centers = [1.5, 4.5, 7.5, 10.5, 13.5, 16.5]
    for q, xc in enumerate(quarter_centers, start=1):
        ax.text(
            xc / 18.0, 1.02, f"Q{q}",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=FS_Q, weight="bold", color=COL["text"]
        )


import textwrap

def draw_lane(ax, lane_label, minis, y_center):
    # Wrap long aim labels so they don't run off the canvas
    wrap_w = int(max(18, 28 * 15 / FONT_BASE))
    lane_label = "\n".join(textwrap.wrap(lane_label, width=wrap_w))

    # x is in axes coords, y is in data coords -> stable + no clipping
    ax.text(
        -0.02, y_center, lane_label,
        transform=ax.get_yaxis_transform(),
        ha="right", va="center",
        fontsize=FS_LANE, color=COL["text"],
        clip_on=False, linespacing=1.05, multialignment="right",
    )

    total_h = len(minis) * MINI_H + (len(minis) - 1) * MINI_GAP
    y0 = y_center - total_h / 2

    for j, (code, start, end, sk, hatch) in enumerate(minis):
        st = STYLE[sk]
        y = y0 + j * (MINI_H + MINI_GAP)

        ax.broken_barh(
            [(start, end - start)], (y, MINI_H),
            facecolors=st["fill"],
            edgecolors=st["edge"],
            linewidth=2.0,
            hatch=hatch,
            alpha=0.98,
            zorder=3
        )

        if (end - start) >= 2.0:
            ax.text(start + 0.2, y + MINI_H / 2, code,
                    ha="left", va="center",
                    fontsize=FS_BAR, weight="bold", color=COL["text"],
                    zorder=4)


def lane_for_style(sk: str) -> int:
    if sk.startswith("A2"):
        return 1
    if sk == "A1":
        return 0
    if sk == "A3":
        return 2
    return 1


def draw_milestones(ax):
    lane_y = {0: (len(LANES) - 1 - 0),
              1: (len(LANES) - 1 - 1),
              2: (len(LANES) - 1 - 2)}

    for idx, (m, sk, _txt) in enumerate(MILESTONES, start=1):
        lane_idx = lane_for_style(sk)
        y = lane_y[lane_idx] + LANE_BAND_H / 2 + 0.12  # slightly higher for big fonts
        st = STYLE[sk]

        ax.scatter([m], [y], marker="D",
                   s=diamond_s(170),
                   color=st["edge"], edgecolor="white", linewidth=2.2,
                   zorder=10, clip_on=False)

        ax.text(m, y, str(idx),
                ha="center", va="center",
                fontsize=FS_MNUM, weight="bold", color="white",
                zorder=11, clip_on=False)


def draw_right_legend(ax_leg):
    ax_leg.axis("off")

    # --- spacing that scales with FONT_BASE ---
    # With FONT_BASE=30, SCALE≈2, so all steps get ~2x.
    s = SCALE
    y = 0.96

    # helper: safer vertical step
    def step(dy):
        nonlocal y
        y -= dy * s
        return y

    # geometry scaling
    rect_w = 0.10
    rect_h = 0.052 * (0.85 + 0.15 * s)   # slightly taller for big fonts
    rect_yoff = 0.035 * (0.85 + 0.15 * s)
    text_yoff = 0.01  * (0.85 + 0.15 * s)

    # ----- Header -----
    ax_leg.text(
        0.02, y, "Legend",
        transform=ax_leg.transAxes,
        ha="left", va="top",
        fontsize=FS_LEG_T, weight="bold", color=COL["text"]
    )
    step(0.10)

    # ----- Aim colors -----
    ax_leg.text(
        0.02, y, "Aim color schemes",
        transform=ax_leg.transAxes,
        ha="left", va="top",
        fontsize=FS_LEG, weight="bold", color=COL["text"]
    )
    step(0.08)

    aim_items = [("Aim 1 (THF)", "A1"), ("Aim 2 (VC/FEC)", "A2_VC"), ("Aim 3 (DPE)", "A3")]
    for label, sk in aim_items:
        st = STYLE[sk]
        rect = patches.Rectangle(
            (0.02, y - rect_yoff), rect_w, rect_h,
            transform=ax_leg.transAxes,
            facecolor=st["fill"], edgecolor=st["edge"], linewidth=2
        )
        ax_leg.add_patch(rect)

        ax_leg.text(
            0.15, y - text_yoff, label,
            transform=ax_leg.transAxes,
            ha="left", va="center",
            fontsize=FS_LEG, color=COL["text"]
        )
        step(0.085)  # was 0.075

    step(0.03)

    # ----- Codes -----
    ax_leg.text(
        0.02, y, "Codes",
        transform=ax_leg.transAxes,
        ha="left", va="top",
        fontsize=FS_LEG, weight="bold", color=COL["text"]
    )
    step(0.07)

    for short, meaning in CODES_KEY:
        ax_leg.text(
            0.02, y, f"{short}:",
            transform=ax_leg.transAxes,
            ha="left", va="center",
            fontsize=FS_LEG, weight="bold", color=COL["text"]
        )
        ax_leg.text(
            0.22, y, meaning,
            transform=ax_leg.transAxes,
            ha="left", va="center",
            fontsize=FS_LEG, color=COL["text"]
        )
        step(0.07)  # was 0.06

    step(0.03)

    # ----- Hatches -----
    ax_leg.text(
        0.02, y, "Hatches",
        transform=ax_leg.transAxes,
        ha="left", va="top",
        fontsize=FS_LEG, weight="bold", color=COL["text"]
    )
    step(0.08)

    for hatch, meaning in HATCH_KEY:
        rect = patches.Rectangle(
            (0.02, y - rect_yoff), rect_w, rect_h,
            transform=ax_leg.transAxes,
            facecolor="white", edgecolor=COL["grid"],
            linewidth=1.5, hatch=hatch
        )
        ax_leg.add_patch(rect)

        ax_leg.text(
            0.15, y - text_yoff, meaning,
            transform=ax_leg.transAxes,
            ha="left", va="center",
            fontsize=FS_LEG, color=COL["text"]
        )
        step(0.085)  # was 0.075



# =====================
# Plot types
# =====================
def plot_executive(out_dir: Path):
    set_rc()

    fig, (ax, ax_leg) = plt.subplots(
        ncols=2,
        figsize=(EXEC_FIG_W, EXEC_FIG_H),
        gridspec_kw={"width_ratios": [5.7, 2.2]}
    )

    setup_axis(ax)

    # headroom for diamonds (font 30 needs it)
    ax.set_ylim(-0.7, len(LANES) - 0.05)

    for i, (label, minis) in enumerate(LANES):
        y = len(LANES) - 1 - i
        draw_lane(ax, label, minis, y)

    draw_milestones(ax)
    draw_right_legend(ax_leg)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.18, wspace=0.03)
    save_all_formats(fig, out_dir / "gantt_executive")
    plt.close(fig)


def plot_single_aim(out_dir: Path, lane_index: int, name: str):
    set_rc()

    fig, (ax, ax_leg) = plt.subplots(
        ncols=2,
        figsize=(AIM_FIG_W, AIM_FIG_H),
        gridspec_kw={"width_ratios": [5.7, 2.2]}
    )

    setup_axis(ax)
    ax.set_ylim(-0.85, 1.10)  # headroom for diamonds

    label, minis = LANES[lane_index]
    draw_lane(ax, label, minis, 0.0)

    # milestones only for this aim
    for idx, (m, sk, _txt) in enumerate(MILESTONES, start=1):
        if lane_index == 0 and sk != "A1":
            continue
        if lane_index == 1 and not sk.startswith("A2"):
            continue
        if lane_index == 2 and sk != "A3":
            continue

        y = 0.0 + LANE_BAND_H / 2 + 0.12
        st = STYLE[sk]
        ax.scatter([m], [y], marker="D",
                   s=diamond_s(170),
                   color=st["edge"], edgecolor="white", linewidth=2.2,
                   zorder=10, clip_on=False)
        ax.text(m, y, str(idx),
                ha="center", va="center",
                fontsize=FS_MNUM, weight="bold", color="white",
                zorder=11, clip_on=False)

    draw_right_legend(ax_leg)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.28, wspace=0.03)
    save_all_formats(fig, out_dir / name)
    plt.close(fig)


def plot_milestone_key(out_dir: Path):
    set_rc()

    fig, ax = plt.subplots(figsize=(KEY_FIG_W, KEY_FIG_H))
    ax.axis("off")

    ax.text(0.02, 0.96, "Milestone Key (numbers match diamonds)",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=FS_KEY_T, weight="bold", color=COL["text"])

    y = 0.88
    for idx, (_m, sk, txt) in enumerate(MILESTONES, start=1):
        st = STYLE[sk]
        ax.scatter([0.03], [y], transform=ax.transAxes, marker="D",
                   s=diamond_s(180), color=st["edge"], edgecolor="white", linewidth=2.2)

        wrapped = "\n".join(textwrap.wrap(f"{idx}. {txt}", width=WRAP_W))
        ax.text(0.07, y, wrapped, transform=ax.transAxes,
                ha="left", va="center",
                fontsize=FS_LEG, color=COL["text"])

        n_lines = wrapped.count("\n") + 1
        y -= 0.080 + 0.050 * (n_lines - 1)

    fig.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.05)
    save_all_formats(fig, out_dir / "gantt_milestone_key")
    plt.close(fig)


# =====================
# Main
# =====================
def main():
    out_dir = Path("Gantt")
    out_dir.mkdir(exist_ok=True)

    plot_executive(out_dir)
    plot_milestone_key(out_dir)
    plot_single_aim(out_dir, 0, "gantt_aim1")
    plot_single_aim(out_dir, 1, "gantt_aim2")
    plot_single_aim(out_dir, 2, "gantt_aim3")

    print(f"Saved all charts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
