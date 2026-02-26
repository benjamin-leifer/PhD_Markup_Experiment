# electrolyte_style.py
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ============================
# STYLE ENCODING CONSTANTS
# ============================

# DT family (your “normal” additive colors)
BASE_COLOR_DT = {
    "NONE": "#2B2B2B",  # baseline (no additives)
    "F":    "#E68600",  # FEC-only (orange)
    "V":    "#7B4FA4",  # VC-only (purple)
    "FV":   "#009E73",  # FEC+VC (green/teal)
    "WHITE": "#FFFFFF",
}

# TPT family (red-coded palette; still varies by additive package)
BASE_COLOR_TPT = {
    "NONE": "#9B111E",  # deep red
    "F":    "#D1495B",  # red-orange
    "V":    "#B5179E",  # magenta-red
    "FV":   "#7A1E3A",  # red-purple (keeps it “TPT-red” without going muddy)
    "WHITE": "#FFFFFF",
}

SYSTEM_LS = {
    "DT":  "-",    # DME:THF
    "TPT": "--",   # TFSI - DPE:THF
}

# “More wt% = darker/more saturated” blending strength (t=0 -> white, t=1 -> base hue)
WT_TO_T = {1: 0.25, 2: 0.45, 5: 0.75, 10: 1.00}
ALLOWED_WT = {1, 2, 5, 10}

# VC amount -> marker encoding (so VC is obvious without extra colors)
# marker, fillmode, mew, ms
VC_MARKER = {
    0: (None, None, None, None),
    1: ("o", "filled", 0.0, 4.8),
    2: ("s", "open",   1.2, 5.2),
}

# ============================
# COLOR UTILITIES
# ============================

def _hex_to_rgb01(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return int(h[0:2], 16)/255.0, int(h[2:4], 16)/255.0, int(h[4:6], 16)/255.0

def _rgb01_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = rgb
    r = max(0, min(255, int(round(r * 255))))
    g = max(0, min(255, int(round(g * 255))))
    b = max(0, min(255, int(round(b * 255))))
    return f"#{r:02X}{g:02X}{b:02X}"

def blend_hex(base_hex: str, mix_hex: str, t: float) -> str:
    """t=0 -> mix_hex, t=1 -> base_hex"""
    t = max(0.0, min(1.0, float(t)))
    br, bg, bb = _hex_to_rgb01(base_hex)
    mr, mg, mb = _hex_to_rgb01(mix_hex)
    r = mr * (1 - t) + br * t
    g = mg * (1 - t) + bg * t
    b = mb * (1 - t) + bb * t
    return _rgb01_to_hex((r, g, b))

# ============================
# PARSING
# ============================

@dataclass(frozen=True)
class ElectrolyteSpec:
    system: str              # "DT" or "TPT"
    ratio: str               # e.g. "14"
    additives: str           # "NONE", "F", "V", "FV"
    fec_wt: int = 0
    vc_wt: int = 0

def clean_elyte_str(s: str) -> str:
    """
    Normalize messy electrolyte strings from lookup keys/notes:
    - Uppercase
    - Drop anything in parentheses
    - Remove spaces/underscores
    - Remove dashes (so DTF14-5 -> DTF145)
    """
    s = str(s).upper().strip()
    s = re.sub(r"\(.*?\)", "", s)            # drop ( ... )
    s = s.replace("_", "").replace(" ", "")
    s = s.replace("-", "")                   # per your rule: drop dash and parse normally
    # If string contains multiple tokens, keep the first DT*/TPT* token
    m = re.search(r"(TPT[FV]{0,2}\d+|DT[FV]{0,2}\d+)", s)
    return m.group(1) if m else s

def _split_letters_digits(code: str) -> Tuple[str, str]:
    m = re.match(r"^([A-Z]+)(\d+)$", code)
    if not m:
        return code, ""
    return m.group(1), m.group(2)

def parse_electrolyte(code_raw: str) -> ElectrolyteSpec:
    code = clean_elyte_str(code_raw)

    letters, digits = _split_letters_digits(code)
    system = "TPT" if letters.startswith("TPT") else ("DT" if letters.startswith("DT") else "DT")

    # letters after system define additives (F/V/FV)
    suffix = letters[len(system):] if letters.startswith(system) else ""
    has_f = "F" in suffix
    has_v = "V" in suffix
    additives = "FV" if (has_f and has_v) else ("F" if has_f else ("V" if has_v else "NONE"))

    ratio = digits[:2] if len(digits) >= 2 else digits
    tail = digits[2:] if len(digits) > 2 else ""

    fec_wt = 0
    vc_wt = 0

    if additives == "F":
        if tail:
            fec_wt = int(tail)
    elif additives == "V":
        if tail:
            vc_wt = int(tail)
    elif additives == "FV":
        # e.g., 52 => 5% FEC + 2% VC ; 102 => 10% FEC + 2% VC
        if tail.startswith("10"):
            fec_wt = 10
            vc_wt = int(tail[2:]) if tail[2:] else 0
        elif len(tail) >= 2:
            fec_wt = int(tail[0])
            vc_wt = int(tail[1:]) if len(tail) > 1 else 0

    # sanitize to allowed sets
    if fec_wt not in (0, *ALLOWED_WT):
        # keep it, but you can clamp if you want
        pass
    if vc_wt not in (0, 1, 2):
        pass

    return ElectrolyteSpec(system=system, ratio=ratio, additives=additives, fec_wt=fec_wt, vc_wt=vc_wt)

def pretty_label(code_raw: str, *, show_details: bool = False) -> str:
    code = clean_elyte_str(code_raw)
    if not show_details:
        return code
    spec = parse_electrolyte(code)
    parts = [code]
    if spec.additives in ("F", "FV") and spec.fec_wt:
        parts.append(f"{spec.fec_wt}%FEC")
    if spec.additives in ("V", "FV") and spec.vc_wt:
        parts.append(f"{spec.vc_wt}%VC")
    return " · ".join(parts)

# ============================
# STYLE LOOKUP
# ============================

def style_for_electrolyte(
    code_raw: str,
    *,
    lw_base: float = 2.2,
    lw: float = 2.6,
    markevery: Optional[int] = None,
    ms_scale: float = 1.0,
    mew_scale: float = 1.0,
) -> Dict:
    """
    Encoding:
      - Color = additive package (DT palette) OR red-coded TPT palette
      - Shade = wt% (F/FV shade by FEC wt; V shade by VC wt)
      - Marker = VC wt% (1 filled circle; 2 open square)
      - Linestyle = DT solid, TPT dashed
    """
    spec = parse_electrolyte(code_raw)
    palette = BASE_COLOR_TPT if spec.system == "TPT" else BASE_COLOR_DT

    linestyle = SYSTEM_LS.get(spec.system, "-")

    # Color + shade
    if spec.additives == "NONE":
        color = palette["NONE"]
        linewidth = lw_base
    else:
        hue = palette[spec.additives]
        wt = spec.fec_wt if spec.additives in ("F", "FV") else spec.vc_wt
        t = WT_TO_T.get(int(wt), 1.0) if wt else WT_TO_T[1]
        color = blend_hex(hue, palette["WHITE"], t=t)
        linewidth = lw

    # VC markers only for V / FV packages
    marker = None
    mfc = None
    mec = None
    mew = None
    ms = None

    if spec.additives in ("V", "FV"):
        marker, fillmode, mew0, ms0 = VC_MARKER.get(int(spec.vc_wt), (None, None, None, None))
        if marker is not None:
            ms = (ms0 or 5.0) * ms_scale
            mew = (mew0 or 0.0) * mew_scale
            if fillmode == "filled":
                mfc = color
                mec = color
                mew = 0.0
            else:
                mfc = "none"
                mec = color

    out = dict(
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=ms,
        markerfacecolor=mfc,
        markeredgecolor=mec,
        markeredgewidth=mew,
    )
    if marker is not None and markevery is not None:
        out["markevery"] = markevery

    return {k: v for k, v in out.items() if v is not None}

# ============================
# LEGEND EXPORT
# ============================

def save_curve_legend_png(
    handles,
    labels,
    out_png: str,
    *,
    ncol: int = 2,
    fontsize: int = 11,
    dpi: int = 600,
    transparent: bool = True,
    pad_in: float = 0.02,
) -> None:
    fig = plt.figure(figsize=(6.8, 1.8))
    fig.patch.set_alpha(0.0 if transparent else 1.0)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.legend(handles, labels, ncol=ncol, frameon=False, fontsize=fontsize, loc="center")
    fig.savefig(out_png, dpi=dpi, transparent=transparent, bbox_inches="tight", pad_inches=pad_in)
    plt.close(fig)

def save_encoding_legend_png(out_png: str, *, fontsize: int = 11, dpi: int = 600) -> None:
    fig = plt.figure(figsize=(7.2, 2.4))
    ax = fig.add_subplot(111)
    ax.axis("off")

    # additive package swatches (DT palette, since it’s the “main” mapping)
    h = [
        mlines.Line2D([], [], color=BASE_COLOR_DT["NONE"], lw=3.2, linestyle="-",  label="NONE (baseline)"),
        mlines.Line2D([], [], color=BASE_COLOR_DT["F"],    lw=3.0, linestyle="-",  label="FEC package (F)"),
        mlines.Line2D([], [], color=BASE_COLOR_DT["V"],    lw=3.0, linestyle="-",  label="VC package (V)"),
        mlines.Line2D([], [], color=BASE_COLOR_DT["FV"],   lw=3.0, linestyle="-",  label="FEC+VC package (FV)"),
        mlines.Line2D([], [], color="#4A4A4A", lw=2.6, linestyle=SYSTEM_LS["DT"],  label="DT system (solid)"),
        mlines.Line2D([], [], color=BASE_COLOR_TPT["NONE"], lw=2.6, linestyle=SYSTEM_LS["TPT"], label="TPT system (dashed; red-coded)"),
        mlines.Line2D([], [], color="#4A4A4A", lw=2.6, linestyle="-", marker="o", markersize=6, markerfacecolor="#4A4A4A",
                      markeredgewidth=0.0, label="VC 1% marker"),
        mlines.Line2D([], [], color="#4A4A4A", lw=2.6, linestyle="-", marker="s", markersize=6, markerfacecolor="none",
                      markeredgewidth=1.2, label="VC 2% marker"),
    ]

    ax.legend(h, [x.get_label() for x in h], ncol=2, frameon=False, fontsize=fontsize, loc="center")
    fig.savefig(out_png, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
