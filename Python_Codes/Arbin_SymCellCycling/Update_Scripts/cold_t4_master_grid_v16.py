



def main_7():
    """
    Update of main_6 per latest tweaks:
      - Bottom row: keep the SAME blue gradient scheme as main_5 (darker with higher additive level)
      - Top row y-axis *tick label values* (SWITCHED):
          * Y1 (left axis, Cap)  tick labels ONLY on LEFTMOST panel
          * Y2 (right axis, Vmid) tick labels ONLY on RIGHTMOST panel
        …and ylabel TEXT the same way:
          * Cap ylabel only LEFTMOST (left axis)
          * Vmid ylabel only RIGHTMOST (right axis)
      - Top row points + errorbars:
          * Cap = Blues (by additive level)
          * Vmid = Oranges (by additive level)
      - Std-dev bars recolored to match
      - Discharge curves solid; legends inside each bottom subplot; legend text only additive wt% (no cell codes)
      - No fit lines; print fit params + R^2 to console
      - Drop figure suptitle
    """
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.container import ErrorbarContainer

    # ---------------- styling knobs ----------------
    FONT_TITLE = 18
    FONT_LABEL = 16
    FONT_TICK = 13
    FONT_LEGEND = 12

    Y1_COLOR = "tab:blue"    # capacity axis
    Y2_COLOR = "tab:orange"  # voltage axis

    # ----------- NEW: subplot letter + contrast knobs -----------
    SUBPLOT_LETTER_FONT = FONT_LABEL
    SUBPLOT_LETTER_X = 0.02
    SUBPLOT_LETTER_Y = 0.98

    # Bottom (V vs Q) curve contrast: spread across more of the colormap
    BOTTOM_BLUE_LO = 0.12
    BOTTOM_BLUE_HI = 0.98
    CURVE_LW = 2.4

    import os

    # ----------- NEW: output + sizing knobs -----------
    SAVE_DIR = r"C:\Users\benja\Downloads\Dilute THF Data\11_25_25\-51C_Repeats\plots_t19\_exports"
    os.makedirs(SAVE_DIR, exist_ok=True)

    SAVE_DPI = 150  # used for screen->inch conversion and for file export
    USE_SCREEN_SIZE = True  # True = use monitor size; False = use FIGSIZE_OVERRIDE
    SCREEN_FRAC = 0.95  # 0.95 fills most of the screen (avoid taskbar)
    FIGSIZE_OVERRIDE = None  # e.g. (18, 10) in inches, if you want manual control

    rc = {
        "figure.titlesize": FONT_TITLE,
        "axes.titlesize": FONT_TITLE - 2,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
    }
    # --- enforce replicate marker mapping for top plots (CellNumber == replicate) ---
    # Replicate 1 = triangle, Replicate 2 = square, Replicate 3 = diamond
    # Enforce replicate marker mapping (make sure all are unique)
    # Rep 1 triangle, Rep 2 square, Rep 3 diamond, Rep 4 circle (or pick another)
    CELLNUM_MARKERS.update({1: "^", 2: "s", 3: "D", 4: "o"})

    # ---------------- helpers ----------------
    def _get_new_fig_id(before_ids):
        after = set(plt.get_fignums())
        new = sorted(list(after - set(before_ids)))
        return new[-1] if new else None

    def _get_screen_figsize_inches(dpi=100, frac=0.95, fallback=(18, 10)):
        """
        Returns (w_in, h_in) sized to your primary display.
        Uses tkinter to query screen pixel dims, then converts to inches using dpi.
        """
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            w_px = root.winfo_screenwidth()
            h_px = root.winfo_screenheight()
            root.destroy()
            return (max(6.0, (w_px * frac) / float(dpi)),
                    max(4.0, (h_px * frac) / float(dpi)))
        except Exception:
            return fallback

    def add_top_left_legends(fig, dfm_trial, rep_loc="lower center", rep_bbox=(0.5, 0.10)):
        """
        Single-call legend builder for the TOP-LEFT top-row panel:
          - Replicate legend (top of the stack) using your existing placement style
          - Metric legend (filled vs hollow) directly BELOW replicate legend
        """
        from matplotlib.lines import Line2D

        groups = _group_axes_by_position(fig)
        top_keys = _row_keys(groups, "top")
        if not top_keys:
            return

        # top-left panel = first key (row_keys sorts left->right)
        key = top_keys[0]
        ax_pair = groups.get(key, [])
        if len(ax_pair) != 2:
            return

        axL, axR = _pick_left_right_axes(ax_pair)
        if axL is None or axR is None:
            return

        # ---------- replicate legend handles ----------
        d = dfm_trial.copy()
        if "CellNumber" not in d.columns:
            d["CellNumber"] = d["CellCode"].apply(get_cell_number)

        present = sorted(set(d["CellNumber"].dropna().astype(int).tolist()))
        rep_marker_map = {1: "^", 2: "s", 3: "D", 4: "o"}  # keep in sync with CELLNUM_MARKERS.update(...)
        present = [r for r in present if r in rep_marker_map]

        rep_handles = [
            Line2D([0], [0],
                   marker=rep_marker_map[r], linestyle="None",
                   color="black", markersize=8, label=f"Rep {r}")
            for r in present
        ]

        # ---------- metric legend handles (filled vs hollow) ----------
        cap_lab = (axL.get_ylabel() or "Cap @2.5V").strip()
        v_lab = (axR.get_ylabel() or "V@50%Q").strip()

        metric_handles = [
            Line2D([0], [0],
                   marker="o", linestyle="None",
                   markerfacecolor="black", markeredgecolor="black",
                   markersize=7, label=cap_lab),
            Line2D([0], [0],
                   marker="o", linestyle="None",
                   markerfacecolor="none", markeredgecolor="black",
                   markersize=7, label=v_lab),
        ]

        # Remove any existing legends on this axis (avoid duplicates)
        try:
            old = axL.get_legend()
            if old is not None:
                old.remove()
        except Exception:
            pass

        # ---------- 1) Replicate legend (top of stack) ----------
        # Keep YOUR replicate placement (same style), but a bit higher so room exists below.
        # You can tweak rep_bbox y if needed.
        rep_leg = None
        if rep_handles:
            rep_leg = axL.legend(
                handles=rep_handles,
                loc=rep_loc,
                bbox_to_anchor=rep_bbox,  # <-- replicate placement anchor
                ncol=2,  # two rows for 3–4 entries
                frameon=False,
                title="Replicate",
                borderaxespad=0.0,
                handletextpad=0.6,
                columnspacing=1.0,
            )

        # ---------- 2) Metric legend (below replicate) ----------
        # Place it just below replicate. Use the same anchor x, smaller y.
        # If you change rep_bbox, metric y should stay a bit smaller.
        metric_bbox = (rep_bbox[0], rep_bbox[1] - 0.075)  # <-- stack below
        metric_leg = axL.legend(
            handles=metric_handles,
            loc=rep_loc,
            bbox_to_anchor=metric_bbox,
            ncol=1,
            frameon=False,
            title=None,
            borderaxespad=0.0,
            handletextpad=0.6,
            columnspacing=0.8,
        )

        # Add replicate legend back on top (so both show)
        if rep_leg is not None:
            axL.add_artist(rep_leg)

    def add_metric_fill_legend_to_top_left(fig):
        """
        Add a small legend INSIDE the TOP-LEFT top-row panel that explains:
          - filled marker = Y1 metric (uses left-axis ylabel text)
          - hollow marker = Y2 metric (uses right-axis ylabel text)

        This keeps any existing legend(s) on that axis by re-adding them.
        """
        from matplotlib.lines import Line2D

        groups = _group_axes_by_position(fig)
        top_keys = _row_keys(groups, "top")
        if not top_keys:
            return

        # top-left panel = first key (row_keys sorts left->right)
        key = top_keys[0]
        ax_pair = groups.get(key, [])
        if len(ax_pair) != 2:
            return

        axL, axR = _pick_left_right_axes(ax_pair)
        if axL is None or axR is None:
            return

        # Use the *actual* axis label text (fallbacks if empty)
        cap_lab = (axL.get_ylabel() or "Cap @2.5V").strip()
        v_lab = (axR.get_ylabel() or "V@50%Q").strip()

        handles = [
            Line2D([0], [0],
                   marker="o", linestyle="None",
                   markerfacecolor="black", markeredgecolor="black",
                   markersize=7, label=cap_lab),
            Line2D([0], [0],
                   marker="o", linestyle="None",
                   markerfacecolor="none", markeredgecolor="black",
                   markersize=7, label=v_lab),
        ]

        # Preserve any existing legend on axL
        leg_existing = axL.get_legend()

        # Slightly left-shifted placement
        metric_leg = axL.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.01, 0.02),  # <-- move left (more negative = more left)
            ncol=1,
            frameon=False,
            borderaxespad=0.0,
            handletextpad=0.6,
            columnspacing=0.8,
        )

        if leg_existing is not None:
            axL.add_artist(leg_existing)

    def _add_subplot_letters(fig, start_letter="a"):
        """
        Adds a), b), c)... to each PANEL:
          - top row panels: label only the LEFT axis of each twin-axes pair
          - bottom row panels: label the single axis
        Order: top row L->R then bottom row L->R
        """
        import string
        letters = string.ascii_lowercase

        groups = _group_axes_by_position(fig)

        panel_axes = []

        # top row panels (use left axis of twin pair)
        for key in _row_keys(groups, "top"):
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, _axR = _pick_left_right_axes(ax_pair)
            if axL is not None:
                panel_axes.append(axL)

        # bottom row panels (single axis)
        for key in _row_keys(groups, "bottom"):
            ax_list = groups.get(key, [])
            if len(ax_list) == 1:
                panel_axes.append(ax_list[0])

        if not panel_axes:
            return

        start_idx = letters.index(start_letter)
        for i, ax in enumerate(panel_axes):
            if start_idx + i >= len(letters):
                break  # unlikely you exceed 26 panels here
            lab = f"{letters[start_idx + i]})"
            ax.text(
                SUBPLOT_LETTER_X, SUBPLOT_LETTER_Y, lab,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=SUBPLOT_LETTER_FONT,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.0),
                clip_on=False,
            )

    def _group_axes_by_position(fig):
        groups = {}
        for ax in fig.get_axes():
            b = ax.get_position().bounds  # (x0, y0, w, h)
            key = tuple(round(v, 4) for v in b)
            groups.setdefault(key, []).append(ax)
        return groups

    def _bold_top_panel_titles(fig):
        """Bold the top-row panel titles like 'VC 0 wt%' or 'FEC 1 wt%'."""
        groups = _group_axes_by_position(fig)
        for key in _row_keys(groups, "top"):
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, _axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue
            t = axL.get_title() or ""
            if t.strip():
                axL.set_title(t, fontweight="bold")

    def _add_replicate_symbol_legend(fig, dfm_trial):
        from matplotlib.lines import Line2D

        # ensure CellNumber exists
        if "CellNumber" not in dfm_trial.columns:
            dfm_trial = dfm_trial.copy()
            dfm_trial["CellNumber"] = dfm_trial["CellCode"].apply(get_cell_number)

        present_reps = sorted(set(dfm_trial["CellNumber"].dropna().astype(int).tolist()))

        rep_marker_map = {1: "^", 2: "s", 3: "D", 4: "o"}  # keep in sync with update above
        handles = []
        for r in present_reps:
            if r in rep_marker_map:
                handles.append(
                    Line2D([0], [0], marker=rep_marker_map[r], linestyle="None",
                           color="black", markersize=8, label=f"Replicate {r}")
                )

        if not handles:
            return

        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.75, 0.995),
            ncol=min(4, len(handles)),
            frameon=False,
            title="Replicate (marker)",
            fontsize=FONT_LEGEND,
            title_fontsize=FONT_LEGEND,
        )

    def set_all_ticks_inward(fig):
        for ax in fig.get_axes():
            ax.tick_params(axis="both", which="both",
                           direction="in",
                           top=True, right=True)

    def _row_keys(groups, which="top"):
        if not groups:
            return []
        y0s = [k[1] for k in groups.keys()]
        target = max(y0s) if which == "top" else min(y0s)
        keys = [k for k in groups.keys() if abs(k[1] - target) < 0.02]
        return sorted(keys, key=lambda k: k[0])  # left->right

    def _pick_left_right_axes(ax_list):
        if len(ax_list) != 2:
            return None, None
        a0, a1 = ax_list
        if str(a0.yaxis.get_ticks_position()).lower() == "right":
            return a1, a0
        if str(a1.yaxis.get_ticks_position()).lower() == "right":
            return a0, a1
        return a0, a1

    def _nearest_level(x, levels):
        levels = np.asarray(levels, dtype=float)
        return float(levels[np.argmin(np.abs(levels - float(x)))])

    def _palette(levels, cmap_name, lo=0.35, hi=0.90):
        lv = list(sorted([float(v) for v in levels]))
        cmap = plt.get_cmap(cmap_name)
        if len(lv) == 1:
            return {lv[0]: cmap(hi)}
        vals = np.linspace(lo, hi, len(lv))
        return {lv[i]: cmap(vals[i]) for i in range(len(lv))}

    def _remove_global_fig_legends(fig):
        for leg in list(getattr(fig, "legends", [])):
            try:
                leg.remove()
            except Exception:
                pass

    def _drop_suptitle(fig):
        try:
            if getattr(fig, "_suptitle", None) is not None:
                fig._suptitle.set_text("")
                fig._suptitle.set_visible(False)
        except Exception:
            pass

    def add_replicate_marker_legend_to_top_left(fig, dfm_trial):
        """
        Put replicate marker legend onto the TOP-LEFT top-row subplot (the (1,1) panel),
        in TWO ROWS at the BOTTOM of that subplot.
        """
        from matplotlib.lines import Line2D

        groups = _group_axes_by_position(fig)
        top_keys = _row_keys(groups, "top")
        if not top_keys:
            return

        # top-left panel = first key (row_keys sorts left->right)
        key = top_keys[0]
        ax_pair = groups.get(key, [])
        if len(ax_pair) != 2:
            return
        axL, _axR = _pick_left_right_axes(ax_pair)
        if axL is None:
            return

        d = dfm_trial.copy()
        if "CellNumber" not in d.columns:
            d["CellNumber"] = d["CellCode"].apply(get_cell_number)

        present = sorted(set(d["CellNumber"].dropna().astype(int).tolist()))
        rep_marker_map = {1: "^", 2: "s", 3: "D", 4: "o"}  # keep in sync with CELLNUM_MARKERS.update(...)
        present = [r for r in present if r in rep_marker_map]
        if not present:
            return

        handles = [
            Line2D([0], [0], marker=rep_marker_map[r], linestyle="None",
                   color="black", markersize=8, label=f"Rep {r}")
            for r in present
        ]

        # Keep any existing legend
        leg1 = axL.get_legend()

        # Two rows: ncol=2 gives 2 rows for 3–4 entries
        rep_leg = axL.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=2,
            frameon=False,
            title="Replicate",
            borderaxespad=0.0,
            handletextpad=0.6,
            columnspacing=1.0,
        )

        if leg1 is not None:
            axL.add_artist(leg1)

    def _legend_inside(ax):
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(loc="best", frameon=False, fontsize=FONT_LEGEND)

    def _bump_tick_fonts(fig):
        for ax in fig.get_axes():
            ax.tick_params(axis="both", which="both", labelsize=FONT_TICK)

    def _style_y_axes(axL, axR):
        # left axis styling (capacity)
        axL.spines["left"].set_color(Y1_COLOR)
        axL.tick_params(axis="y", colors=Y1_COLOR)
        axL.yaxis.label.set_color(Y1_COLOR)
        # right axis styling (voltage)
        axR.spines["right"].set_color(Y2_COLOR)
        axR.tick_params(axis="y", colors=Y2_COLOR)
        axR.yaxis.label.set_color(Y2_COLOR)

    def _set_toprow_ticks_and_ylabels_switched(fig):
        """
        Top row only (SWITCHED vs main_6):
          - Show Y1 (left-axis) tick labels + ylabel ONLY on LEFTMOST panel
          - Show Y2 (right-axis) tick labels + ylabel ONLY on RIGHTMOST panel
        """
        groups = _group_axes_by_position(fig)
        top_keys = _row_keys(groups, "top")
        if not top_keys:
            return

        leftmost = top_keys[0]
        rightmost = top_keys[-1]

        cap_ylabel = None
        vmid_ylabel = None
        for key in top_keys:
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue
            cap_ylabel = cap_ylabel or (axL.get_ylabel() or "")
            vmid_ylabel = vmid_ylabel or (axR.get_ylabel() or "")
        cap_ylabel = cap_ylabel or "Capacity"
        vmid_ylabel = vmid_ylabel or "Voltage"

        for key in top_keys:
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue

            axL.tick_params(labelleft=(key == leftmost))    # Y1 ticks only leftmost
            axR.tick_params(labelright=(key == rightmost))  # Y2 ticks only rightmost

            axL.set_ylabel(cap_ylabel if key == leftmost else "")
            axR.set_ylabel(vmid_ylabel if key == rightmost else "")

            _style_y_axes(axL, axR)

    def _fit_stats(x, y, use_level_means=True):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 2:
            return np.nan, np.nan, np.nan

        if use_level_means:
            ux = np.unique(x)
            if ux.size < 2:
                return np.nan, np.nan, np.nan
            xs, ys = [], []
            for u in ux:
                yy = y[x == u]
                yy = yy[np.isfinite(yy)]
                if yy.size:
                    xs.append(u)
                    ys.append(float(np.mean(yy)))
            x = np.asarray(xs, dtype=float)
            y = np.asarray(ys, dtype=float)
            if x.size < 2:
                return np.nan, np.nan, np.nan

        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = np.nan if ss_tot <= 0 else (1.0 - ss_res / ss_tot)
        return float(slope), float(intercept), float(r2)

    def _print_bestfits_vc_shift(fig, df_use, cap_col, v_col):
        groups = _group_axes_by_position(fig)
        for key in _row_keys(groups, "top"):
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, _axR = _pick_left_right_axes(ax_pair)
            title = axL.get_title() or ""
            m = re.search(r"FEC\s*([0-9.]+)", title)
            if not m:
                continue
            fec = float(m.group(1))

            sub = df_use.copy()
            sub["FEC_wt"] = pd.to_numeric(sub["FEC_wt"], errors="coerce")
            sub["VC_wt"] = pd.to_numeric(sub["VC_wt"], errors="coerce")
            sub = sub[sub["FEC_wt"] == fec]

            s1, b1, r1 = _fit_stats(sub["VC_wt"].to_numpy(float), sub[cap_col].to_numpy(float), use_level_means=True)
            s2, b2, r2 = _fit_stats(sub["VC_wt"].to_numpy(float), sub[v_col].to_numpy(float), use_level_means=True)

            print(f"[VC-shift | Trial 3] FEC={fec:g} wt%")
            print(f"  Cap  (mean-per-VC):   slope={s1:.6g}, intercept={b1:.6g}, R^2={r1:.4f}")
            print(f"  Vmid (mean-per-VC):   slope={s2:.6g}, intercept={b2:.6g}, R^2={r2:.4f}")

    def _print_bestfits_fec_shift(fig, df_use, cap_col, v_col):
        groups = _group_axes_by_position(fig)
        for key in _row_keys(groups, "top"):
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, _axR = _pick_left_right_axes(ax_pair)
            title = axL.get_title() or ""
            m = re.search(r"VC\s*([0-9.]+)", title)
            if not m:
                continue
            vc = float(m.group(1))

            sub = df_use.copy()
            sub["FEC_wt"] = pd.to_numeric(sub["FEC_wt"], errors="coerce")
            sub["VC_wt"] = pd.to_numeric(sub["VC_wt"], errors="coerce")
            sub = sub[sub["VC_wt"] == vc]

            s1, b1, r1 = _fit_stats(sub["FEC_wt"].to_numpy(float), sub[cap_col].to_numpy(float), use_level_means=True)
            s2, b2, r2 = _fit_stats(sub["FEC_wt"].to_numpy(float), sub[v_col].to_numpy(float), use_level_means=True)

            print(f"[FEC-shift | Trial 3] VC={vc:g} wt%")
            print(f"  Cap  (mean-per-FEC):  slope={s1:.6g}, intercept={b1:.6g}, R^2={r1:.4f}")
            print(f"  Vmid (mean-per-FEC):  slope={s2:.6g}, intercept={b2:.6g}, R^2={r2:.4f}")

    def _recolor_toprow_points_and_errorbars(fig, levels):
        """
        Top row:
          - Capacity points + errorbars: BLUE shades (by x level)
          - Voltage points + errorbars: ORANGE shades (by x level)
        """
        cap_map = _level_color_map(levels)  # categorical per wt%

        v_map = _palette(levels, "Oranges", lo=0.35, hi=0.90)

        groups = _group_axes_by_position(fig)
        top_keys = _row_keys(groups, "top")

        for key in top_keys:
            ax_pair = groups.get(key, [])
            if len(ax_pair) != 2:
                continue
            axL, axR = _pick_left_right_axes(ax_pair)
            if axL is None:
                continue

            # raw scatter collections
            for coll in list(axL.collections):  # cap (filled blue)
                try:
                    off = coll.get_offsets()
                    if off is None or len(off) == 0:
                        continue
                    x = float(off[0, 0])
                    lvl = _nearest_level(x, levels)
                    c = cap_map.get(lvl, Y1_COLOR)
                    coll.set_facecolor(c)
                    coll.set_edgecolor(c)
                except Exception:
                    pass

            for coll in list(axR.collections):  # v (hollow orange edge)
                try:
                    off = coll.get_offsets()
                    if off is None or len(off) == 0:
                        continue
                    x = float(off[0, 0])
                    lvl = _nearest_level(x, levels)
                    c = v_map.get(lvl, Y2_COLOR)
                    coll.set_facecolor("none")
                    coll.set_edgecolor(c)
                except Exception:
                    pass

            # errorbar containers (means + SD bars)
            for cont in list(getattr(axL, "containers", [])):
                if isinstance(cont, ErrorbarContainer):
                    try:
                        data_line, caplines, barlinecols = cont.lines
                        xdat = data_line.get_xdata()
                        if len(xdat) == 0:
                            continue
                        lvl = _nearest_level(float(xdat[0]), levels)
                        c = cap_map.get(lvl, Y1_COLOR)

                        data_line.set_color(c)
                        data_line.set_markerfacecolor(c)
                        data_line.set_markeredgecolor(c)
                        for cl in caplines:
                            cl.set_color(c)
                        for blc in barlinecols:
                            blc.set_color(c)
                    except Exception:
                        pass

            for cont in list(getattr(axR, "containers", [])):
                if isinstance(cont, ErrorbarContainer):
                    try:
                        data_line, caplines, barlinecols = cont.lines
                        xdat = data_line.get_xdata()
                        if len(xdat) == 0:
                            continue
                        lvl = _nearest_level(float(xdat[0]), levels)
                        c = v_map.get(lvl, Y2_COLOR)

                        data_line.set_color(c)
                        data_line.set_markerfacecolor("none")
                        data_line.set_markeredgecolor(c)
                        for cl in caplines:
                            cl.set_color(c)
                        for blc in barlinecols:
                            blc.set_color(c)
                    except Exception:
                        pass
    # --- NEW: categorical colors for wt% levels (Y1 + discharge curves) ---
    LEVEL_COLORS_5 = {
        0.0:  "#1f77b4",  # blue
        1.0:  "#2ca02c",  # green
        2.0:  "#9467bd",  # purple
        5.0:  "#d62728",  # red
        10.0: "#8c564b",  # brown
    }

    def _level_color_map(levels):
        """
        Discrete high-contrast mapping for additive wt% levels.
        If a level isn't in LEVEL_COLORS_5 (unlikely), falls back to tab10 cycling.
        """
        lv = list(sorted([float(v) for v in levels]))
        tab = plt.get_cmap("tab10")
        out = {}
        for i, v in enumerate(lv):
            out[v] = LEVEL_COLORS_5.get(float(v), tab(i % 10))
        return out

    def _recolor_and_relabel_bottom_vc_blue(fig, vc_levels):
        """
        Bottom row curves for VC-shift:
          - blue gradient by VC level
          - solid lines
          - legend labels ONLY: "{vc} wt%"
        """
        blue_map = _level_color_map(vc_levels)  # categorical per wt%

        groups = _group_axes_by_position(fig)
        bot_keys = _row_keys(groups, "bottom")

        for key in bot_keys:
            ax_list = groups.get(key, [])
            if len(ax_list) != 1:
                continue
            ax = ax_list[0]
            seen = set()
            for ln in ax.get_lines():
                lab = ln.get_label() or ""
                m = re.search(r"VC\s*([0-9.]+)", lab)
                ln.set_linestyle("-")
                ln.set_linewidth(CURVE_LW)
                if m:
                    vc = float(m.group(1))
                    lvl = _nearest_level(vc, vc_levels)
                    ln.set_color(blue_map.get(lvl, Y1_COLOR))
                    new_lab = f"{int(lvl)} wt%"
                    if new_lab in seen:
                        ln.set_label("_nolegend_")
                    else:
                        ln.set_label(new_lab)
                        seen.add(new_lab)
                else:
                    ln.set_label("_nolegend_")
            _legend_inside(ax)

    def _recolor_and_relabel_bottom_fec_blue(fig, fec_levels):
        """
        Bottom row curves for FEC-shift:
          - blue gradient by FEC level
          - solid lines
          - legend labels ONLY: "{fec} wt%"
        """
        blue_map = _level_color_map(fec_levels)  # categorical per wt%

        groups = _group_axes_by_position(fig)
        bot_keys = _row_keys(groups, "bottom")

        for key in bot_keys:
            ax_list = groups.get(key, [])
            if len(ax_list) != 1:
                continue
            ax = ax_list[0]
            seen = set()
            for ln in ax.get_lines():
                lab = ln.get_label() or ""
                m = re.search(r"FEC\s*([0-9.]+)", lab)
                ln.set_linestyle("-")
                ln.set_linewidth(CURVE_LW)
                if m:
                    fec = float(m.group(1))
                    lvl = _nearest_level(fec, fec_levels)
                    ln.set_color(blue_map.get(lvl, Y1_COLOR))
                    new_lab = f"{int(lvl)} wt%"
                    if new_lab in seen:
                        ln.set_label("_nolegend_")
                    else:
                        ln.set_label(new_lab)
                        seen.add(new_lab)
                else:
                    ln.set_label("_nolegend_")
            _legend_inside(ax)

    # ---------------- build data (Trial 3 only) ----------------
    files_with_source = find_discharge_files(base_dir, old_directory, temp_tag="-51")
    if not files_with_source:
        print("No matching .xlsx files found for -51C.")
        return

    lookup = load_lookup_table(lookup_table_path)

    groups = {}
    for p, _src in files_with_source:
        code = get_cell_code(p)
        groups.setdefault(code, []).append(p)

    cell_meta = {}
    for cell_code in groups.keys():
        electrolyte = get_electrolyte_name(cell_code, lookup)
        cell_meta[cell_code] = {
            "electrolyte": electrolyte,
            "total_additive": get_total_additive(electrolyte),
        }

    # --- COPY/PASTE PATCH: replace your alpha colormap block with this ---

    unique_alphas = sorted({get_alpha_prefix(c) for c in groups.keys()})

    # High-contrast / colorblind-friendly categorical palette (hex)
    high_contrast_colors = [
        "#1f77b4",  # blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#17becf",  # cyan
        "#7f7f7f",  # gray
    ]

    alpha_colors = {
        a: high_contrast_colors[i % len(high_contrast_colors)]
        for i, a in enumerate(unique_alphas)
    }

    dfm = compute_cell_discharge_metrics(
        groups, lookup, cell_meta,
        fec_levels=GRID_FEC_LEVELS, vc_levels=GRID_VC_LEVELS,
        capacity_voltage=CAPACITY_VOLTAGE,
    )
    dfm = dfm[dfm["Trial"] == 3].copy()
    debug_marker_collisions(dfm, trial=3)
    cap_col = f"cap_at_{CAPACITY_VOLTAGE:.1f}V"
    v_col = "MidpointDischargeV_V"

    # ---------------- no-save / no-close + plotting ----------------
    orig_savefig = Figure.savefig
    orig_close = plt.close
    try:
        Figure.savefig = lambda self, *args, **kwargs: None
        plt.close = lambda *args, **kwargs: None

        with plt.rc_context(rc):
            # --- VC shift combined ---
            before = list(plt.get_fignums())
            plot_effect_of_vc_shift_cap_and_midV_with_curves(
                dfm, groups, lookup, alpha_colors, plots_dir,
                title_prefix="",
                capacity_voltage=CAPACITY_VOLTAGE,
                cap_col=cap_col,
                v_col=v_col,
                v_label="V @ 50% Q (V)",
            )
            fid = _get_new_fig_id(before)
            fig_vc = plt.figure(fid) if fid is not None else plt.gcf()

            _remove_global_fig_legends(fig_vc)
            _drop_suptitle(fig_vc)
            _bold_top_panel_titles(fig_vc)

            _recolor_toprow_points_and_errorbars(fig_vc, GRID_VC_LEVELS)
            _recolor_and_relabel_bottom_vc_blue(fig_vc, GRID_VC_LEVELS)
            _set_toprow_ticks_and_ylabels_switched(fig_vc)
            _bump_tick_fonts(fig_vc)
            _print_bestfits_vc_shift(fig_vc, dfm, cap_col, v_col)
            #add_replicate_marker_legend_to_top_left(fig_vc, dfm[dfm["Trial"] == 3],)
            #add_metric_fill_legend_to_top_left(fig_vc)
            add_top_left_legends(fig_vc, dfm[dfm["Trial"] == 3])
            #add_top_left_legends(fig_fec, dfm[dfm["Trial"] == 3])

            # --- FEC shift combined ---
            before = list(plt.get_fignums())
            plot_effect_of_fec_shift_cap_and_midV_with_curves(
                dfm, groups, lookup, alpha_colors, plots_dir,
                title_prefix="",
                capacity_voltage=CAPACITY_VOLTAGE,
                cap_col=cap_col,
                v_col=v_col,
                v_label="V @ 50% Q (V)",
            )
            fid = _get_new_fig_id(before)
            fig_fec = plt.figure(fid) if fid is not None else plt.gcf()

            _remove_global_fig_legends(fig_fec)
            _drop_suptitle(fig_fec)
            _bold_top_panel_titles(fig_fec)

            _recolor_toprow_points_and_errorbars(fig_fec, GRID_FEC_LEVELS)
            _recolor_and_relabel_bottom_fec_blue(fig_fec, GRID_FEC_LEVELS)
            _set_toprow_ticks_and_ylabels_switched(fig_fec)
            _bump_tick_fonts(fig_fec)
            _print_bestfits_fec_shift(fig_fec, dfm, cap_col, v_col)
            #add_replicate_marker_legend_to_top_left(fig_fec, dfm[dfm["Trial"] == 3])
            #add_metric_fill_legend_to_top_left(fig_fec)
            #add_top_left_legends(fig_vc, dfm[dfm["Trial"] == 3])
            add_top_left_legends(fig_fec, dfm[dfm["Trial"] == 3])
            set_all_ticks_inward(fig_vc)
            set_all_ticks_inward(fig_fec)
            _add_subplot_letters(fig_vc, start_letter="a")
            _add_subplot_letters(fig_fec, start_letter="a")
            # ----------- NEW: resize to screen (or override) and save final figs -----------
            figsize_in = FIGSIZE_OVERRIDE
            if figsize_in is None and USE_SCREEN_SIZE:
                figsize_in = _get_screen_figsize_inches(dpi=SAVE_DPI, frac=SCREEN_FRAC)

            if figsize_in is not None:
                fig_vc.set_size_inches(figsize_in[0], figsize_in[1], forward=True)
                fig_fec.set_size_inches(figsize_in[0], figsize_in[1], forward=True)

            # Use orig_savefig because Figure.savefig is monkeypatched to no-op above
            out_vc = os.path.join(SAVE_DIR, "vc_shift_cap_and_midV_trial3.png")
            out_fec = os.path.join(SAVE_DIR, "fec_shift_cap_and_midV_trial3.png")

            orig_savefig(fig_vc, out_vc, dpi=SAVE_DPI, bbox_inches="tight")
            orig_savefig(fig_fec, out_fec, dpi=SAVE_DPI, bbox_inches="tight")

            print(f"Saved:\n  {out_vc}\n  {out_fec}")

            plt.show()

    finally:
        Figure.savefig = orig_savefig
        plt.close = orig_close


if __name__ == "__main__":
    main_7()