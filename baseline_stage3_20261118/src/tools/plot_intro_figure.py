#!/usr/bin/env python3
"""
Intro Fig. 1: benchmark and forecasting framework overview.
Left: matched radar / satellite / rainfall observations over Sichuan (real validation tiles).
Middle: seven input combinations and the shared model. Right: forecast outputs (real tiles).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from PIL import Image

from src.tools.plot_discussion_figures import (
    DEFAULT_GEO_BOUNDS,
    _iter_polygon_rings,
    _load_sichuan_geojson,
)

NOFILL = "--nofill" in sys.argv  # outline-only variant (no filled headers / bars / arrows)


def darken(hex_color, f=0.78):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % (int(r * f), int(g * f), int(b * f))


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# soft, moderately saturated palette
C_RAD = "#5B8DB8"     # soft blue
C_SAT = "#5FA392"     # soft teal
C_RAIN = "#D4A24C"    # warm amber
C_OUT = "#CF7B6A"     # soft coral
INK = "#222222"
SUB = "#4A4F57"

# (header, light fill) per panel
LWS = 0.82 if NOFILL else 1.0   # global line-width scale

P_OBS = ("#6A93BC", "#F2F7FC")
P_IN = ("#6BA899", "#F1F8F6")
P_MODEL = ("#8E8BBA", "#F5F4FB")
P_OUT = ("#D58F7E", "#FCF4F1")

if NOFILL:
    C_RAD, C_SAT, C_RAIN, C_OUT = "#4F7CA8", "#4E9484", "#C4903A", "#C46F5F"
    _SLATE = "#62748C"
    P_OBS = P_IN = P_MODEL = P_OUT = (_SLATE, "#FFFFFF")

TIMELINE = (
    REPO_ROOT / "runs" / "next_frame_final" / "2026-10-12" / "20-17-52_stage1_next_frame_block"
    / "val_viz" / "step_00096000" / "rollout_sample3_timeline.jpg"
)
ROW = {"rad": 0, "sat": 1, "rain": 2}
HIST_LAST_COL = 3
PRED_COLS = [4, 5, 6, 7]
GT_COLS = [8, 9, 10, 11]

PANEL_Y0, PANEL_Y1, HEAD_H = 0.08, 4.78, 0.52


def load_tile(img, modality, col):
    left_label_w, cell_gap, top_header_h, row_gap = 84, 6, 26, 8
    cols, rows = 12, 3
    tile_w = (img.shape[1] - left_label_w - (cols - 1) * cell_gap) // cols
    tile_h = (img.shape[0] - top_header_h - (rows - 1) * row_gap) // rows
    y0 = top_header_h + ROW[modality] * (tile_h + row_gap)
    x0 = left_label_w + col * (tile_w + cell_gap)
    return img[y0:y0 + tile_h, x0:x0 + tile_w]


PANEL_IDX = [0]


def panel(ax, x0, x1, title, colors, fs=14):
    edge, fill = colors
    if NOFILL:
        PANEL_IDX[0] += 1
        ax.add_patch(FancyBboxPatch((x0, PANEL_Y0), x1 - x0, PANEL_Y1 - PANEL_Y0, boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor="white", edgecolor=edge, lw=1.5, zorder=0))
        ax.plot([x0 + 0.18, x1 - 0.18], [PANEL_Y1 - HEAD_H, PANEL_Y1 - HEAD_H], color=edge, lw=1.0, alpha=0.7, zorder=1)
        yh = PANEL_Y1 - 0.5 * HEAD_H - 0.01
        ax.add_patch(Circle((x0 + 0.25, yh), 0.12, facecolor="white", edgecolor=edge, lw=1.3, zorder=2))
        ax.text(x0 + 0.25, yh - 0.005, str(PANEL_IDX[0]), fontsize=10, fontweight="bold", ha="center", va="center",
                color=darken(edge, 0.7), zorder=3)
        ax.text(x0 + 0.48, yh, title, fontsize=fs - 1.6, fontweight="bold", ha="left", va="center",
                color=darken(edge, 0.7), zorder=2)
        return
    ax.add_patch(FancyBboxPatch((x0, PANEL_Y0), x1 - x0, PANEL_Y1 - PANEL_Y0, boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor=fill, edgecolor=edge, lw=1.8, zorder=0))
    ax.add_patch(Rectangle((x0 + 0.02, PANEL_Y1 - HEAD_H), x1 - x0 - 0.04, HEAD_H - 0.02, facecolor=edge,
                           edgecolor="none", zorder=1))
    ax.text(0.5 * (x0 + x1), PANEL_Y1 - 0.5 * HEAD_H - 0.01, title, fontsize=fs, fontweight="bold", ha="center",
            va="center", color="white", zorder=2)


def block_arrow(ax, x0, x1, y, color="#A9B8C8"):
    if NOFILL:
        ax.add_patch(FancyArrowPatch((x0 + 0.03, y), (x1 - 0.02, y), arrowstyle="-|>", mutation_scale=17, lw=1.8,
                                     color="#62748C", zorder=5, shrinkA=0, shrinkB=0))
        return
    ax.add_patch(FancyArrow(x0, y, x1 - x0, 0, width=0.15, head_width=0.38, head_length=0.19,
                            length_includes_head=True, facecolor=color, edgecolor="none", zorder=5))


def down_arrow(ax, x, y_top, y_bot, color="#8E8BBA"):
    if NOFILL:
        color = "#7C8CA3"
    ax.add_patch(FancyArrow(x, y_top, 0, y_bot - y_top, width=0.05, head_width=0.17, head_length=0.09,
                            length_includes_head=True, facecolor=color, edgecolor="none", zorder=5))


def card(ax, x, y, w, h, edge, fill="white", lw=1.5, ls="-", z=2, r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fill, edgecolor=edge,
                                lw=lw * LWS, linestyle=ls, zorder=z))


def thumb(ax, tile, x, y, w, h, edge=INK, lw=1.0, z=3):
    ax.imshow(tile, extent=(x, x + w, y, y + h), origin="upper", interpolation="bilinear", zorder=z)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=edge, lw=lw * LWS, zorder=z + 1))


def draw_layer(ax, tile, aff, geojson, bounds, boundary_color, edge_color):
    lon0, lon1, lat0, lat1 = bounds
    tr = aff + ax.transData
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    clip = Polygon(corners, closed=True, transform=tr)
    ax.imshow(tile, extent=(0, 1, 0, 1), origin="upper", transform=tr, interpolation="bilinear", zorder=3)
    for feature in geojson.get("features", []):
        for ring in _iter_polygon_rings(feature.get("geometry", {})):
            u = (ring[:, 0] - lon0) / (lon1 - lon0)
            v = (ring[:, 1] - lat0) / (lat1 - lat0)
            (line,) = ax.plot(u, v, color=boundary_color, lw=0.5, alpha=0.9, transform=tr, zorder=4)
            line.set_clip_path(clip)
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=edge_color, lw=2.0 * LWS, transform=tr, zorder=5))


TXT = {"rad": "#3F6890", "sat": "#3A7466", "rain": "#9A6B1E"}


def modality_tile(ax, x, y, w, h, key, present, fs=10):
    col = {"rad": C_RAD, "sat": C_SAT, "rain": C_RAIN}[key]
    if present:
        card(ax, x, y, w, h, edge=col, fill="white", lw=1.7, z=3, r=0.05)
        ax.text(x + 0.5 * w, y + 0.5 * h, key, fontsize=fs, fontweight="bold", color=TXT[key], ha="center", va="center", zorder=4)
    else:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white" if NOFILL else "#F4F4F4", edgecolor="#B3B3B3" if NOFILL else "#9A9A9A", lw=0.8, hatch="////", zorder=3))
        cx, cy = x + 0.5 * w, y + 0.5 * h
        ax.add_patch(Circle((cx, cy), 0.075, facecolor="white", edgecolor="#8A8A8A", lw=1.0, zorder=4))
        d = 0.034
        ax.plot([cx - d, cx + d], [cy - d, cy + d], color="#8A8A8A", lw=1.2, zorder=5)
        ax.plot([cx - d, cx + d], [cy + d, cy - d], color="#8A8A8A", lw=1.2, zorder=5)


def draw_model_panel(ax, P3):
    panel(ax, *P3, "Shared model", P_MODEL)
    x0, x1 = P3
    cx = 0.5 * (x0 + x1)
    cw = x1 - x0 - 0.30
    cxl = x0 + 0.15
    EDGE = "#7C8CA3" if NOFILL else "#8E8BBA"
    DARK = "#3A4658" if NOFILL else "#3F3D63"
    TAN = "#D9A15B"

    # A: missing-modality replacement
    ya0, ya1 = 3.34, 4.18
    card(ax, cxl, ya0, cw, ya1 - ya0, edge=EDGE)
    ax.text(cx, ya1 - 0.15, "Missing-modality replacement", fontsize=9.4, fontweight="bold", ha="center", va="center",
            color=DARK, zorder=4)
    tw, th, gap = 0.58, 0.30, 0.10
    sx = cx - 0.5 * (3 * tw + 2 * gap)
    yy = ya0 + 0.10
    modality_tile(ax, sx, yy, tw, th, "rad", True)
    ex = sx + tw + gap
    card(ax, ex, yy, tw, th, edge=C_SAT, fill="white", lw=1.5, ls=(0, (3, 2)), z=3, r=0.05)
    ax.text(ex + 0.5 * tw, yy + 0.5 * th, r"$\mathbf{E}^{m}$", fontsize=12.5, ha="center", va="center", color=TXT["sat"], zorder=4)
    modality_tile(ax, ex + tw + gap, yy, tw, th, "rain", True)

    # B: L x causal spatiotemporal blocks
    yb0, yb1 = 1.36, 3.20
    down_arrow(ax, cx, ya0 - 0.01, yb1 + 0.01)
    card(ax, cxl, yb0, cw, yb1 - yb0, edge=TAN, fill="white", lw=1.7, ls=(0, (4, 2)))
    ax.add_patch(FancyBboxPatch((cxl + 0.10, yb1 - 0.17), 0.58, 0.30, boxstyle="round,pad=0,rounding_size=0.07",
                                facecolor="white", edgecolor=TAN, lw=1.4, zorder=4))
    ax.text(cxl + 0.39, yb1 - 0.02, r"$\times L$", fontsize=12.5, fontweight="bold", color="#8A5A1E", ha="center", va="center", zorder=5)
    ax.text(cx + 0.14, yb1 - 0.40, "Causal spatiotemporal\ntransformer", fontsize=10.2, fontweight="bold", color="#8A5A1E",
            ha="center", va="center", linespacing=1.05, zorder=4)
    gw, ggap = 0.96, 0.16
    gy0, gy1 = yb0 + 0.58, yb1 - 0.70
    sxc = cx - 0.5 * (2 * gw + ggap) + 0.5 * gw
    card(ax, sxc - 0.5 * gw, gy0, gw, gy1 - gy0, edge=C_RAD, fill="white", lw=1.4, z=3)
    gcy = 0.5 * (gy0 + gy1)
    nodes = [(sxc + dx, gcy + dy) for dy in (-0.12, 0.0, 0.12) for dx in (-0.14, 0.0, 0.14)]
    for a_ in range(9):
        for b_ in range(a_ + 1, 9):
            if (a_ + b_) % 3 == 0 or a_ == 4 or b_ == 4:
                ax.plot([nodes[a_][0], nodes[b_][0]], [nodes[a_][1], nodes[b_][1]], color="#B7CBE0", lw=0.7, zorder=4)
    for (nx, ny) in nodes:
        ax.add_patch(Circle((nx, ny), 0.028, facecolor="white", edgecolor=C_RAD, lw=1.0, zorder=5))
    ax.text(sxc, yb0 + 0.27, "Spatial\nattention", fontsize=9.4, fontweight="bold", color=TXT["rad"], ha="center",
            va="center", linespacing=1.05, zorder=4)
    tcx = cx + 0.5 * (2 * gw + ggap) - 0.5 * gw
    card(ax, tcx - 0.5 * gw, gy0, gw, gy1 - gy0, edge=TAN, fill="white", lw=1.4, z=3)
    cell = 0.085
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((tcx - 2 * cell + j * cell, gcy + 2 * cell - (i + 1) * cell), cell, cell,
                                   facecolor="#E3C79A" if j <= i else "#EEEEEE", edgecolor="white", lw=0.8, zorder=4))
    ax.text(tcx, yb0 + 0.27, "Causal temporal\nattention", fontsize=8.5, fontweight="bold", color="#8A5A1E", ha="center",
            va="center", linespacing=1.05, zorder=4)
    ax.add_patch(FancyArrow(sxc + 0.5 * gw + 0.01, gcy, ggap - 0.02, 0, width=0.03, head_width=0.10, head_length=0.07,
                            length_includes_head=True, facecolor="#8A8F98", edgecolor="none", zorder=5))

    # C: modality-specific decoders (three parallel heads on the shared features)
    yc0, yc1 = 0.12, 1.20
    down_arrow(ax, cx, yb0 - 0.01, yc1 + 0.01)
    card(ax, cxl, yc0, cw, yc1 - yc0, edge=EDGE)
    ax.text(cx, yc1 - 0.15, "Modality-specific decoders", fontsize=10.2, fontweight="bold", ha="center", va="center",
            color=DARK, zorder=4)
    dw, dg = 0.68, 0.06
    dsx = cx - 0.5 * (3 * dw + 2 * dg)
    items = [("Rainfall\ndecoder", C_OUT, "#B8604E"), ("Radar\ndecoder", C_RAD, TXT["rad"]), ("Satellite\ndecoder", C_SAT, TXT["sat"])]
    for k, (txt, ec, tc) in enumerate(items):
        bx = dsx + k * (dw + dg)
        card(ax, bx, yc0 + 0.10, dw, 0.58, edge=ec, fill="white", lw=1.6, z=3, r=0.06)
        ax.text(bx + 0.5 * dw, yc0 + 0.39, txt, fontsize=8.8, fontweight="bold", color=tc, ha="center", va="center",
                linespacing=1.05, zorder=4)


def main():
    out_dir = REPO_ROOT / "Rainpaper_Elsevier" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    geojson = _load_sichuan_geojson()
    bounds = DEFAULT_GEO_BOUNDS
    lon0, lon1, lat0, lat1 = bounds
    timeline = np.asarray(Image.open(TIMELINE).convert("RGB"))
    hist = {m: load_tile(timeline, m, HIST_LAST_COL) for m in ROW}
    pred = {m: [load_tile(timeline, m, c) for c in PRED_COLS] for m in ROW}
    gt_rain = [load_tile(timeline, "rain", c) for c in GT_COLS]

    fig_w, fig_h = 12.95, 4.86
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------------- panel 1: matched observations ----------------
    P1 = (0.08, 3.98)
    panel(ax, *P1, "Matched observations", P_OBS)
    geo_aspect = (lat1 - lat0) / (lon1 - lon0)
    W, S = 2.10, 0.38
    H = W * geo_aspect * 0.56
    ox = 1.45
    layers = [
        ("rad", "Radar", "reflectivity", C_RAD, 3.12, "#F4F4F4"),
        ("sat", "Satellite", "bands 7\u201316", C_SAT, 2.06, "#B0645A"),
        ("rain", "Rainfall", "stations", C_RAIN, 1.00, "#B0645A"),
    ]
    mu, mv = (104.07 - lon0) / (lon1 - lon0), (30.67 - lat0) / (lat1 - lat0)
    pts = []
    for key, n1, n2, color, oy, bcol in layers:
        aff = Affine2D.from_values(W, 0.0, S, H, ox, oy)
        if not NOFILL:
            ax.add_patch(Polygon(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), closed=True, facecolor="black", alpha=0.10,
                                 edgecolor="none", transform=Affine2D().translate(0.014, -0.02) + aff + ax.transData, zorder=2))
        draw_layer(ax, hist[key], aff, geojson, bounds, bcol, color)
        pts.append(aff.transform((mu, mv)))
        ym = oy + 0.5 * H
        ax.add_patch(FancyBboxPatch((0.24, ym - 0.27), 0.11, 0.54, boxstyle="round,pad=0,rounding_size=0.04",
                                    facecolor="white" if NOFILL else color, edgecolor=color if NOFILL else "none", lw=1.5, zorder=3))
        ax.text(0.45, ym + 0.12, n1, fontsize=13, fontweight="bold", ha="left", va="center", color=INK, zorder=3)
        ax.text(0.45, ym - 0.15, n2, fontsize=10.5, ha="left", va="center", color=SUB, style="italic", zorder=3)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.plot([xs[0], xs[-1]], [ys[0], ys[-1]], color="#333333", lw=1.1, ls=(0, (3, 2)), zorder=6)
    ax.plot(xs, ys, "o", ms=5.2, mfc="white", mec="#333333", mew=1.3, zorder=7)
    ax.text(xs[-1] - 0.12, ys[-1] - 0.17, "same location", fontsize=10, ha="right", va="center", color="#333333", style="italic",
            zorder=8, bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.85))
    ax.text(0.5 * sum(P1), 0.58, r"$97.3^{\circ}$–$108.4^{\circ}$E, $26.1^{\circ}$–$34.25^{\circ}$N", fontsize=11, ha="center",
            va="center", color=SUB, zorder=3)
    ax.text(0.5 * sum(P1), 0.30, "same area and observation time", fontsize=11, ha="center", va="center", color=SUB, zorder=3)

    # ---------------- panel 2: input combinations ----------------
    P2 = (4.38, 7.00)
    panel(ax, *P2, r"Input combinations $\mathcal{A}$", P_IN)
    tw, th, tgap = 0.60, 0.30, 0.08
    chips_w = 3 * tw + 2 * tgap
    tx0 = 0.5 * sum(P2) - 0.5 * chips_w
    keys = ["rad", "sat", "rain"]
    y = 4.18
    groups = [
        ("Complete input", "#5E9C80", [(1, 1, 1)]),
        ("One modality missing", "#C0902F", [(1, 1, 0), (1, 0, 1), (0, 1, 1)]),
        ("Two modalities missing", "#C46F5F", [(1, 0, 0), (0, 1, 0), (0, 0, 1)]),
    ]
    row_h, pill_h = 0.385, 0.26
    for g_idx, (gname, gcol, items) in enumerate(groups):
        if g_idx > 0:
            y -= 0.06
        card(ax, P2[0] + 0.14, y - pill_h, P2[1] - P2[0] - 0.28, pill_h, edge=gcol, fill="white", lw=1.5, z=2, r=0.10)
        ax.text(P2[0] + 0.28, y - 0.5 * pill_h, gname, fontsize=11, fontweight="bold", ha="left", va="center", color=gcol, zorder=3)
        y -= pill_h + 0.07
        for flags in items:
            yc = y - 0.5 * th
            for j, (k, flag) in enumerate(zip(keys, flags)):
                modality_tile(ax, tx0 + j * (tw + tgap), yc - 0.5 * th, tw, th, k, bool(flag))
            y -= row_h
    y_mid = 0.5 * (PANEL_Y0 + PANEL_Y1 - HEAD_H)

    # ---------------- panel 3: shared model ----------------
    P3 = (7.40, 10.05)
    draw_model_panel(ax, P3)

    # ---------------- panel 4: forecast outputs ----------------
    P4 = (10.45, 12.87)
    panel(ax, *P4, "Forecast outputs", P_OUT)
    ow, oh, og = 0.44, 0.42, 0.04
    ox2 = P4[0] + 0.40

    def tag(yc, text, col):
        ax.text(P4[0] + 0.22, yc, text, fontsize=10, fontweight="bold", color=col, rotation=90, ha="center", va="center", zorder=3)

    def tlabels(yy):
        for t in range(4):
            ax.text(ox2 + t * (ow + og) + 0.5 * ow, yy, rf"$t{{+}}{t + 1}$", fontsize=10.5, ha="center", va="center",
                    color=SUB, zorder=3)

    ax.text(P4[0] + 0.14, 4.00, "Rainfall (primary)", fontsize=12, fontweight="bold", ha="left", va="center", color="#B8604E", zorder=3)
    for t in range(4):
        x = ox2 + t * (ow + og)
        thumb(ax, pred["rain"][t], x, 3.36, ow, oh, edge=C_OUT, lw=2.0)
        thumb(ax, gt_rain[t], x, 2.90, ow, oh, edge="#8A8F98", lw=1.2)
    tag(3.36 + 0.5 * oh, "Pred.", "#B8604E")
    tag(2.90 + 0.5 * oh, "Obs.", SUB)
    tlabels(2.77)
    for key, title, col, tcol, yy in [("rad", "Radar (auxiliary)", C_RAD, "#4A78A3", 1.88),
                                       ("sat", "Satellite (auxiliary)", C_SAT, "#3E7A6B", 0.66)]:
        ax.text(P4[0] + 0.14, yy + oh + 0.24, title, fontsize=12, fontweight="bold", ha="left", va="center", color=tcol, zorder=3)
        for t in range(4):
            thumb(ax, pred[key][t], ox2 + t * (ow + og), yy, ow, oh, edge=col, lw=1.4)
        tag(yy + 0.5 * oh, "Pred.", tcol)
        tlabels(yy - 0.13)

    block_arrow(ax, P1[1] + 0.04, P2[0] - 0.04, y_mid)
    block_arrow(ax, P2[1] + 0.04, P3[0] - 0.04, y_mid)
    block_arrow(ax, P3[1] + 0.04, P4[0] - 0.04, y_mid)

    stem = "Fig_intro_benchmark_a_nofill" if NOFILL else "Fig_intro_benchmark_a"
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close(fig)
    print("Saved Fig_intro_benchmark_a")


if __name__ == "__main__":
    main()
