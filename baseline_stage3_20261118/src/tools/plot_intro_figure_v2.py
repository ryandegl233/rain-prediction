#!/usr/bin/env python3
"""
Fig. 1 v2 (prototype): Clean outline style with comic-sci-illustration 3D atmospheric column
for Panel 1 (Satellite -> Radar -> Ground Stations), keeping original scripts and outputs untouched.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrow, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge
from matplotlib.transforms import Affine2D
from PIL import Image

from src.tools.plot_discussion_figures import (
    DEFAULT_GEO_BOUNDS,
    _iter_polygon_rings,
    _load_sichuan_geojson,
)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def darken(hex_color: str, factor: float = 0.72) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % (int(r * factor), int(g * factor), int(b * factor))


def lighten(hex_color: str, amount: float = 0.82) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return "#%02X%02X%02X" % (r, g, b)


C_RAD = "#4F7CA8"
C_SAT = "#4E9484"
C_RAIN = "#C4903A"
C_OUT = "#C46F5F"
SLATE = "#62748C"
COMIC_INK = "#24303C"
SUB = "#4A525D"
TXT = {"rad": "#386087", "sat": "#326D5F", "rain": "#8F6116"}

TIMELINE = (
    REPO_ROOT / "runs" / "next_frame_final" / "2026-10-12" / "20-17-52_stage1_next_frame_block"
    / "val_viz" / "step_00096000" / "rollout_sample3_timeline.jpg"
)
ROW = {"rad": 0, "sat": 1, "rain": 2}
HIST_LAST_COL = 3
PRED_COLS = [4, 5, 6, 7]
GT_COLS = [8, 9, 10, 11]

PANEL_Y0, PANEL_Y1, HEAD_H = 0.08, 4.84, 0.52


def load_tile(img: np.ndarray, modality: str, col: int) -> np.ndarray:
    left_label_w, cell_gap, top_header_h, row_gap = 84, 6, 26, 8
    cols, rows = 12, 3
    tile_w = (img.shape[1] - left_label_w - (cols - 1) * cell_gap) // cols
    tile_h = (img.shape[0] - top_header_h - (rows - 1) * row_gap) // rows
    y0 = top_header_h + ROW[modality] * (tile_h + row_gap)
    x0 = left_label_w + col * (tile_w + cell_gap)
    return img[y0:y0 + tile_h, x0:x0 + tile_w]


def draw_panel_frame(ax: plt.Axes, x0: float, x1: float, idx: int, title: str, fs: float = 12.6) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x0, PANEL_Y0),
            x1 - x0,
            PANEL_Y1 - PANEL_Y0,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor="white",
            edgecolor=SLATE,
            lw=1.45,
            zorder=0,
        )
    )
    ax.plot([x0 + 0.16, x1 - 0.16], [PANEL_Y1 - HEAD_H, PANEL_Y1 - HEAD_H], color=SLATE, lw=0.95, alpha=0.65, zorder=1)
    yh = PANEL_Y1 - 0.5 * HEAD_H - 0.01
    ax.add_patch(Circle((x0 + 0.25, yh), 0.125, facecolor="white", edgecolor=SLATE, lw=1.25, zorder=2))
    ax.text(x0 + 0.25, yh - 0.005, str(idx), fontsize=9.8, fontweight="bold", ha="center", va="center", color=COMIC_INK, zorder=3)
    ax.text(x0 + 0.46, yh, title, fontsize=fs, fontweight="bold", ha="left", va="center", color=COMIC_INK, zorder=2)


def draw_block_arrow(ax: plt.Axes, x0: float, x1: float, y: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0 + 0.03, y),
            (x1 - 0.02, y),
            arrowstyle="-|>",
            mutation_scale=17,
            lw=1.8,
            color=SLATE,
            zorder=5,
            shrinkA=0,
            shrinkB=0,
        )
    )


def draw_down_arrow(ax: plt.Axes, x: float, y_top: float, y_bot: float) -> None:
    ax.add_patch(
        FancyArrow(
            x,
            y_top,
            0,
            y_bot - y_top,
            width=0.048,
            head_width=0.16,
            head_length=0.085,
            length_includes_head=True,
            facecolor="#7C8CA3",
            edgecolor="none",
            zorder=5,
        )
    )


def draw_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    edge: str,
    fill: str = "white",
    lw: float = 1.4,
    ls: str | tuple = "-",
    z: int = 2,
    r: float = 0.07,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fill,
            edgecolor=edge,
            lw=lw,
            linestyle=ls,
            zorder=z,
        )
    )


def draw_thumb(ax: plt.Axes, tile: np.ndarray, x: float, y: float, w: float, h: float, edge: str, lw: float = 1.0, z: int = 3) -> None:
    ax.imshow(tile, extent=(x, x + w, y, y + h), origin="upper", interpolation="bilinear", zorder=z)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=edge, lw=lw, zorder=z + 1))


def draw_satellite_icon(ax: plt.Axes, cx: float, cy: float, slab_target: tuple[float, float, float, float]) -> None:
    # Subtle cel-shaded vector weather satellite + downward scan cone toward top slab
    fill_wing = lighten(C_SAT, 0.72)
    fill_body = lighten(C_SAT, 0.88)
    edge = COMIC_INK

    # Solar panel wings (left-up and right-down tilted slightly)
    ang = np.deg2rad(-22)
    c, s = np.cos(ang), np.sin(ang)

    def rot(px: float, py: float) -> tuple[float, float]:
        return cx + px * c - py * s, cy + px * s + py * c

    for sign in (-1, 1):
        pts = np.array([rot(sign * 0.11, -0.075), rot(sign * 0.34, -0.075), rot(sign * 0.34, 0.075), rot(sign * 0.11, 0.075)])
        ax.add_patch(Polygon(pts + [0.018, -0.02], closed=True, facecolor="#D6DFE8", edgecolor="none", zorder=2))
        ax.add_patch(Polygon(pts, closed=True, facecolor=fill_wing, edgecolor=edge, lw=1.25, zorder=3))
        # Panel grid lines
        for frac in (0.35, 0.68):
            x_g = sign * (0.11 + frac * 0.23)
            p1, p2 = rot(x_g, -0.075), rot(x_g, 0.075)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=C_SAT, lw=0.85, zorder=4)
        # Strut bar
        s1, s2 = rot(sign * 0.07, 0.0), rot(sign * 0.11, 0.0)
        ax.plot([s1[0], s2[0]], [s1[1], s2[1]], color=edge, lw=1.4, zorder=3)

    # Central spacecraft bus
    bus_pts = np.array([rot(-0.08, -0.085), rot(0.08, -0.085), rot(0.08, 0.085), rot(-0.08, 0.085)])
    ax.add_patch(Polygon(bus_pts + [0.018, -0.02], closed=True, facecolor="#C7D3DF", edgecolor="none", zorder=3))
    ax.add_patch(Polygon(bus_pts, closed=True, facecolor=fill_body, edgecolor=edge, lw=1.35, zorder=4))

    # Downward sensor dish pointing toward satellite layer
    dish_c = rot(0.02, -0.10)
    ax.add_patch(Wedge(dish_c, 0.065, 200, 340, facecolor=C_SAT, edgecolor=edge, lw=1.15, zorder=5))

    # Translucent downward-right scan FOV beam into the satellite map slab
    tx_l, ty_t, tx_r, ty_b = slab_target
    beam = np.array([dish_c, (tx_l, ty_t), (tx_l + 0.08, ty_b)])
    ax.add_patch(Polygon(beam, closed=True, facecolor=C_SAT, alpha=0.10, edgecolor="none", zorder=1))
    ax.plot([dish_c[0], tx_l], [dish_c[1], ty_t], color=C_SAT, lw=0.95, ls=(0, (3, 2.5)), alpha=0.85, zorder=2)
    ax.plot([dish_c[0], tx_l + 0.08], [dish_c[1], ty_b], color=C_SAT, lw=0.95, ls=(0, (3, 2.5)), alpha=0.85, zorder=2)


def draw_radar_icon(ax: plt.Axes, cx: float, cy: float, slab_mid_left: tuple[float, float]) -> None:
    # Crisp comic-sci Doppler radar radome tower + sweeping wave arcs
    edge = COMIC_INK
    # Ground anchor line
    yb = cy - 0.20
    ax.plot([cx - 0.24, cx + 0.24], [yb, yb], color=edge, lw=1.2, solid_capstyle="round", zorder=3)

    # Tapered tower pedestal + shadow
    tower = np.array([[cx - 0.11, yb], [cx + 0.11, yb], [cx + 0.065, cy - 0.02], [cx - 0.065, cy - 0.02]])
    ax.add_patch(Polygon(tower + [0.02, -0.01], closed=True, facecolor="#D2DCE6", edgecolor="none", zorder=2))
    ax.add_patch(Polygon(tower, closed=True, facecolor="#F0F5FA", edgecolor=edge, lw=1.25, zorder=3))

    # Spherical radome
    rc = (cx, cy + 0.07)
    ax.add_patch(Circle((rc[0] + 0.02, rc[1] - 0.015), 0.125, facecolor="#C7D6E6", edgecolor="none", zorder=3))
    ax.add_patch(Circle(rc, 0.125, facecolor=lighten(C_RAD, 0.85), edgecolor=edge, lw=1.35, zorder=4))
    # Geodesic seam accents
    ax.add_patch(Arc(rc, 0.16, 0.24, angle=0, theta1=60, theta2=240, color=C_RAD, lw=0.85, zorder=5))
    ax.plot([rc[0] - 0.12, rc[0] + 0.12], [rc[1], rc[1]], color=C_RAD, lw=0.75, alpha=0.8, zorder=5)

    # Outward radar pulse arcs toward the radar reflectivity map
    ang = np.degrees(np.arctan2(slab_mid_left[1] - rc[1], slab_mid_left[0] - rc[0]))
    for rad_r, alp in ((0.22, 0.90), (0.33, 0.72), (0.44, 0.50)):
        ax.add_patch(Arc(rc, 2 * rad_r, 2 * rad_r, angle=0, theta1=ang - 24, theta2=ang + 24,
                         color=C_RAD, lw=1.35, alpha=alp, zorder=4))
    ax.plot([rc[0] + 0.44 * np.cos(np.deg2rad(ang)), slab_mid_left[0] + 0.02],
            [rc[1] + 0.44 * np.sin(np.deg2rad(ang)), slab_mid_left[1]],
            color=C_RAD, lw=0.95, ls=(0, (3, 2.5)), alpha=0.8, zorder=2)


def draw_gauge_icon(ax: plt.Axes, cx: float, cy: float, slab_bot_left: tuple[float, float]) -> None:
    # Crisp comic-sci rain-gauge cylinder + minimal cloud & falling raindrops
    edge = COMIC_INK
    yb = cy - 0.20
    ax.plot([cx - 0.25, cx + 0.25], [yb, yb], color=edge, lw=1.2, solid_capstyle="round", zorder=3)

    # Gauge cylinder with cel-shaded shadow and collected water band
    gw, gh = 0.16, 0.23
    gx = cx - 0.08
    ax.add_patch(FancyBboxPatch((gx + 0.022, yb - 0.005), gw, gh, boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor="#DFD6C5", edgecolor="none", zorder=2))
    ax.add_patch(FancyBboxPatch((gx, yb), gw, gh, boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor="white", edgecolor=edge, lw=1.3, zorder=3))
    # Collected water level inside gauge
    ax.add_patch(Rectangle((gx + 0.012, yb + 0.012), gw - 0.024, 0.075,
                           facecolor=lighten(C_RAIN, 0.62), edgecolor="none", zorder=3))
    # Graduation tick lines
    for dy in (0.06, 0.115, 0.17):
        ax.plot([gx + gw - 0.055, gx + gw - 0.008], [yb + dy, yb + dy], color=C_RAIN, lw=0.95, zorder=4)
    # Funnel rim cap
    ax.add_patch(FancyBboxPatch((gx - 0.015, yb + gh - 0.02), gw + 0.03, 0.04,
                                boxstyle="round,pad=0,rounding_size=0.01",
                                facecolor=lighten(C_RAIN, 0.78), edgecolor=edge, lw=1.2, zorder=4))

    # Minimal raindrops & tiny cloud puff above the gauge
    cloud_y = yb + gh + 0.13
    for ox, oy, cr in ((-0.065, -0.01, 0.048), (0.0, 0.02, 0.062), (0.065, -0.01, 0.046)):
        ax.add_patch(Circle((cx + ox, cloud_y + oy), cr, facecolor="#F2F5F8", edgecolor=edge, lw=1.05, zorder=4))
    ax.add_patch(Rectangle((cx - 0.07, cloud_y - 0.055), 0.14, 0.06, facecolor="#F2F5F8", edgecolor="none", zorder=4))
    ax.plot([cx - 0.075, cx + 0.075], [cloud_y - 0.055, cloud_y - 0.055], color=edge, lw=1.05, zorder=5)
    for dx in (-0.04, 0.02):
        ax.plot([cx + dx, cx + dx - 0.018], [cloud_y - 0.075, cloud_y - 0.11], color=C_RAIN, lw=1.35, solid_capstyle="round", zorder=5)

    # Link arrow/ray to the bottom station rainfall slab
    ax.plot([gx + gw + 0.03, slab_bot_left[0] + 0.02], [yb + 0.55 * gh, slab_bot_left[1]],
            color=C_RAIN, lw=0.95, ls=(0, (3, 2.5)), alpha=0.85, zorder=2)


def draw_3d_layer(
    ax: plt.Axes,
    tile: np.ndarray,
    aff: Affine2D,
    geojson: dict,
    bounds: tuple[float, float, float, float],
    boundary_color: str,
    edge_color: str,
) -> list[tuple[float, float]]:
    lon0, lon1, lat0, lat1 = bounds
    tr = aff + ax.transData
    corners_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    corners_xy = [tuple(aff.transform(pt)) for pt in corners_uv]

    # Subtle stacked historical frame edges behind the main tile (signifying T_h sequence)
    for back_dx, back_dy, alp in ((0.11, 0.075, 0.45), (0.055, 0.038, 0.70)):
        back_tr = Affine2D().translate(back_dx, back_dy) + aff + ax.transData
        ax.add_patch(
            Polygon(
                corners_uv,
                closed=True,
                facecolor="#F8FAFC",
                edgecolor="#98A6B8",
                lw=0.8,
                alpha=alp,
                transform=back_tr,
                zorder=2,
            )
        )

    # Cel-shaded hard drop-shadow + 3D extruded slab thickness
    thick_dx, thick_dy = 0.0, -0.058
    shadow_pts = np.array([
        (corners_xy[0][0] + 0.045, corners_xy[0][1] - 0.095),
        (corners_xy[1][0] + 0.045, corners_xy[1][1] - 0.095),
        (corners_xy[2][0] + 0.045, corners_xy[2][1] - 0.095),
        (corners_xy[3][0] + 0.045, corners_xy[3][1] - 0.095),
    ])
    ax.add_patch(Polygon(shadow_pts, closed=True, facecolor="#DCE4EE", edgecolor="none", zorder=2))

    # Front extruded edge (between corners 0 and 1)
    front_wall = np.array([
        corners_xy[0],
        corners_xy[1],
        (corners_xy[1][0] + thick_dx, corners_xy[1][1] + thick_dy),
        (corners_xy[0][0] + thick_dx, corners_xy[0][1] + thick_dy),
    ])
    ax.add_patch(Polygon(front_wall, closed=True, facecolor=darken(edge_color, 0.72), edgecolor=COMIC_INK, lw=1.05, zorder=3))

    # Map raster & geo boundaries clipped cleanly to main tilted parallelogram
    clip = Polygon(corners_uv, closed=True, transform=tr)
    ax.imshow(tile, extent=(0, 1, 0, 1), origin="upper", transform=tr, interpolation="bilinear", zorder=4)
    for feature in geojson.get("features", []):
        for ring in _iter_polygon_rings(feature.get("geometry", {})):
            u = (ring[:, 0] - lon0) / (lon1 - lon0)
            v = (ring[:, 1] - lat0) / (lat1 - lat0)
            (line,) = ax.plot(u, v, color=boundary_color, lw=0.48, alpha=0.88, transform=tr, zorder=5)
            line.set_clip_path(clip)
    ax.add_patch(Polygon(corners_uv, closed=True, fill=False, edgecolor=COMIC_INK, lw=1.45, transform=tr, zorder=6))
    ax.add_patch(Polygon(corners_uv, closed=True, fill=False, edgecolor=edge_color, lw=0.9, transform=tr, zorder=6))

    return corners_xy


def modality_tile(ax: plt.Axes, x: float, y: float, w: float, h: float, key: str, present: bool, fs: float = 9.8) -> None:
    col = {"rad": C_RAD, "sat": C_SAT, "rain": C_RAIN}[key]
    if present:
        # Subtle cel-shaded hard micro-shadow + faint 10% tint wash matching Panel 1
        draw_card(ax, x + 0.015, y - 0.018, w, h, edge="none", fill="#DCE4EE", lw=0.0, z=2, r=0.055)
        draw_card(ax, x, y, w, h, edge=col, fill=lighten(col, 0.90), lw=1.45, z=3, r=0.055)
        ax.text(x + 0.5 * w, y + 0.5 * h, key, fontsize=fs, fontweight="bold", color=TXT[key], ha="center", va="center", zorder=4)
    else:
        # Clean dashed ghost slot visually echoing Panel 3's E^m replacement block
        draw_card(ax, x, y, w, h, edge="#BEC8D5", fill="#FAFBFD", lw=1.05, ls=(0, (3, 2)), z=3, r=0.055)
        ax.text(x + 0.5 * w, y + 0.5 * h, "\u2014", fontsize=10.5, fontweight="bold", color="#9CA9B8", ha="center", va="center", zorder=4)


def draw_model_panel(ax: plt.Axes, P3: tuple[float, float]) -> None:
    draw_panel_frame(ax, *P3, 3, "Shared model")
    x0, x1 = P3
    cx = 0.5 * (x0 + x1)
    cw = x1 - x0 - 0.30
    cxl = x0 + 0.15
    EDGE = "#7C8CA3"
    DARK = "#2B3746"
    TAN = "#D9A15B"

    ya0, ya1 = 3.38, 4.22
    draw_card(ax, cxl, ya0, cw, ya1 - ya0, edge=EDGE)
    ax.text(cx, ya1 - 0.15, "Missing-modality replacement", fontsize=9.4, fontweight="bold", ha="center", va="center", color=DARK, zorder=4)
    tw, th, gap = 0.58, 0.30, 0.10
    sx = cx - 0.5 * (3 * tw + 2 * gap)
    yy = ya0 + 0.10
    modality_tile(ax, sx, yy, tw, th, "rad", True)
    ex = sx + tw + gap
    draw_card(ax, ex, yy, tw, th, edge=C_SAT, fill="white", lw=1.4, ls=(0, (3, 2)), z=3, r=0.05)
    ax.text(ex + 0.5 * tw, yy + 0.5 * th, r"$\mathbf{E}^{m}$", fontsize=12.5, ha="center", va="center", color=TXT["sat"], zorder=4)
    modality_tile(ax, ex + tw + gap, yy, tw, th, "rain", True)

    yb0, yb1 = 1.38, 3.24
    draw_down_arrow(ax, cx, ya0 - 0.01, yb1 + 0.01)
    draw_card(ax, cxl, yb0, cw, yb1 - yb0, edge=TAN, fill="white", lw=1.45, ls=(0, (4, 2)))
    ax.add_patch(
        FancyBboxPatch((cxl + 0.10, yb1 - 0.17), 0.58, 0.30, boxstyle="round,pad=0,rounding_size=0.07",
                       facecolor="white", edgecolor=TAN, lw=1.35, zorder=4)
    )
    ax.text(cxl + 0.39, yb1 - 0.02, r"$\times L$", fontsize=12.2, fontweight="bold", color="#8A5A1E", ha="center", va="center", zorder=5)
    ax.text(cx + 0.14, yb1 - 0.40, "Causal spatiotemporal\ntransformer", fontsize=10.1, fontweight="bold", color="#8A5A1E",
            ha="center", va="center", linespacing=1.05, zorder=4)

    gw, ggap = 0.96, 0.16
    gy0, gy1 = yb0 + 0.58, yb1 - 0.70
    sxc = cx - 0.5 * (2 * gw + ggap) + 0.5 * gw
    draw_card(ax, sxc - 0.5 * gw, gy0, gw, gy1 - gy0, edge=C_RAD, fill="white", lw=1.25, z=3)
    gcy = 0.5 * (gy0 + gy1)
    nodes = [(sxc + dx, gcy + dy) for dy in (-0.12, 0.0, 0.12) for dx in (-0.14, 0.0, 0.14)]
    for a_ in range(9):
        for b_ in range(a_ + 1, 9):
            if (a_ + b_) % 3 == 0 or a_ == 4 or b_ == 4:
                ax.plot([nodes[a_][0], nodes[b_][0]], [nodes[a_][1], nodes[b_][1]], color="#B7CBE0", lw=0.7, zorder=4)
    for (nx, ny) in nodes:
        ax.add_patch(Circle((nx, ny), 0.028, facecolor="white", edgecolor=C_RAD, lw=1.0, zorder=5))
    ax.text(sxc, yb0 + 0.27, "Spatial\nattention", fontsize=9.3, fontweight="bold", color=TXT["rad"], ha="center",
            va="center", linespacing=1.05, zorder=4)

    tcx = cx + 0.5 * (2 * gw + ggap) - 0.5 * gw
    draw_card(ax, tcx - 0.5 * gw, gy0, gw, gy1 - gy0, edge=TAN, fill="white", lw=1.25, z=3)
    cell = 0.085
    for i in range(4):
        for j in range(4):
            ax.add_patch(
                Rectangle(
                    (tcx - 2 * cell + j * cell, gcy + 2 * cell - (i + 1) * cell),
                    cell,
                    cell,
                    facecolor="#E3C79A" if j <= i else "#EEEEEE",
                    edgecolor="white",
                    lw=0.8,
                    zorder=4,
                )
            )
    ax.text(tcx, yb0 + 0.27, "Causal temporal\nattention", fontsize=8.5, fontweight="bold", color="#8A5A1E", ha="center",
            va="center", linespacing=1.05, zorder=4)
    ax.add_patch(
        FancyArrow(sxc + 0.5 * gw + 0.01, gcy, ggap - 0.02, 0, width=0.03, head_width=0.10, head_length=0.07,
                   length_includes_head=True, facecolor="#8A8F98", edgecolor="none", zorder=5)
    )

    yc0, yc1 = 0.14, 1.22
    draw_down_arrow(ax, cx, yb0 - 0.01, yc1 + 0.01)
    draw_card(ax, cxl, yc0, cw, yc1 - yc0, edge=EDGE)
    ax.text(cx, yc1 - 0.15, "Modality-specific decoders", fontsize=10.1, fontweight="bold", ha="center", va="center", color=DARK, zorder=4)
    dw, dg = 0.68, 0.06
    dsx = cx - 0.5 * (3 * dw + 2 * dg)
    items = [("Rainfall\ndecoder", C_OUT, "#B8604E"), ("Radar\ndecoder", C_RAD, TXT["rad"]), ("Satellite\ndecoder", C_SAT, TXT["sat"])]
    for k, (txt, ec, tc) in enumerate(items):
        bx = dsx + k * (dw + dg)
        draw_card(ax, bx, yc0 + 0.10, dw, 0.58, edge=ec, fill="white", lw=1.4, z=3, r=0.06)
        ax.text(bx + 0.5 * dw, yc0 + 0.39, txt, fontsize=8.7, fontweight="bold", color=tc, ha="center", va="center", linespacing=1.05, zorder=4)


def main() -> None:
    out_dir = REPO_ROOT / "Rainpaper_Elsevier" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    geojson = _load_sichuan_geojson()
    bounds = DEFAULT_GEO_BOUNDS
    lon0, lon1, lat0, lat1 = bounds
    timeline = np.asarray(Image.open(TIMELINE).convert("RGB"))
    hist = {m: load_tile(timeline, m, HIST_LAST_COL) for m in ROW}
    pred = {m: [load_tile(timeline, m, c) for c in PRED_COLS] for m in ROW}
    gt_rain = [load_tile(timeline, "rain", c) for c in GT_COLS]

    fig_w, fig_h = 13.25, 4.92
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---------------- panel 1: matched observations (vertical physical column + comic sci icons) ----------------
    P1 = (0.08, 4.28)
    draw_panel_frame(ax, *P1, 1, "Matched observations")

    geo_aspect = (lat1 - lat0) / (lon1 - lon0)
    W, S = 1.96, 0.34
    H = W * geo_aspect * 0.54
    ox = 1.72

    # Physical atmospheric altitude order: Satellite (top) -> Radar (mid) -> Rainfall gauges (bottom)
    layers = [
        ("sat", "Satellite", "bands 7\u201316", C_SAT, 3.12, "#B0645A"),
        ("rad", "Radar", "reflectivity", C_RAD, 2.06, "#F4F4F4"),
        ("rain", "Rainfall", "stations", C_RAIN, 1.00, "#B0645A"),
    ]
    mu, mv = (104.07 - lon0) / (lon1 - lon0), (30.67 - lat0) / (lat1 - lat0)
    pts: list[tuple[float, float]] = []
    all_corners: list[list[tuple[float, float]]] = []

    for key, name_lbl, sub_lbl, color, oy, bcol in layers:
        aff = Affine2D.from_values(W, 0.0, S, H, ox, oy)
        corners = draw_3d_layer(ax, hist[key], aff, geojson, bounds, bcol, color)
        all_corners.append(corners)
        pts.append(aff.transform((mu, mv)))

        # Sensor icon positioned cleanly above-left; label placed in clear open space below the scan beam
        ix = 0.50
        if key == "sat":
            draw_satellite_icon(ax, ix, oy + H + 0.06, (corners[3][0], corners[3][1], corners[0][0], corners[0][1]))
        elif key == "rad":
            mid_l = (0.5 * (corners[0][0] + corners[3][0]), 0.5 * (corners[0][1] + corners[3][1]))
            draw_radar_icon(ax, ix, oy + H - 0.06, mid_l)
        else:
            mid_l = (0.5 * (corners[0][0] + corners[3][0]), 0.5 * (corners[0][1] + corners[3][1]))
            draw_gauge_icon(ax, ix, oy + H - 0.05, mid_l)

        lx = 0.90
        ly = oy + 0.17
        ax.text(lx, ly + 0.11, name_lbl, fontsize=11.0, fontweight="bold", ha="center", va="center",
                color=TXT[key], zorder=6)
        ax.text(lx, ly - 0.11, sub_lbl, fontsize=9.1, ha="center", va="center",
                color=SUB, style="italic", zorder=6)

    # 3D co-registered corner struts linking top and bottom slabs
    top_c, bot_c = all_corners[0], all_corners[-1]
    for ci in range(4):
        ax.plot(
            [top_c[ci][0], bot_c[ci][0]],
            [top_c[ci][1], bot_c[ci][1]],
            color=SLATE,
            lw=0.75,
            ls=(0, (2.5, 2.5)),
            alpha=0.42,
            zorder=1,
        )

    # Highlighted sampling alignment ray & external clean callout (never overlapping Sichuan map)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.plot([xs[0], xs[-1]], [ys[0], ys[-1]], color=COMIC_INK, lw=1.15, ls=(0, (3, 2)), zorder=7)
    for x_p, y_p in zip(xs, ys):
        ax.add_patch(Circle((x_p, y_p), 0.052, facecolor="white", edgecolor=COMIC_INK, lw=1.25, zorder=8))
        ax.add_patch(Circle((x_p, y_p), 0.020, facecolor=C_OUT, edgecolor="none", zorder=9))

    # Clean elbow callout pointing cleanly below-right of the bottom slab
    lbl_x, lbl_y = 3.42, 0.78
    ax.plot([xs[-1], xs[-1] + 0.18, lbl_x - 0.04], [ys[-1], lbl_y, lbl_y], color=COMIC_INK, lw=0.9, zorder=7)
    ax.text(lbl_x, lbl_y, "same point", fontsize=8.8, ha="left", va="center", color=COMIC_INK, style="italic", zorder=8)

    ax.text(0.5 * sum(P1), 0.50, r"$97.3^{\circ}$–$108.4^{\circ}$E, $26.1^{\circ}$–$34.25^{\circ}$N",
            fontsize=10.4, ha="center", va="center", color=COMIC_INK, zorder=4)
    ax.text(0.5 * sum(P1), 0.25, r"aligned $T_h$-frame stacks & observation time",
            fontsize=9.6, ha="center", va="center", color=SUB, zorder=4)

    # ---------------- panel 2: modality availability ----------------
    P2 = (4.64, 7.26)
    draw_panel_frame(ax, *P2, 2, r"Modality availability $\mathcal{A}$")
    tw, th, tgap = 0.58, 0.30, 0.09
    chips_w = 3 * tw + 2 * tgap
    tx0 = P2[0] + 0.24
    keys = ["rad", "sat", "rain"]

    # Clean group dividers without heavy bordered pill banners + right convergence bracket
    groups = [
        ("Complete", r"$\mathcal{A}=\mathcal{M}$", "#3B7A6C", [(1, 1, 1)]),
        ("One missing", r"$|\mathcal{A}|=2$", "#A87923", [(1, 1, 0), (1, 0, 1), (0, 1, 1)]),
        ("Two missing", r"$|\mathcal{A}|=1$", "#B55D4D", [(1, 0, 0), (0, 1, 0), (0, 0, 1)]),
    ]
    y = 4.06
    row_h = 0.405
    row_centers: list[float] = []
    for g_idx, (gname, gmath, gcol, items) in enumerate(groups):
        if g_idx > 0:
            y -= 0.14
        ax.text(tx0, y - 0.04, gname, fontsize=9.8, fontweight="bold", ha="left", va="center", color=gcol, zorder=3)
        ax.text(tx0 + chips_w, y - 0.04, gmath, fontsize=8.8, ha="right", va="center", color=SUB, zorder=3)
        ax.plot([tx0, tx0 + chips_w], [y - 0.16, y - 0.16], color=gcol, lw=0.85, alpha=0.45, zorder=2)
        y -= 0.27
        for flags in items:
            yc = y - 0.5 * th
            row_centers.append(yc)
            for j, (k, flag) in enumerate(zip(keys, flags)):
                modality_tile(ax, tx0 + j * (tw + tgap), yc - 0.5 * th, tw, th, k, bool(flag))
            y -= row_h

    # Clean right collector bracket tying all 7 configurations to one shared parameter set
    bx = tx0 + chips_w + 0.12
    by_top, by_bot = row_centers[0] + 0.07, row_centers[-1] - 0.07
    by_mid = 0.5 * (by_top + by_bot)
    ax.plot([bx, bx + 0.08, bx + 0.08, bx], [by_top, by_top, by_bot, by_bot], color="#95A4B7", lw=1.15, zorder=2)
    ax.plot([bx + 0.08, bx + 0.13], [by_mid, by_mid], color="#95A4B7", lw=1.15, zorder=2)
    ax.text(bx + 0.20, by_mid, r"7 settings $\rightarrow$ shared $\theta$", fontsize=8.1, fontweight="bold",
            color=SUB, rotation=90, ha="center", va="center", zorder=3)

    # ---------------- panel 3: shared model ----------------
    P3 = (7.62, 10.30)
    draw_model_panel(ax, P3)

    # ---------------- panel 4: forecast outputs ----------------
    P4 = (10.66, 13.15)
    draw_panel_frame(ax, *P4, 4, "Forecast outputs")
    ow, oh, og = 0.44, 0.42, 0.04
    ox2 = P4[0] + 0.44

    def tag(yc: float, text: str, col: str) -> None:
        ax.text(P4[0] + 0.24, yc, text, fontsize=9.8, fontweight="bold", color=col, rotation=90, ha="center", va="center", zorder=3)

    def tlabels(yy: float) -> None:
        for t in range(4):
            ax.text(ox2 + t * (ow + og) + 0.5 * ow, yy, rf"$t{{+}}{t + 1}$", fontsize=10.0, ha="center", va="center", color=SUB, zorder=3)

    ax.text(P4[0] + 0.16, 4.04, "Rainfall (primary)", fontsize=11.6, fontweight="bold", ha="left", va="center", color="#B8604E", zorder=3)
    for t in range(4):
        x = ox2 + t * (ow + og)
        draw_thumb(ax, pred["rain"][t], x, 3.40, ow, oh, edge=C_OUT, lw=1.65)
        draw_thumb(ax, gt_rain[t], x, 2.94, ow, oh, edge="#8A8F98", lw=1.0)
    tag(3.40 + 0.5 * oh, "Pred.", "#B8604E")
    tag(2.94 + 0.5 * oh, "Obs.", SUB)
    tlabels(2.81)

    for key, title, col, tcol, yy in [("rad", "Radar (auxiliary)", C_RAD, TXT["rad"], 1.92),
                                       ("sat", "Satellite (auxiliary)", C_SAT, TXT["sat"], 0.70)]:
        ax.text(P4[0] + 0.16, yy + oh + 0.24, title, fontsize=11.6, fontweight="bold", ha="left", va="center", color=tcol, zorder=3)
        for t in range(4):
            draw_thumb(ax, pred[key][t], ox2 + t * (ow + og), yy, ow, oh, edge=col, lw=1.2)
        tag(yy + 0.5 * oh, "Pred.", tcol)
        tlabels(yy - 0.13)

    y_mid = 0.5 * (PANEL_Y0 + PANEL_Y1 - HEAD_H)
    draw_block_arrow(ax, P1[1] + 0.04, P2[0] - 0.04, y_mid)
    draw_block_arrow(ax, P2[1] + 0.04, P3[0] - 0.04, y_mid)
    draw_block_arrow(ax, P3[1] + 0.04, P4[0] - 0.04, y_mid)

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"Fig_intro_benchmark_v2.{ext}", bbox_inches="tight", pad_inches=0.04, dpi=300, facecolor="white")
    plt.close(fig)
    print("Saved Fig_intro_benchmark_v2.pdf and Fig_intro_benchmark_v2.png")


if __name__ == "__main__":
    main()
