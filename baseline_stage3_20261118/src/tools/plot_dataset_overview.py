#!/usr/bin/env python3
"""
Fig. 2: Benchmark dataset overview across spatial gauge density, chronological splits,
diurnal precipitation cycle, and frame-level wet-area distribution.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path as MplPath
from scipy.ndimage import binary_closing, gaussian_filter

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
    "axes.edgecolor": "#4B5968",
    "axes.linewidth": 0.85,
})

C_TRAIN = "#4F7CA8"
C_TEST = "#C46F5F"
C_INK = "#1E242B"
C_SUB = "#465362"
C_GRID = "#DCE3EC"
C_RAW_FILL = "#EDF2F7"
C_RAW_EDGE = "#8DA0B5"
C_POLY_EDGE = "#6B7C90"
C_PROV_EDGE = "#2B3642"
C_NIGHT = "#F2F5F9"
C_TEST_BAND = "#FCF3F1"

DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "sichuan_gauge_density",
    ["#F7FAFC", "#DEE9F4", "#A4C2DE", "#5683AF", "#264B70"],
)

DOC_DIR = REPO_ROOT / "doc"
FIG_DIR = REPO_ROOT / "Rainpaper_Elsevier" / "figs"


def _load_inputs() -> tuple[pd.DataFrame, dict, dict, dict[str, np.ndarray], dict]:
    stations = pd.read_csv(DOC_DIR / "station_lonlat.csv")
    stats = json.loads((DOC_DIR / "rainfall_bins_stats.json").read_text(encoding="utf-8"))
    sichuan = _load_sichuan_geojson()
    metrics = dict(np.load(DOC_DIR / "dataset_frame_metrics.npz", allow_pickle=False))
    diurnal = json.loads((DOC_DIR / "diurnal_stats.json").read_text(encoding="utf-8"))
    return stations, stats, sichuan, metrics, diurnal


def _build_sichuan_rings(sichuan: dict) -> list[np.ndarray]:
    rings: list[np.ndarray] = []
    for feature in sichuan.get("features", []):
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        for ring in _iter_polygon_rings(geometry):
            if len(ring) >= 3:
                closed = ring if np.allclose(ring[0], ring[-1]) else np.vstack([ring, ring[0]])
                rings.append(closed)
    return rings


def _draw_spatial_panel(
    ax: plt.Axes,
    fig: plt.Figure,
    stations: pd.DataFrame,
    stats: dict,
    sichuan: dict,
) -> None:
    lon_grid = np.linspace(97.0, 108.8, 300)
    lat_grid = np.linspace(25.8, 34.6, 240)
    dlon = float(lon_grid[1] - lon_grid[0])
    dlat = float(lat_grid[1] - lat_grid[0])
    lon_edges = np.linspace(lon_grid[0] - 0.5 * dlon, lon_grid[-1] + 0.5 * dlon, len(lon_grid) + 1)
    lat_edges = np.linspace(lat_grid[0] - 0.5 * dlat, lat_grid[-1] + 0.5 * dlat, len(lat_grid) + 1)

    # Convert binned counts into physical spatial density (stations per 100 km^2)
    hist, _, _ = np.histogram2d(stations["lat"].to_numpy(), stations["lng"].to_numpy(), bins=[lat_edges, lon_edges])
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat_grid))[:, None]
    cell_area_100km2 = (dlat * km_per_deg_lat) * (dlon * km_per_deg_lon) / 100.0
    density = gaussian_filter(hist / cell_area_100km2, sigma=2.3, mode="constant")

    rings = _build_sichuan_rings(sichuan)
    ll_lon, ll_lat = np.meshgrid(lon_grid, lat_grid)
    pts = np.column_stack([ll_lon.ravel(), ll_lat.ravel()])
    inside_mask = np.zeros(len(pts), dtype=bool)
    for ring in rings:
        inside_mask |= MplPath(ring).contains_points(pts)
    inside_mask = binary_closing(inside_mask.reshape(density.shape), iterations=1)
    masked_density = np.ma.masked_where(~inside_mask, density)

    pcm = ax.pcolormesh(
        lon_edges,
        lat_edges,
        masked_density,
        cmap=DENSITY_CMAP,
        vmin=0.0,
        vmax=8.0,
        shading="flat",
        rasterized=True,
        zorder=2,
    )

    for ring in rings:
        ax.plot(ring[:, 0], ring[:, 1], color=C_POLY_EDGE, linewidth=0.5, alpha=0.85, zorder=3)

    ax.scatter(
        stations["lng"].to_numpy(),
        stations["lat"].to_numpy(),
        s=0.65,
        color="#1B2A3A",
        alpha=0.22,
        linewidths=0.0,
        rasterized=True,
        zorder=4,
    )

    lon_min, lon_max, lat_min, lat_max = DEFAULT_GEO_BOUNDS
    ax.add_patch(
        Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            facecolor="none",
            edgecolor=C_TEST,
            linestyle="--",
            linewidth=1.35,
            zorder=5,
        )
    )

    km_per_deg_lon_30 = 111.32 * np.cos(np.deg2rad(30.2))
    bar_deg = 200.0 / km_per_deg_lon_30
    sx0, sy0 = 97.6, 26.55
    ax.plot([sx0, sx0 + bar_deg], [sy0, sy0], color=C_PROV_EDGE, lw=2.0, solid_capstyle="butt", zorder=5)
    ax.plot([sx0, sx0], [sy0 - 0.08, sy0 + 0.08], color=C_PROV_EDGE, lw=1.2, zorder=5)
    ax.plot([sx0 + 0.5 * bar_deg, sx0 + 0.5 * bar_deg], [sy0 - 0.06, sy0 + 0.06], color=C_PROV_EDGE, lw=1.0, zorder=5)
    ax.plot([sx0 + bar_deg, sx0 + bar_deg], [sy0 - 0.08, sy0 + 0.08], color=C_PROV_EDGE, lw=1.2, zorder=5)
    ax.text(sx0, sy0 + 0.16, "0", fontsize=7.3, color=C_INK, ha="center", va="bottom", zorder=5)
    ax.text(sx0 + 0.5 * bar_deg, sy0 + 0.16, "100", fontsize=7.3, color=C_INK, ha="center", va="bottom", zorder=5)
    ax.text(sx0 + bar_deg, sy0 + 0.16, "200 km", fontsize=7.3, color=C_INK, ha="center", va="bottom", zorder=5)

    nx, ny = 108.05, 33.42
    ax.annotate(
        "",
        xy=(nx, ny + 0.55),
        xytext=(nx, ny),
        arrowprops=dict(arrowstyle="-|>", color=C_PROV_EDGE, lw=1.3, mutation_scale=10),
        zorder=5,
    )
    ax.text(nx, ny - 0.14, "N", fontsize=8.1, fontweight="bold", color=C_PROV_EDGE, ha="center", va="top", zorder=5)

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="#1B2A3A", markeredgecolor="none",
            markersize=3.8, alpha=0.75, label=f"Rain gauges ({stats['stations_total']:,} total; {stats['stations_in_domain']:,} in bounds)"
        ),
        Patch(
            facecolor="none", edgecolor=C_TEST, linestyle="--", linewidth=1.25,
            label=r"Matched $512\times512$ domain ($97.3^{\circ}$–$108.4^{\circ}$E)"
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=7.3,
        frameon=True,
        facecolor="white",
        edgecolor=C_GRID,
        framealpha=0.94,
        borderpad=0.32,
        handletextpad=0.5,
    )

    ax.set_xlim(96.85, 108.75)
    ax.set_ylim(25.8, 34.65)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(30.2)), adjustable="box")
    ax.set_xticks([98, 101, 104, 107])
    ax.set_xticklabels([r"$98^{\circ}$E", r"$101^{\circ}$E", r"$104^{\circ}$E", r"$107^{\circ}$E"], fontsize=8.2)
    ax.set_yticks([26, 28, 30, 32, 34])
    ax.set_yticklabels([r"$26^{\circ}$N", r"$28^{\circ}$N", r"$30^{\circ}$N", r"$32^{\circ}$N", r"$34^{\circ}$N"], fontsize=8.2)
    ax.set_axisbelow(True)
    ax.grid(color=C_GRID, linestyle=":", linewidth=0.55, zorder=0)
    ax.set_title("(a) Observation Domain & Gauge Network", fontsize=10.2, fontweight="bold", color=C_INK, pad=8)

    cbar = fig.colorbar(pcm, ax=ax, orientation="horizontal", fraction=0.044, pad=0.075, extend="max")
    cbar.set_label(r"Rain-gauge density (stations / $100\ \mathrm{km}^2$)", fontsize=8.1, color=C_INK, labelpad=3)
    cbar.ax.tick_params(labelsize=7.7)
    cbar.outline.set_edgecolor(C_POLY_EDGE)
    cbar.outline.set_linewidth(0.7)


def _draw_split_panel(
    axes: tuple[plt.Axes, plt.Axes],
    stats: dict,
) -> None:
    months_order = ["202305", "202306", "202307", "202308", "202309", "202506", "202507", "202509"]
    month_labels = ["23-05", "23-06", "23-07", "23-08", "23-09", "25-06", "25-07", "25-09"]
    x_pos = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.35, 6.35, 7.65])
    bar_w = 0.66

    monthly_all = stats["monthly_details"]
    seq_totals = stats["seq_summary"]
    bins_summary = stats["bins_summary"]

    configs = [
        ("30", axes[0], 1880, True),
        ("10", axes[1], 5650, False),
    ]

    for interval, ax, ylim_max, is_left in configs:
        m_dict = monthly_all[interval]
        raw_vals = np.array([m_dict[m]["raw_seq_within_month"] if m in m_dict else 0 for m in months_order])
        sel_vals = np.array([m_dict[m]["sel_seq_within_month"] if m in m_dict else 0 for m in months_order])

        ax.axvspan(7.02, 8.70, facecolor=C_TEST_BAND, edgecolor="none", zorder=0)
        ax.axvline(4.68, color=C_RAW_EDGE, linestyle=":", linewidth=0.85, ymin=0.0, ymax=0.92, zorder=1)
        ax.axvline(7.02, color=C_TEST, linestyle="--", linewidth=0.95, alpha=0.75, ymin=0.0, ymax=0.95, zorder=1)

        for i, m in enumerate(months_order):
            x = x_pos[i]
            if m not in m_dict:
                ax.text(x, ylim_max * 0.04, "excl.", ha="center", va="bottom", fontsize=7.0, color=C_SUB, style="italic")
                continue

            is_test = m_dict[m]["split"] == "test"
            sel_color = C_TEST if is_test else C_TRAIN
            ax.bar(
                x,
                raw_vals[i],
                width=bar_w,
                facecolor=C_RAW_FILL,
                edgecolor=C_RAW_EDGE,
                linewidth=0.8,
                zorder=2,
            )
            ax.bar(
                x,
                sel_vals[i],
                width=bar_w,
                facecolor=sel_color,
                edgecolor="#26313D",
                linewidth=0.5,
                alpha=0.9,
                zorder=3,
            )

        tr_obs = bins_summary[interval]["train"]["n_obs"]
        te_obs = bins_summary[interval]["test"]["n_obs"]
        tr_sel = seq_totals[interval]["train"]["selected"]
        tr_raw = seq_totals[interval]["train"]["raw"]
        te_sel = seq_totals[interval]["test"]["selected"]
        te_raw = seq_totals[interval]["test"]["raw"]

        # Place observation counts cleanly in subtitle and short sequence totals strictly clear of axis edges
        ax.set_title(
            f"{interval}-min ({tr_obs:,} train / {te_obs:,} test obs)",
            fontsize=8.35,
            fontweight="bold",
            color=C_INK,
            pad=5,
        )
        ax.text(
            2.35,
            ylim_max * 0.87,
            f"Train: {tr_sel:,}/{tr_raw:,}",
            ha="center",
            va="bottom",
            fontsize=7.25,
            color="#2E547A",
            fontweight="bold",
        )
        ax.text(
            x_pos[-1],
            raw_vals[-1] + ylim_max * 0.035,
            f"Test\n{te_sel:,}/{te_raw:,}",
            ha="center",
            va="bottom",
            fontsize=6.9,
            fontweight="bold",
            color="#9B4B3D",
            linespacing=1.12,
        )

        ax.set_ylim(0, ylim_max)
        ax.set_xlim(-0.58, 8.70)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(month_labels, fontsize=7.4, rotation=26, ha="right")
        ax.tick_params(axis="y", labelsize=7.7)
        ax.yaxis.grid(True, color=C_GRID, linestyle="--", linewidth=0.55, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if is_left:
            ax.set_ylabel("Sequences", fontsize=8.4, color=C_INK)


def _draw_diurnal_panel(
    ax: plt.Axes,
    diurnal: dict,
) -> None:
    ax.axvspan(0, 7, facecolor=C_NIGHT, edgecolor="none", zorder=0)
    ax.axvspan(22, 23, facecolor=C_NIGHT, edgecolor="none", zorder=0)
    ax.text(3.5, 11.95, "Nocturnal peak", ha="center", va="top", fontsize=7.25, color=C_SUB, style="italic")

    series = [
        ("30", "test", "-", "o", C_TEST, "30-min Test (Sep 25)"),
        ("10", "test", "--", "s", C_TEST, "10-min Test (Sep 25)"),
        ("30", "train", "-", "o", C_TRAIN, "30-min Train"),
        ("10", "train", "--", "s", C_TRAIN, "10-min Train"),
    ]
    for interval, split_name, ls, marker, color, label in series:
        d = diurnal[interval][split_name]
        hrs = np.array(d["hours"])
        mean_v = np.array(d["mean_r02_pct"])
        sem_v = np.array(d["sem_r02_pct"])
        ax.fill_between(hrs, mean_v - sem_v, mean_v + sem_v, color=color, alpha=0.11, zorder=1)
        ax.plot(
            hrs,
            mean_v,
            color=color,
            linestyle=ls,
            linewidth=1.55,
            marker=marker,
            markersize=2.8,
            markevery=3,
            label=label,
            zorder=3,
        )

    ax.set_xlim(0, 23)
    ax.set_ylim(0, 12.8)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
    ax.set_xticklabels(["00", "04", "08", "12", "16", "20", "23"], fontsize=7.8)
    ax.set_xlabel("Local time (Hour, UTC+8)", fontsize=8.3, color=C_INK)
    ax.set_ylabel("Mean wet area >0.02 mm (%)", fontsize=8.2, color=C_INK)
    ax.tick_params(axis="y", labelsize=7.7)
    ax.grid(True, color=C_GRID, linestyle="--", linewidth=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper right",
        fontsize=6.85,
        frameon=True,
        facecolor="white",
        edgecolor=C_GRID,
        framealpha=0.95,
        labelspacing=0.22,
        borderpad=0.26,
        handlelength=1.8,
    )
    ax.set_title("(c) Diurnal Precipitation Cycle", fontsize=9.8, fontweight="bold", color=C_INK, pad=7)


def _draw_coverage_shift_panel(
    ax: plt.Axes,
    metrics: dict[str, np.ndarray],
) -> None:
    groups = [
        ("30m\n>0.02 mm", "i30", "r02"),
        ("30m\n>0.10 mm", "i30", "r10"),
        ("10m\n>0.02 mm", "i10", "r02"),
        ("10m\n>0.10 mm", "i10", "r10"),
    ]
    centers = np.arange(len(groups))
    offset = 0.17
    box_w = 0.25

    ax.axhline(2.0, color="#6A7786", linestyle=":", linewidth=1.1, zorder=1)

    for idx, (_, prefix, metric_key) in enumerate(groups):
        for split_name, dx, color in [("train", -offset, C_TRAIN), ("test", offset, C_TEST)]:
            vals = metrics[f"{prefix}_{split_name}_{metric_key}"] * 100.0
            p10, p25, p50, p75, p90 = np.quantile(vals, [0.10, 0.25, 0.50, 0.75, 0.90])
            mean_val = float(vals.mean())
            x = centers[idx] + dx

            ax.plot([x, x], [p10, p90], color=color, linewidth=1.15, zorder=2)
            ax.plot([x - 0.06, x + 0.06], [p10, p10], color=color, linewidth=1.05, zorder=2)
            ax.plot([x - 0.06, x + 0.06], [p90, p90], color=color, linewidth=1.05, zorder=2)

            fill_rgba = mpl.colors.to_rgba(color, alpha=0.22)
            ax.add_patch(
                Rectangle(
                    (x - 0.5 * box_w, p25),
                    box_w,
                    max(p75 - p25, 0.05),
                    facecolor=fill_rgba,
                    edgecolor=color,
                    linewidth=1.1,
                    zorder=3,
                )
            )
            ax.plot([x - 0.5 * box_w, x + 0.5 * box_w], [p50, p50], color=color, linewidth=1.7, zorder=4)
            ax.scatter([x], [mean_val], marker="D", s=16, facecolor="white", edgecolor=color, linewidth=1.1, zorder=5)

            if split_name == "test":
                ax.text(
                    x,
                    p90 + 0.45,
                    f"{mean_val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6.85,
                    fontweight="bold",
                    color="#9B4B3D",
                )
            else:
                ax.text(
                    x,
                    p90 + 0.45,
                    f"{mean_val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6.85,
                    color="#31567B",
                )

    ax.axvline(1.5, color=C_GRID, linestyle="-", linewidth=0.9, zorder=1)
    ax.set_xticks(centers)
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.6)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0.0, 19.2)
    ax.set_ylabel("Frame wet pixel coverage (%)", fontsize=8.2, color=C_INK)
    ax.tick_params(axis="y", labelsize=7.7)
    ax.grid(True, axis="y", color=C_GRID, linestyle="--", linewidth=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor=mpl.colors.to_rgba(C_TRAIN, 0.22), edgecolor=C_TRAIN, linewidth=1.0, label="Train (IQR, P10–90)"),
        Patch(facecolor=mpl.colors.to_rgba(C_TEST, 0.22), edgecolor=C_TEST, linewidth=1.0, label="Test Sep 25"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=C_INK, markersize=4.0, label="Mean"),
        Line2D([0], [0], color="#6A7786", linestyle=":", linewidth=1.15, label="2% selection cutoff"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=6.7,
        frameon=True,
        facecolor="white",
        edgecolor=C_GRID,
        framealpha=0.95,
        labelspacing=0.20,
        borderpad=0.25,
    )
    ax.set_title("(d) Rain-Coverage Distribution Shift", fontsize=9.8, fontweight="bold", color=C_INK, pad=7)


def main() -> None:
    stations, stats, sichuan, metrics, diurnal = _load_inputs()

    fig = plt.figure(figsize=(14.4, 5.75), facecolor="white")
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.10, 1.30], wspace=0.18)

    ax_map = fig.add_subplot(outer[0, 0])

    right_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], height_ratios=[1.0, 1.0], hspace=0.40)
    top_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=right_gs[0, 0], wspace=0.21)
    ax_b30 = fig.add_subplot(top_gs[0, 0])
    ax_b10 = fig.add_subplot(top_gs[0, 1])

    bot_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=right_gs[1, 0], wspace=0.24)
    ax_diurnal = fig.add_subplot(bot_gs[0, 0])
    ax_cov = fig.add_subplot(bot_gs[0, 1])

    _draw_spatial_panel(ax_map, fig, stations, stats, sichuan)
    _draw_split_panel((ax_b30, ax_b10), stats)
    _draw_diurnal_panel(ax_diurnal, diurnal)
    _draw_coverage_shift_panel(ax_cov, metrics)

    fig.canvas.draw()
    pos_b30 = ax_b30.get_position()
    pos_b10 = ax_b10.get_position()
    bx_mid = 0.5 * (pos_b30.x0 + pos_b10.x1)
    by_top = max(pos_b30.y1, pos_b10.y1)
    fig.text(
        bx_mid,
        by_top + 0.076,
        "(b) Chronological Train/Test Splits & Sample Selection",
        ha="center",
        va="bottom",
        fontsize=10.2,
        fontweight="bold",
        color=C_INK,
    )
    b_handles = [
        Patch(facecolor=C_RAW_FILL, edgecolor=C_RAW_EDGE, linewidth=0.8, label="All 8-frame sequences"),
        Patch(facecolor=C_TRAIN, edgecolor="#26313D", linewidth=0.5, label="Train (selected)"),
        Patch(facecolor=C_TEST, edgecolor="#26313D", linewidth=0.5, label="Test Sep 2025 (selected)"),
    ]
    fig.legend(
        handles=b_handles,
        loc="lower center",
        bbox_to_anchor=(bx_mid, by_top + 0.030),
        ncol=3,
        fontsize=7.35,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.2,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIG_DIR / "Fig_dataset_overview.pdf"
    out_png = FIG_DIR / "Fig_dataset_overview.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.06, facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
