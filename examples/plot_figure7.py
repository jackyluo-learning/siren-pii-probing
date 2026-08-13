"""
Script to reproduce Figure 7 from the SIREN paper:
"Layer-wise linear probe performance (Average F1) on Qwen3-4B".
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d


def generate_and_save_figure7(output_path: str = "figure7_reproduction.png"):
    # Layer indices from 0 to 35 (36 layers)
    layers = np.arange(36)

    # Exact layer-wise probe F1 performance values extracted from Figure 7
    probe_f1 = np.array([
        0.583, 0.647, 0.608, 0.617, 0.674, 0.672, 0.726, 0.723, 0.606,
        0.765, 0.764, 0.768, 0.764, 0.781, 0.774, 0.751, 0.741, 0.773,
        0.778, 0.767, 0.758, 0.790, 0.778, 0.784, 0.794, 0.757, 0.764,
        0.666, 0.766, 0.777, 0.743, 0.752, 0.738, 0.745, 0.761, 0.597
    ])

    siren_f1 = 0.867  # SIREN overall performance (0.867)
    guard_f1 = 0.834  # Baseline Guard model performance (0.834)

    # Configure Matplotlib styles to match publication quality
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 1.2

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    # Light background grid lines
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.35, color="#CCCCCC", zorder=1)

    # 1. Plot raw probe points and connecting light line
    line_color = "#C88BA8"      # Light purple connecting line
    marker_fill = "#D8A3BE"     # Marker fill color
    marker_edge = "#5C2344"     # Dark marker border

    ax.plot(
        layers, probe_f1,
        color=line_color,
        linewidth=1.2,
        alpha=0.7,
        zorder=2
    )
    ax.scatter(
        layers, probe_f1,
        s=38,
        facecolors=marker_fill,
        edgecolors=marker_edge,
        linewidth=1.0,
        alpha=0.85,
        zorder=3
    )

    # 2. Smooth trend curve (using polynomial/LOESS fit)
    smooth_color = "#992E65"    # Deep magenta/purple for smooth trend
    poly_coeffs = np.polyfit(layers, probe_f1, deg=4)
    poly_fit = np.poly1d(poly_coeffs)
    x_smooth = np.linspace(0, 35, 300)
    y_smooth = poly_fit(x_smooth)

    ax.plot(
        x_smooth, y_smooth,
        color=smooth_color,
        linewidth=3.2,
        label="Layer-wise Probes",
        zorder=4
    )

    # 3. Plot SIREN horizontal baseline (dashed blue)
    siren_color = "#1F77B4"
    ax.axhline(
        y=siren_f1,
        color=siren_color,
        linestyle="--",
        linewidth=2.5,
        label="SIREN",
        zorder=5
    )

    # 4. Plot Guard horizontal baseline (dashed orange)
    guard_color = "#FF7F0E"
    ax.axhline(
        y=guard_f1,
        color=guard_color,
        linestyle="--",
        linewidth=2.5,
        label="Guard",
        zorder=5
    )

    # Axes limits and labels formatting
    ax.set_xlim(-1, 36)
    ax.set_ylim(0.46, 1.01)
    ax.set_xticks(np.arange(0, 36, 5))
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    ax.set_xlabel("Layer Index", fontsize=18, labelpad=8)
    ax.set_ylabel("Performance", fontsize=18, labelpad=8)

    ax.tick_params(axis="both", which="major", labelsize=15, length=5, width=1.2)

    # Legend formatting
    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        fontsize=15,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.95
    )
    legend.get_frame().set_boxstyle("round,pad=0.4")
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()

    # Save artifact image
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully saved Figure 7 reproduction to: {output_path}")


if __name__ == "__main__":
    generate_and_save_figure7("figure7_reproduction.png")
