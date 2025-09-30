# -*- coding: utf-8 -*-
"""
Beta PDF & CDF 2×4 grid (top row = PDFs, bottom row = CDFs) with perceptually uniform colors
- Variances (by column): [0.02, 0.05, 0.10, 0.15]
- Means (curves within each panel): [0.2, 0.4, 0.6, 0.8]
- PDFs use a linear y-axis with a visibility-friendly cap (99th pct, bounded to [5, 60]).
- CDFs use a linear y-axis in [0, 1].
- Colors are sequential ("magma"), darker for higher μ (tougher market).
- Saves high-res PNG & vector PDF to {grandparent_dir}/model_figures.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import beta as sp_beta
from PIL import Image
from io import BytesIO

# ---------------------------
# Paths (mirrors your pattern)
# ---------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
target_dir = os.path.join(grandparent_dir, "model_figures")
os.makedirs(target_dir, exist_ok=True)

# ---------------------------
# Config
# ---------------------------
variances = [0.02, 0.05, 0.10, 0.15]
means = [0.2, 0.4, 0.6, 0.8]

dpi = 300
linewidth = 2.5
x = np.linspace(1e-6, 1 - 1e-6, 5000)

# PDF y-axis handling:
#  - Per-panel y-cap: 99th percentile of y values, bounded to [5, 60].
#  - Set FIX_GLOBAL_YCAP = True to use the SAME cap across all four PDF panels.
FIX_GLOBAL_YCAP = False

# ---------------------------
# Color strategy
# ---------------------------
def make_gradual_colors(n, cmap_name="magma", start=0.20, stop=0.92, reverse=True):
    """
    Build n evenly spaced colors from a perceptually uniform colormap.
    start/stop avoid extreme ends that can crush detail in print.
    reverse=True => larger μ gets darker (tougher market).
    """
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    positions = np.linspace(start, stop, n)
    if reverse:
        positions = positions[::-1]
    return [cmap(p) for p in positions]

# Choose "magma" (excellent in print) or "viridis" (great on screens)
colors = make_gradual_colors(len(means), cmap_name="magma", start=0.20, stop=0.92, reverse=True)

# Optional: redundant encoding for grayscale printing
line_styles = {
    0.2: (0, (1, 0)),   # solid
    0.4: (0, (4, 2)),   # long-dash
    0.6: (0, (2, 2)),   # dash
    0.8: (0, (1, 2)),   # dotted
}

# ---------------------------
# Utilities
# ---------------------------
def alpha_beta_from_mean_var(mu, var):
    """
    Compute (alpha, beta) from (mean=mu, variance=var).
    Feasible iff 0 < var < mu(1-mu). Otherwise return None.
    """
    denom = mu * (1 - mu)
    if var >= denom or var <= 0:
        return None
    t = denom / var - 1.0
    a = mu * t
    b = (1 - mu) * t
    if a <= 0 or b <= 0:
        return None
    return a, b

def compute_panel_ycap_pdf(v, means, x):
    """
    For a given variance v, compute a visibility-friendly y-axis cap for PDFs:
    99th percentile across all feasible curves, bounded to [5, 60].
    """
    yvals = []
    for mu in means:
        params = alpha_beta_from_mean_var(mu, v)
        if params is None:
            continue
        a, b = params
        y = sp_beta.pdf(x, a, b)
        if np.any(np.isfinite(y)):
            yvals.append(y[np.isfinite(y)])
    if not yvals:
        return 5.0
    y_all = np.concatenate(yvals)
    ycap = float(np.quantile(y_all, 0.99))
    return max(5.0, min(ycap, 60.0))

# Precompute PDF y-caps
pdf_caps = {v: compute_panel_ycap_pdf(v, means, x) for v in variances}
if FIX_GLOBAL_YCAP:
    global_cap = max(pdf_caps.values())  # or np.median for a tighter common scale
    pdf_caps = {v: global_cap for v in variances}

# ---------------------------
# Plot helpers
# ---------------------------
def render_pdf_column(v):
    """Render one PDF column (for variance v) and return a PIL Image."""
    fig = plt.figure(figsize=(6.2, 4.2), dpi=dpi)
    ax = plt.gca()
    plotted = 0

    for mu, color in zip(means, colors):
        params = alpha_beta_from_mean_var(mu, v)
        if params is None:
            continue
        a, b = params
        y = sp_beta.pdf(x, a, b)
        ax.plot(
            x, y,
            label=f"μ={mu:.1f}",
            color=color,
            linewidth=linewidth,
            linestyle=line_styles.get(mu, (0, (1, 0)))
        )
        plotted += 1

    ax.set_title(f"Beta PDFs (σ²={v:.2f})", fontsize=12)
    ax.set_xlabel("θ", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, pdf_caps[v])
    if plotted > 0:
        ax.legend(title="Mean μ (darker = higher μ)",
                  frameon=False, fontsize=9, title_fontsize=10, ncol=1)
    ax.text(0.01, 0.96, f"y-cap = {pdf_caps[v]:.1f}",
            transform=ax.transAxes, fontsize=8, va="top", ha="left", alpha=0.85)
    ax.text(0.99, 0.96, f"Curves: {plotted}/4",
            transform=ax.transAxes, fontsize=8, va="top", ha="right", alpha=0.85)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def render_cdf_column(v):
    """Render one CDF column (for variance v) and return a PIL Image."""
    fig = plt.figure(figsize=(6.2, 4.2), dpi=dpi)
    ax = plt.gca()
    plotted = 0

    for mu, color in zip(means, colors):
        params = alpha_beta_from_mean_var(mu, v)
        if params is None:
            continue
        a, b = params
        F = sp_beta.cdf(x, a, b)
        ax.plot(
            x, F,
            label=f"μ={mu:.1f}",
            color=color,
            linewidth=linewidth,
            linestyle=line_styles.get(mu, (0, (1, 0)))
        )
        plotted += 1

    ax.set_title(f"Beta CDFs (σ²={v:.2f})", fontsize=12)
    ax.set_xlabel("θ", fontsize=11)
    ax.set_ylabel("Cumulative probability", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if plotted > 0:
        ax.legend(title="Mean μ (darker = higher μ)",
                  frameon=False, fontsize=9, title_fontsize=10, ncol=1, loc="lower right")
    ax.text(0.99, 0.06, f"Curves: {plotted}/4",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="right", alpha=0.85)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

# ---------------------------
# Build top row (PDFs) and bottom row (CDFs)
# ---------------------------
pdf_imgs = [render_pdf_column(v) for v in variances]
cdf_imgs = [render_cdf_column(v) for v in variances]

# Concatenate horizontally into 1×4 rows
def concat_h(images):
    max_h = max(im.height for im in images)
    total_w = sum(im.width for im in images)
    row = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x_off = 0
    for im in images:
        y_off = (max_h - im.height) // 2
        row.paste(im, (x_off, y_off))
        x_off += im.width
    return row

row_pdf = concat_h(pdf_imgs)
row_cdf = concat_h(cdf_imgs)

# Stack rows vertically (2×4 grid) with a small gutter
gutter = 20  # pixels of white space between rows
W = max(row_pdf.width, row_cdf.width)
H = row_pdf.height + gutter + row_cdf.height
grid = Image.new("RGB", (W, H), (255, 255, 255))
grid.paste(row_pdf, (0, 0))
grid.paste(row_cdf, (0, row_pdf.height + gutter))

# ---------------------------
# Save outputs + preview
# ---------------------------
png_path = os.path.join(target_dir, "beta_dist_PDF_CDF_grid_2x4.png")
grid.save(png_path, format="PNG")



plt.figure(figsize=(16, 8), dpi=dpi)
plt.imshow(grid)
plt.axis("off")
plt.tight_layout()
plt.show()

print("Saved:", png_path)
