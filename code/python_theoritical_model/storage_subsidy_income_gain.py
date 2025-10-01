# -*- coding: utf-8 -*-
"""
4x3 grid: % Mean Income Gain vs No-Storage
Rows:     κ in {0.95, 0.90, 0.85, 0.80} (top -> bottom)
Columns:  Beta variance σ² in {0.10, 0.15, 0.20} (left -> right)

Within each subplot:
- y-axis: % mean income gain vs no-storage
- x-axis: μ1 grid (mean of θ1), step 0.05
- curves: gaps g ∈ {0.00, 0.05, 0.15}, interpreted as μ2 = μ1 − g

Feasible μ1 ranges by column (fixed variance feasibility you specified):
- Col 1 (σ²=0.10): base μ1 ∈ [0.20, 0.80]; gaps use [0.30, 0.80]
- Col 2 (σ²=0.15): base μ1 ∈ [0.20, 0.80]; gaps use [0.35, 0.80]
- Col 3 (σ²=0.20): base μ1 ∈ [0.30, 0.70]; gaps use [0.45, 0.70]

Runtime-friendly try: R = 200 worlds; N = 100 farmers; γ ~ Uniform(0, 10).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import beta
from numpy.polynomial.legendre import leggauss

# -----------------------------
# Paths (robust to interactive)
# -----------------------------
try:
    CURRENT_FILE = os.path.abspath(__file__)
    current_dir = os.path.dirname(CURRENT_FILE)
except NameError:
    current_dir = os.getcwd()

parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
target_dir = os.path.join(grandparent_dir, "model_figures")
os.makedirs(target_dir, exist_ok=True)

# -----------------------------
# Global settings
# -----------------------------
REUSE_WORLDS = True
STORE_THRESHOLD = 0.01      # not directly used here, kept for consistency
DPI = 300
FIG_PATH = os.path.join(target_dir, "gainpct_grid_4x4.png")

rng = np.random.default_rng(314)

# Farmers & worlds
N = 100
gammas = rng.uniform(0.0, 10.0, size=N); gammas.sort()
# gammas = 10.0 * rng.beta(1, 5, size=N); gammas.sort()

R = 20000  # <-- The number of Worlds

# s*(θ1,γ) interpolation grid
theta1_grid = np.linspace(0.001, 0.999, 30)
gamma_grid  = np.linspace(0.0, 10.0, 30)

# Quadrature nodes on [0,1] for Eθ2 when solving s*
nodes, weights = leggauss(12)
x_nodes = 0.5 * (nodes + 1.0)
w_nodes = 0.5 * weights

# Rows and columns
kappa_rows = [0.95, 0.90, 0.85, 0.80]
sigma2_cols = [0.02, 0.05, 0.10, 0.15]
gap_list = [0.00, 0.05, 0.15]  # curves

# Column-specific μ1 base ranges and narrower ranges for g>0
col_base_ranges = {
    0.02: (0.05, 0.95),
    0.05: (0.10, 0.90),
    0.10: (0.15, 0.85),
    0.15: (0.20, 0.80),
}

col_gap005_ranges = {  # for g in {0.05}
    0.02: (0.10, 0.95),
    0.05: (0.15, 0.90),
    0.10: (0.20, 0.85),
    0.15: (0.25, 0.80),
}

col_gap015_ranges = {  # for g in {0.15}
    0.02: (0.20, 0.95),
    0.05: (0.25, 0.90),
    0.10: (0.30, 0.85),
    0.15: (0.35, 0.80),
}

# Curve aesthetics (meaningful colors & shapes)
CURVE_STYLE = {
    0.00: dict(color="#4D4D4D", marker="o", linewidth=2.0, markersize=3, label=r"$\mu_2=\mu_1$"),
    0.05: dict(color="#2CA02C", marker="s", linewidth=2.0, markersize=3, label=r"$\mu_2=\mu_1-0.05$"),
    0.15: dict(color="#006D2C", marker="^", linewidth=2.0, markersize=3, label=r"$\mu_2=\mu_1-0.15$"),
}

# -----------------------------
# Utility and helpers
# -----------------------------
def crra_u(pi, gamma):
    pi = np.maximum(pi, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(pi)
    return (pi**(1-gamma) - 1) / (1-gamma)

def alpha_beta_from_strict(mu, sigma2, eps=1e-12):
    """
    Strict feasibility: return (alpha, beta) only if sigma2 < mu(1-mu).
    Otherwise return None. No auto-shrinking of variance.
    """
    mu = float(mu)
    if not (eps < mu < 1.0 - eps):
        return None
    max_var = mu * (1.0 - mu)
    if sigma2 >= max_var - 1e-12:
        return None
    factor = mu * (1.0 - mu) / sigma2 - 1.0
    a, b = mu * factor, (1.0 - mu) * factor
    if a <= 0.0 or b <= 0.0:
        return None
    return a, b

def golden_max(f, a=0.0, b=1.0, tol=5e-5, max_iter=90):
    gr = (np.sqrt(5.0) + 1.0) / 2.0
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / gr
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / gr
            fd = f(d)
    return 0.5 * (a + b)

def solve_s(theta1, gamma, kappa, theta2_nodes, w_nodes):
    """
    Return s* that maximizes expected utility given θ1 (current), γ, κ,
    and the expectation over θ2 via quadrature nodes.
    """
    p1 = 1.0 / (1.0 + theta1)
    if abs(gamma) < 1e-12:  # risk-neutral shortcut
        E_inv = float(np.sum(1.0 / (1.0 + theta2_nodes) * w_nodes))
        return 1.0 if (kappa * E_inv > p1) else 0.0

    def EU_of_s(s):
        inc = (1 - s) * p1 + s * kappa / (1.0 + theta2_nodes)
        if abs(gamma - 1.0) < 1e-12:
            u = np.log(np.maximum(inc, 1e-12))
        else:
            u = (np.maximum(inc, 1e-12)**(1-gamma) - 1) / (1-gamma)
        return float(np.sum(u * w_nodes))

    return float(golden_max(EU_of_s, a=0.0, b=1.0, tol=5e-5, max_iter=90))

def precompute_cache(kappa, theta1_grid, gamma_grid, theta2_nodes, w_nodes):
    """
    Precompute s*(θ1, γ) for a single κ over the (θ1, γ) grid.
    """
    Z = np.zeros((len(theta1_grid), len(gamma_grid)))
    for i, t1 in enumerate(theta1_grid):
        for j, g in enumerate(gamma_grid):
            Z[i, j] = solve_s(t1, g, kappa, theta2_nodes, w_nodes)
    return Z

def interp2_vec(xgrid, ygrid, Z, x, y_arr):
    """
    Bilinear interpolation at a single x with a vector of y's.
    Z shape: (len(xgrid), len(ygrid)).
    """
    xi = np.searchsorted(xgrid, x) - 1
    xi = np.clip(xi, 0, len(xgrid) - 2)
    yi = np.searchsorted(ygrid, y_arr) - 1
    yi = np.clip(yi, 0, len(ygrid) - 2)

    x0, x1 = xgrid[xi], xgrid[xi + 1]
    y0, y1 = ygrid[yi], ygrid[yi + 1]
    tx = (x - x0) / (x1 - x0 + 1e-12)
    ty = (y_arr - y0) / (y1 - y0 + 1e-12)

    z00 = Z[xi, yi]
    z01 = Z[xi, yi + 1]
    z10 = Z[xi + 1, yi]
    z11 = Z[xi + 1, yi + 1]

    return (1 - tx) * (1 - ty) * z00 + (1 - tx) * ty * z01 + tx * (1 - ty) * z10 + tx * ty * z11

# -----------------------------
# μ1 grids per column & gap
# -----------------------------
def build_mu1_grid(sigma2, gap):
    if gap == 0.0:
        lo, hi = col_base_ranges[sigma2]
    elif gap == 0.05:
        lo, hi = col_gap005_ranges[sigma2]
    elif gap == 0.15:
        lo, hi = col_gap015_ranges[sigma2]
    else:
        raise ValueError(f"Unsupported gap value: {gap}")
    n_steps = int(round((hi - lo) / 0.05)) + 1
    grid = np.round(np.linspace(lo, hi, n_steps), 4)
    return grid


# -----------------------------
# Core computation for one (κ, σ²) subplot
# -----------------------------
def compute_curve_for_gap(kappa, sigma2, gap, mu1_grid):
    """
    For a fixed (kappa, sigma2, gap), compute % gain vs no-storage
    across the provided feasible μ1 grid. Enforce strict feasibility
    for both μ1 and μ2 = μ1 - gap. Returns (mu1_vec, pct_gain_vec).
    """
    mu1_list = []
    pct_list = []

    for mu1 in mu1_grid:
        mu2 = mu1 - gap
        # Strict feasibility checks for both μ1 and μ2 at fixed σ²
        ab1 = alpha_beta_from_strict(mu1, sigma2)
        ab2 = alpha_beta_from_strict(mu2, sigma2)
        if ab1 is None or ab2 is None:
            continue
        a1, b1 = ab1
        a2, b2 = ab2

        # Quadrature nodes (θ2) for s* expectation
        theta2_nodes = beta.ppf(x_nodes, a2, b2)
        theta2_nodes = np.clip(theta2_nodes, 1e-9, 1 - 1e-9)

        # Precompute s*(θ1,γ) surface for this κ & μ2
        s_cache = precompute_cache(kappa, theta1_grid, gamma_grid, theta2_nodes, w_nodes)

        # Sample worlds
        if REUSE_WORLDS:
            theta1_worlds = beta.rvs(a1, b1, size=R, random_state=rng)
            theta2_worlds = beta.rvs(a2, b2, size=R, random_state=rng)
        else:
            theta1_worlds = beta.rvs(a1, b1, size=R)
            theta2_worlds = beta.rvs(a2, b2, size=R)

        # No-storage benchmark: village mean income
        p1s = 1.0 / (1.0 + theta1_worlds)
        mean_income_no = float(np.mean(p1s))

        # With storage: evaluate s* for each world and farmer γ_i
        Y_bar = 0.0
        for r_idx in range(R):
            t1 = theta1_worlds[r_idx]
            t2 = theta2_worlds[r_idx]
            p1 = 1.0 / (1.0 + t1)
            p2 = 1.0 / (1.0 + t2)

            s_star_vec = interp2_vec(theta1_grid, gamma_grid, s_cache, t1, gammas)
            incomes = (1.0 - s_star_vec) * p1 + s_star_vec * kappa * p2
            Y_bar += float(np.mean(incomes))
        mean_income_with = Y_bar / R

        pct_gain = 100.0 * (mean_income_with - mean_income_no) / (mean_income_no + 1e-12)
        mu1_list.append(mu1)
        pct_list.append(pct_gain)

    return np.array(mu1_list), np.array(pct_list)

# -----------------------------
# Build all data first (for global y-limits), then plot
# -----------------------------
def run_and_plot():
    # Precompute all curves to determine global y-range
    results = {}  # (row_idx, col_idx) -> {gap: (mu1, pct)}
    all_pct_values = []

    for r_idx, kappa in enumerate(kappa_rows):
        for c_idx, sigma2 in enumerate(sigma2_cols):
            col_dict = {}
            for g in gap_list:
                mu1_grid = build_mu1_grid(sigma2, g)
                mu1_vec, pct_vec = compute_curve_for_gap(kappa, sigma2, g, mu1_grid)
                col_dict[g] = (mu1_vec, pct_vec)
                if pct_vec.size > 0:
                    all_pct_values.append(pct_vec)
            results[(r_idx, c_idx)] = col_dict

    # Global y-limits with gentle padding
    if len(all_pct_values) > 0:
        all_pcts = np.concatenate(all_pct_values)
        y_lo, y_hi = float(np.min(all_pcts)), float(np.max(all_pcts))
        if abs(y_hi - y_lo) < 1e-9:
            pad = 0.05 * (abs(y_hi) + 1.0)
            y_lim = (y_lo - pad, y_hi + pad)
        else:
            span = y_hi - y_lo
            y_lim = (y_lo - 0.05 * span, y_hi + 0.05 * span)
    else:
        y_lim = (0.0, 1.0)  # fallback

    # Plotting
    fig, axes = plt.subplots(len(kappa_rows), len(sigma2_cols),
                             figsize=(12, 12), sharex=False, sharey=True)

    for r_idx, kappa in enumerate(kappa_rows):
        for c_idx, sigma2 in enumerate(sigma2_cols):
            ax = axes[r_idx, c_idx]
            col_data = results[(r_idx, c_idx)]

            # X ranges per column
            ax.set_xlim(0.05, 0.95) # Force all x-axes to show 0–1, for consistent appearance
            ax.set_ylim(*y_lim)

            # Plot curves with meaningful colors/markers
            for g in gap_list:
                mu1_vec, pct_vec = col_data[g]
                if mu1_vec.size == 0:
                    continue
                style = CURVE_STYLE[g]
                ax.plot(
                    mu1_vec, pct_vec,
                    marker=style["marker"],
                    linewidth=style["linewidth"],
                    markersize=style["markersize"],
                    color=style["color"],
                    label=style["label"],
                    alpha=0.95,
                )

            # Titles & labels
            if r_idx == 0:
                ax.set_title(rf"$\sigma^2={sigma2:.2f}$", fontsize=11, pad=6)
            if c_idx == 0:
                ax.set_ylabel(rf"$\kappa={kappa:.2f}$" + "\n" + "Gain vs no-storage (%)",
                              fontsize=10)
            else:
                ax.set_ylabel("")

            ax.set_xlabel(r"$\mu_1$ (mean of $\theta_1$)", fontsize=9)
            ax.grid(True, alpha=0.35, linewidth=0.7)

    # Clean, non-overlapping legend: figure-level, between title and panels
    legend_handles = [
        Line2D([0], [0], color=CURVE_STYLE[0.00]["color"], marker=CURVE_STYLE[0.00]["marker"],
               linewidth=CURVE_STYLE[0.00]["linewidth"], markersize=CURVE_STYLE[0.00]["markersize"],
               label=CURVE_STYLE[0.00]["label"]),
        Line2D([0], [0], color=CURVE_STYLE[0.05]["color"], marker=CURVE_STYLE[0.05]["marker"],
               linewidth=CURVE_STYLE[0.05]["linewidth"], markersize=CURVE_STYLE[0.05]["markersize"],
               label=CURVE_STYLE[0.05]["label"]),
        Line2D([0], [0], color=CURVE_STYLE[0.15]["color"], marker=CURVE_STYLE[0.15]["marker"],
               linewidth=CURVE_STYLE[0.15]["linewidth"], markersize=CURVE_STYLE[0.15]["markersize"],
               label=CURVE_STYLE[0.15]["label"]),
    ]

    # Global title (slightly lower) and legend (just below it)
    fig.suptitle(
        "% Mean Income Gain vs No-Storage\n"
        "Rows: κ ∈ {0.95, 0.90, 0.85, 0.80}; "
        "Cols: Var(θ) = {0.02, 0.05, 0.10, 0.15}",
        fontsize=15, y=0.95
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),  # below suptitle, above panels
        ncols=3,
        frameon=True,
        fontsize=10,
        handlelength=2.5,
        columnspacing=1.6,
        borderpad=0.6
    )

    # Footer metadata
    footer = (f"N={N} farmers; R= {R} worlds; γ~Uniform[0,10]; "
              "s* via golden-section; Beta(μ,σ²) strict feasibility; "
              "markers aid B/W reproduction.")
    fig.text(0.5, 0.018, footer, ha="center", va="center", fontsize=12)

    # Tight layout: leave space for title & legend
    fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.88])
    plt.savefig(FIG_PATH, dpi=DPI, bbox_inches="tight")
    plt.show()
    return FIG_PATH

if __name__ == "__main__":
    out = run_and_plot()
    print(out)
