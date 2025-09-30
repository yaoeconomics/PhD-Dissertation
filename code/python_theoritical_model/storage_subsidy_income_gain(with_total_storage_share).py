# -*- coding: utf-8 -*-
"""
4x4 grid: % Mean Income Gain vs No-Storage (ZERO-GAP ONLY: μ2 = μ1)
Rows:     κ in {0.95, 0.90, 0.85, 0.80} (top -> bottom)
Columns:  Beta variance σ² in {0.05, 0.10, 0.15, 0.20} (left -> right)

Within each subplot:
- Solid line (left y-axis): % mean income gain vs no-storage (village average, heterogeneous γ)
- Dashed line (left y-axis): % mean income gain if all farmers were risk-neutral (γ=0)
- Histogram bars (right y-axis): Total Storage Share = sum_i s_i^* (0..100) under heterogeneous γ

Simulation:
- R = 400 worlds; N = 100 farmers; γ ~ Beta(1,5) scaled to [0,10]
- Strict feasibility: Beta(μ, σ²) only when σ² < μ(1-μ)
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
DPI = 300
FIG_PATH = os.path.join(target_dir, "gainpct_grid_4x4_zero_gap_total_storage_with_RN.png")

rng = np.random.default_rng(314)

# Farmers & worlds
N = 100
gammas = rng.uniform(0.0, 10.0, size=N); gammas.sort()
# gammas = 10.0 * rng.beta(1, 5, size=N); gammas.sort()  # heterogeneous risk aversion

R = 20000  # increase for smoother curves

# s*(θ1,γ) interpolation grid
theta1_grid = np.linspace(0.01, 0.99, 24)
gamma_grid  = np.linspace(0.0, 10.0, 24)

# Quadrature nodes on [0,1] for Eθ2 when solving s*
nodes, weights = leggauss(12)
x_nodes = 0.5 * (nodes + 1.0)
w_nodes = 0.5 * weights

# Rows and columns
kappa_rows = [0.95, 0.90, 0.85, 0.80]
sigma2_cols = [0.02, 0.05, 0.10, 0.15]

# Column-specific μ1 feasible base ranges (strict feasibility)
col_base_ranges = {
    0.02: (0.05, 0.95),
    0.05: (0.10, 0.90),
    0.10: (0.15, 0.85),
    0.15: (0.20, 0.80),
}

# Styles
LINE_STYLE = dict(color="#4D4D4D", marker="o", linewidth=2.0, markersize=3, label="Total gain (heterogeneous γ)")
LINE_RN_STYLE = dict(color="#1f77b4", linestyle="--", linewidth=2.0, marker=None, label="Risk-neutral gain (γ=0)")
BAR_TOTAL  = dict(color="#2CA02C", alpha=0.55, label="Total storage share")

# -----------------------------
# Helpers
# -----------------------------
def alpha_beta_from_strict(mu, sigma2, eps=1e-12):
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

def golden_max(f, a=0.0, b=1.0, tol=1e-4, max_iter=60):
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
    p1 = 1.0 / (1.0 + theta1)
    if abs(gamma) < 1e-12:  # risk-neutral shortcut
        E_inv = float(np.sum(1.0 / (1.0 + theta2_nodes) * w_nodes))
        return 1.0 if (kappa * E_inv > p1) else 0.0

    def EU_of_s(s):
        inc = (1.0 - s) * p1 + s * kappa / (1.0 + theta2_nodes)
        if abs(gamma - 1.0) < 1e-12:
            u = np.log(np.maximum(inc, 1e-12))
        else:
            u = (np.maximum(inc, 1e-12)**(1.0 - gamma) - 1.0) / (1.0 - gamma)
        return float(np.sum(u * w_nodes))

    return float(golden_max(EU_of_s, a=0.0, b=1.0, tol=1e-4, max_iter=60))

def precompute_cache(kappa, theta1_grid, gamma_grid, theta2_nodes, w_nodes):
    Z = np.zeros((len(theta1_grid), len(gamma_grid)))
    for i, t1 in enumerate(theta1_grid):
        for j, g in enumerate(gamma_grid):
            Z[i, j] = solve_s(t1, g, kappa, theta2_nodes, w_nodes)
    return Z

def interp2_vec(xgrid, ygrid, Z, x, y_arr):
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

def build_mu1_grid(sigma2):
    lo, hi = col_base_ranges[sigma2]
    n_steps = int(round((hi - lo) / 0.05)) + 1
    return np.round(np.linspace(lo, hi, n_steps), 4)

# -----------------------------
# Core computation for ZERO-GAP (μ2 = μ1)
# -----------------------------
def compute_zero_gap_gain_and_total_storage(kappa, sigma2, mu1_grid):
    """
    For fixed (kappa, sigma2) and μ2=μ1, compute:
    - total % gain vs no-storage (village mean, heterogeneous γ)
    - risk-neutral % gain vs no-storage (as-if all farmers had γ=0)
    - Total Storage Share = E_r[ ∑_i s_i^*(θ1_r, γ_i) ] in [0, 100]
    Returns (mu_vec, total_pct, rn_pct, total_storage_units).
    """
    mu_list, total_list, rn_list, storage_list = [], [], [], []

    for mu in mu1_grid:
        ab = alpha_beta_from_strict(mu, sigma2)
        if ab is None:
            continue
        a, b = ab

        # Quadrature nodes under μ2 = μ1
        theta2_nodes = beta.ppf(x_nodes, a, b)
        theta2_nodes = np.clip(theta2_nodes, 1e-9, 1 - 1e-9)

        # Precompute s*(θ1,γ) surface for this κ & μ2=μ1
        s_cache = precompute_cache(kappa, theta1_grid, gamma_grid, theta2_nodes, w_nodes)

        # Sample worlds
        if REUSE_WORLDS:
            theta1_worlds = beta.rvs(a, b, size=R, random_state=rng)
            theta2_worlds = beta.rvs(a, b, size=R, random_state=rng)
        else:
            theta1_worlds = beta.rvs(a, b, size=R)
            theta2_worlds = beta.rvs(a, b, size=R)

        # Baseline (no-storage) village mean income
        p1s = 1.0 / (1.0 + theta1_worlds)
        mean_income_no = float(np.mean(p1s))

        # With storage: compute village-average gain and total storage share
        total_delta = 0.0
        total_delta_rn = 0.0
        total_storage_units = 0.0  # sum_i s_i^* per world, then average

        for r_idx in range(R):
            t1 = theta1_worlds[r_idx]
            t2 = theta2_worlds[r_idx]
            p1 = 1.0 / (1.0 + t1)
            p2 = 1.0 / (1.0 + t2)

            # Heterogeneous γ storage vector
            s_star_vec = interp2_vec(theta1_grid, gamma_grid, s_cache, t1, gammas)
            incomes = (1.0 - s_star_vec) * p1 + s_star_vec * kappa * p2
            dy = incomes - p1
            total_delta += float(np.mean(dy))
            total_storage_units += float(np.sum(s_star_vec))  # sum_i s_i^*

            # Risk-neutral (γ=0) benchmark: same prices/world, common s*∈{0,1}
            s_star_rn = solve_s(t1, 0.0, kappa, theta2_nodes, w_nodes)  # scalar 0 or 1
            income_rn = (1.0 - s_star_rn) * p1 + s_star_rn * kappa * p2
            dy_rn = income_rn - p1
            total_delta_rn += float(dy_rn)  # identical across farmers

        total_delta /= R
        total_delta_rn /= R
        total_storage_units /= R  # expected sum in [0, N]; N=100 here

        # Convert to percentage for gains; keep storage in 0..100 units
        total_pct = 100.0 * total_delta / (mean_income_no + 1e-12)
        rn_pct = 100.0 * total_delta_rn / (mean_income_no + 1e-12)

        mu_list.append(mu)
        total_list.append(total_pct)
        rn_list.append(rn_pct)
        storage_list.append(total_storage_units)

    return (np.array(mu_list),
            np.array(total_list),
            np.array(rn_list),
            np.array(storage_list))

# -----------------------------
# Build all data then plot
# -----------------------------
def run_and_plot():
    results = {}  # (row_idx, col_idx) -> dict with arrays
    all_gain_values = []

    for r_idx, kappa in enumerate(kappa_rows):
        for c_idx, sigma2 in enumerate(sigma2_cols):
            mu1_grid = build_mu1_grid(sigma2)
            mu_vec, total_pct, rn_pct, storage_units = compute_zero_gap_gain_and_total_storage(
                kappa, sigma2, mu1_grid
            )
            results[(r_idx, c_idx)] = {
                "mu": mu_vec,
                "total": total_pct,       # heterogeneous γ (left axis)
                "rn": rn_pct,             # risk-neutral γ=0 (left axis)
                "storage": storage_units  # right axis, 0..100
            }
            if total_pct.size > 0:
                all_gain_values.append(total_pct)
            if rn_pct.size > 0:
                all_gain_values.append(rn_pct)

    # Global y-limits for the gain lines (left axis), covering both series
    if len(all_gain_values) > 0:
        all_gains = np.concatenate(all_gain_values)
        y_lo, y_hi = float(np.min(all_gains)), float(np.max(all_gains))
        if abs(y_hi - y_lo) < 1e-9:
            pad = 0.05 * (abs(y_hi) + 1.0)
            y_lim = (y_lo - pad, y_hi + pad)
        else:
            span = y_hi - y_lo
            y_lim = (y_lo - 0.07 * span, y_hi + 0.07 * span)
    else:
        y_lim = (0.0, 1.0)

    # Plotting
    fig, axes = plt.subplots(len(kappa_rows), len(sigma2_cols),
                             figsize=(14, 12), sharex=False, sharey=True)

    for r_idx, kappa in enumerate(kappa_rows):
        for c_idx, sigma2 in enumerate(sigma2_cols):
            ax = axes[r_idx, c_idx]
            data = results[(r_idx, c_idx)]
            mu = data["mu"]
            total = data["total"]
            rn = data["rn"]
            storage_units = data["storage"]

            # Left axis: gain lines
            ax.set_xlim(0.05, 0.95)
            ax.set_ylim(*y_lim)
            ax.plot(mu, total, **LINE_STYLE, zorder=3)
            ax.plot(mu, rn, **LINE_RN_STYLE, zorder=3)

            # Right axis: total storage share bars (0..100) under heterogeneous γ
            ax2 = ax.twinx()
            ax2.set_ylim(0.0, 100.0)
            barw = 0.02
            ax2.bar(mu, storage_units, width=barw, **BAR_TOTAL, zorder=2)

            # Titles & labels
            if r_idx == 0:
                ax.set_title(rf"$\sigma^2={sigma2:.2f}$", fontsize=11, pad=6)
            if c_idx == 0:
                ax.set_ylabel(rf"$\kappa={kappa:.2f}$" + "\nGain vs no-storage (%)",
                              fontsize=10)
            else:
                ax.set_ylabel("")
            ax.set_xlabel(r"$\mu_1$ (mean of $\theta_1$; $\mu_2=\mu_1$)", fontsize=9)

            # Right-axis label shown on outermost column only
            if c_idx == len(sigma2_cols) - 1:
                ax2.set_ylabel("Total storage share (0–100)", fontsize=10)
            else:
                ax2.set_yticklabels([])

            ax.grid(True, alpha=0.35, linewidth=0.7)

    # Figure legend
    legend_handles = [
        Line2D([0], [0], **{k: LINE_STYLE[k] for k in ["color", "marker", "linewidth", "markersize"]},
               label="Total gain (heterogeneous γ)"),
        Line2D([0], [0], **{k: LINE_RN_STYLE[k] for k in ["color", "linestyle", "linewidth"]},
               label="Risk-neutral gain (γ=0)"),
        Line2D([0], [0], color=BAR_TOTAL["color"], linewidth=8, alpha=BAR_TOTAL["alpha"],
               label="Total storage share (right axis)"),
    ]

    fig.suptitle(
        "% Mean Income Gain vs No-Storage (μ2 = μ1)\n"
        "Rows: κ ∈ {0.95, 0.90, 0.85, 0.80}; Cols: Var(θ) = {0.05, 0.10, 0.15, 0.20}",
        fontsize=15, y=0.95
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncols=3,
        frameon=True,
        fontsize=10,
        handlelength=2.5,
        columnspacing=1.6,
        borderpad=0.6
    )

    footer = (f"N={N} farmers; R={R} worlds; γ~Unif[0,10]; "
              "RN line treats all farmers as γ=0; "
              "Total storage share = E_r[∑ s_i^*] in 0..100.")
    fig.text(0.5, 0.018, footer, ha="center", va="center", fontsize=11.5)

    fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.88])
    plt.savefig(FIG_PATH, dpi=DPI, bbox_inches="tight")
    plt.show()
    return FIG_PATH

if __name__ == "__main__":
    out = run_and_plot()
    print(out)
