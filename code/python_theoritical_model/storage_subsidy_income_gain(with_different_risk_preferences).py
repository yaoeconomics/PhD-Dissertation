# -*- coding: utf-8 -*-
"""
4x4 grid: % Mean Income Gain vs No-Storage (ZERO-GAP ONLY: μ2 = μ1)
Rows:     κ in {0.95, 0.90, 0.85, 0.80} (top -> bottom)
Columns:  Beta variance σ² in {0.05, 0.10, 0.15, 0.20} (left -> right)

Within each subplot (left y-axis only):
- Black solid:   % mean income gain (village avg) with γ_i ~ Unif[0,10]
- Blue dashed:   % mean income gain (village avg) with γ_i ~ Unif[0,5]
- Light-blue -. : % mean income gain if all farmers were risk-neutral (γ=0)

Simulation:
- R = 20000 worlds; N = 100 farmers
- Strict feasibility: Beta(μ, σ²) only when σ² < μ(1-μ)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import beta
from numpy.polynomial.legendre import leggauss


plt.rcParams.update({
    "font.size": 14,            # base font size
    "axes.titlesize": 14,       # subplot titles
    "axes.labelsize": 14,       # x and y labels
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 14      # super-title
})



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
FIG_PATH = os.path.join(
    target_dir,
    "gainpct_grid_4x4_zero_gap_three_curves.png"
)

rng = np.random.default_rng(314)

# Farmers & worlds
N = 100
# Two heterogeneous γ draws (fixed across panels for comparability)
gammas_10 = rng.uniform(0.0, 10.0, size=N); gammas_10.sort()
gammas_5  = rng.uniform(0.0,  5.0, size=N); gammas_5.sort()

R = 20000  # increase for smoother curves

# s*(θ1,γ) interpolation grid
theta1_grid = np.linspace(0.01, 0.99, 24)
gamma_grid  = np.linspace(0.0, 10.0, 24)   # covers both Unif[0,10] and Unif[0,5]

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
LINE_HET10 = dict(color="#000000", linestyle="-",  marker="o",  linewidth=2.0, markersize=3,
                  label="Heterogeneous γ ~ Unif[0,10]")
LINE_HET5  = dict(color="#1f77b4", linestyle="--", marker=None, linewidth=2.0,
                  label="Heterogeneous γ ~ Unif[0,5]")
LINE_RN    = dict(color="#6baed6", linestyle="-.", marker=None, linewidth=2.0,
                  label="Risk-neutral (γ=0)")

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
def compute_zero_gap_gains_three_series(kappa, sigma2, mu1_grid, gammas_hi, gammas_lo):
    """
    For fixed (kappa, sigma2) and μ2=μ1, compute three gain series:
    - total_pct_hi : heterogeneous γ ~ Unif[0,10]
    - total_pct_lo : heterogeneous γ ~ Unif[0,5]
    - rn_pct       : all farmers risk-neutral (γ=0)
    Returns (mu_vec, total_pct_hi, total_pct_lo, rn_pct).
    """
    mu_list, hi_list, lo_list, rn_list = [], [], [], []

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

        # Accumulators
        total_delta_hi = 0.0  # γ~Unif[0,10]
        total_delta_lo = 0.0  # γ~Unif[0,5]
        total_delta_rn = 0.0  # γ=0

        for r_idx in range(R):
            t1 = theta1_worlds[r_idx]
            t2 = theta2_worlds[r_idx]
            p1 = 1.0 / (1.0 + t1)
            p2 = 1.0 / (1.0 + t2)

            # Heterogeneous γ ~ Unif[0,10]
            s_star_hi = interp2_vec(theta1_grid, gamma_grid, s_cache, t1, gammas_hi)
            incomes_hi = (1.0 - s_star_hi) * p1 + s_star_hi * kappa * p2
            total_delta_hi += float(np.mean(incomes_hi - p1))

            # Heterogeneous γ ~ Unif[0,5]
            s_star_lo = interp2_vec(theta1_grid, gamma_grid, s_cache, t1, gammas_lo)
            incomes_lo = (1.0 - s_star_lo) * p1 + s_star_lo * kappa * p2
            total_delta_lo += float(np.mean(incomes_lo - p1))

            # Risk-neutral (γ=0): common s*∈{0,1}
            s_star_rn = solve_s(t1, 0.0, kappa, theta2_nodes, w_nodes)  # scalar 0 or 1
            income_rn = (1.0 - s_star_rn) * p1 + s_star_rn * kappa * p2
            total_delta_rn += float(income_rn - p1)

        # Average across worlds
        total_delta_hi /= R
        total_delta_lo /= R
        total_delta_rn /= R

        # Convert to percentage gains
        pct_hi = 100.0 * total_delta_hi / (mean_income_no + 1e-12)
        pct_lo = 100.0 * total_delta_lo / (mean_income_no + 1e-12)
        pct_rn = 100.0 * total_delta_rn / (mean_income_no + 1e-12)

        mu_list.append(mu)
        hi_list.append(pct_hi)
        lo_list.append(pct_lo)
        rn_list.append(pct_rn)

    return (np.array(mu_list),
            np.array(hi_list),
            np.array(lo_list),
            np.array(rn_list))

# -----------------------------
# Build all data then plot
# -----------------------------
def run_and_plot():
    results = {}  # (row_idx, col_idx) -> dict with arrays
    all_gain_values = []

    for r_idx, kappa in enumerate(kappa_rows):
        for c_idx, sigma2 in enumerate(sigma2_cols):
            mu1_grid = build_mu1_grid(sigma2)
            mu_vec, pct_hi, pct_lo, pct_rn = compute_zero_gap_gains_three_series(
                kappa, sigma2, mu1_grid, gammas_10, gammas_5
            )
            results[(r_idx, c_idx)] = {
                "mu": mu_vec,
                "het10": pct_hi,    # black solid
                "het5":  pct_lo,    # blue dashed
                "rn":    pct_rn,    # light-blue dash-dot
            }
            for arr in (pct_hi, pct_lo, pct_rn):
                if arr.size > 0:
                    all_gain_values.append(arr)

    # Global y-limits for the gain lines (left axis), covering all series
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
            het10 = data["het10"]
            het5  = data["het5"]
            rn    = data["rn"]

            ax.set_xlim(0.05, 0.95)
            ax.set_ylim(*y_lim)

            # Three curves
            ax.plot(mu, het10, **LINE_HET10, zorder=3)
            ax.plot(mu, het5,  **LINE_HET5,  zorder=3)
            ax.plot(mu, rn,    **LINE_RN,    zorder=3)

            # Titles & labels
            if r_idx == 0:
                ax.set_title(rf"$\sigma^2={sigma2:.2f}$", fontsize=14, pad=6)
            if c_idx == 0:
                ax.set_ylabel(rf"$\kappa={kappa:.2f}$" + "\nGain vs no-storage (%)",
                              fontsize=14)
            else:
                ax.set_ylabel("")
            ax.set_xlabel(r"$\mu_1=\mu_2$", fontsize=14)

            ax.grid(True, alpha=0.35, linewidth=0.7)

    # Figure legend
    legend_handles = [
        Line2D([0], [0],
               **{k: LINE_HET10[k] for k in ["color", "linestyle", "marker", "linewidth", "markersize"]}),
        Line2D([0], [0],
               **{k: LINE_HET5[k] for k in ["color", "linestyle", "linewidth"]}),
        Line2D([0], [0],
               **{k: LINE_RN[k] for k in ["color", "linestyle", "linewidth"]}),
    ]

    fig.suptitle(
        "% Mean Income Gain vs No-Storage (μ2 = μ1)\n"
        "Rows: κ ∈ {0.95, 0.90, 0.85, 0.80}; Cols: Var(θ) = {0.05, 0.10, 0.15, 0.20}",
        fontsize=15, y=0.95
    )
    fig.legend(
        handles=legend_handles,
        labels=[LINE_HET10["label"], LINE_HET5["label"], LINE_RN["label"]],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncols=3,
        frameon=True,
        fontsize=14,
        handlelength=2.5,
        columnspacing=1.6,
        borderpad=0.6
    )

    footer = (f"N={N} farmers; R={R} worlds; "
              "Heterogeneous curves use fixed γ samples: Unif[0,10] and Unif[0,5]; "
              "RN curve treats all farmers as γ=0.")
    fig.text(0.5, 0.018, footer, ha="center", va="center", fontsize=12)

    fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.88])
    plt.savefig(FIG_PATH, dpi=DPI, bbox_inches="tight")
    plt.show()
    return FIG_PATH

if __name__ == "__main__":
    out = run_and_plot()
    print(out)
