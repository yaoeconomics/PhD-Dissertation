# -*- coding: utf-8 -*-
"""
3x4 grid: Share of *new adopters* (in %), using **Logit-Normal** beliefs for θ∈(0,1).

Rows:     κ changes in {0.80→0.85, 0.80→0.90, 0.80→0.95} (top → bottom)
Columns:  Target variance σ² in {0.05, 0.10, 0.15, 0.20} (left → right)

Within each subplot:
- y-axis: Share of *new adopters* (%): farmers with s*≤threshold at κ=0.80 who
  switch to s*>threshold at κ=κ_new.
- x-axis: μ1 grid (mean of θ1), step 0.05
- curves: gaps g ∈ {0.00, 0.05, 0.15}, interpreted as μ2 = μ1 − g

Logit-Normal parameterization:
- θ = logistic(Z) with Z ~ N(μ0, σ0²). For given targets (μ, σ²) on θ, we
  numerically solve for (μ0, σ0) so that E[θ]=μ and Var(θ)=σ² via Gauss–Hermite quadrature.
- Feasibility: 0 < μ < 1 and 0 < σ² < 1/4 (Bernoulli bound).

Runtime-friendly defaults: R = 200 worlds; N = 100 farmers; γ ~ Uniform(0,10).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm
from scipy.optimize import root

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
STORE_THRESHOLD = 0.01      # defines “adoption”: s* > STORE_THRESHOLD
DPI = 300
FIG_PATH = os.path.join(target_dir, "new_adopters_logitnormal_3x4.png")

rng = np.random.default_rng(314)

# Farmers & worlds
N = 100
gammas = rng.uniform(0.0, 10.0, size=N); gammas.sort()
R = 20000  # runtime-friendly

# s*(θ1,γ) interpolation grid
theta1_grid = np.linspace(0.001, 0.999, 30)
gamma_grid  = np.linspace(0.0, 10.0, 30)

# Quadrature nodes on [0,1] for Eθ2 when solving s*
# (We still integrate expectation over θ2 using Legendre on [0,1])
nodes, weights = leggauss(12)
x_nodes = 0.5 * (nodes + 1.0)
w_nodes = 0.5 * weights

# Gauss–Hermite nodes for Logit-Normal moments on Z∈(-∞,∞)
H_N = 60  # accuracy for moment-matching
hx, hw = hermgauss(H_N)  # nodes/weights for ∫ f(x) e^{-x^2} dx

# Rows and columns
kappa_rows = [(0.80, 0.85), (0.80, 0.90), (0.80, 0.95)]  # (κ_base, κ_new)
sigma2_cols = [0.05, 0.10, 0.15, 0.20]
gap_list = [0.00, 0.05, 0.15]  # curves

# Column-specific μ1 base ranges and narrower ranges for g>0
col_base_ranges = {
    0.05: (0.10, 0.90),
    0.10: (0.15, 0.85),
    0.15: (0.20, 0.80),
    0.20: (0.30, 0.70),
}

col_gap005_ranges = {  # for g in {0.05}
    0.05: (0.15, 0.90),
    0.10: (0.20, 0.85),
    0.15: (0.25, 0.80),
    0.20: (0.35, 0.70),
}

col_gap015_ranges = {  # for g in {0.15}
    0.05: (0.25, 0.90),
    0.10: (0.30, 0.85),
    0.15: (0.35, 0.80),
    0.20: (0.45, 0.70),
}

# Curve aesthetics
CURVE_STYLE = {
    0.00: dict(color="#4D4D4D", marker="o", linewidth=2.0, markersize=3, label=r"$\mu_2=\mu_1$"),
    0.05: dict(color="#2CA02C", marker="s", linewidth=2.0, markersize=3, label=r"$\mu_2=\mu_1-0.05$"),
    0.15: dict(color="#006D2C", marker="^", linewidth=2.0, markersize=3, label=r"$\mu_2=\mu_1-0.15$"),
}

# -----------------------------
# Utility and helpers
# -----------------------------

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def crra_u(pi, gamma):
    pi = np.maximum(pi, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(pi)
    return (pi**(1-gamma) - 1) / (1-gamma)


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

# -------- Logit-Normal moment-matching (with caching) ---------

def logitnormal_moments(mu0, sigma0):
    """Return mean and variance of θ=logistic(Z), Z~N(mu0, sigma0^2),
    computed via Gauss–Hermite: E[g(Z)] = 1/√π Σ w_i g(mu0+σ0√2 x_i)."""
    z = mu0 + sigma0 * np.sqrt(2.0) * hx
    th = logistic(z)
    m = (hw @ th) / np.sqrt(np.pi)
    v = (hw @ (th**2)) / np.sqrt(np.pi) - m**2
    return float(m), float(v)


def solve_logitnormal_params(mu_target, var_target):
    """Solve for (mu0, sigma0) such that E[θ]=mu_target and Var(θ)=var_target.
    Returns (mu0, sigma0) or None."""
    mu_target = float(mu_target)
    var_target = float(var_target)
    if not (0.0 < mu_target < 1.0):
        return None
    if not (0.0 < var_target < 0.25 + 1e-12):
        return None

    # Initial guess: map mean through logit; moderate dispersion
    eps = 1e-6
    mu0_guess = np.log(mu_target/(1.0-mu_target + eps) + eps)
    sigma0_guess = 0.6
    x0 = np.array([mu0_guess, sigma0_guess])

    def F(x):
        mu0, sigma0 = x[0], max(x[1], 1e-6)
        m, v = logitnormal_moments(mu0, sigma0)
        return np.array([m - mu_target, v - var_target])

    sol = root(F, x0, method="hybr", tol=1e-10)
    if not sol.success:
        # try a couple of alternative spreads
        for s in (0.3, 0.9, 1.2):
            sol = root(F, np.array([mu0_guess, s]), method="hybr", tol=1e-10)
            if sol.success:
                break
    if not sol.success:
        return None

    mu0, sigma0 = sol.x[0], max(sol.x[1], 1e-6)
    m, v = logitnormal_moments(mu0, sigma0)
    if (abs(m - mu_target) > 1e-6) or (abs(v - var_target) > 1e-6):
        return None
    return mu0, sigma0

# Caches to avoid recomputation
LOGIT_CACHE = {}  # key: (mu,var) → (mu0,sigma0)
SCACHE = {}       # key: (kappa, mu0_θ2, sigma0_θ2, len(theta1_grid), len(gamma_grid)) → Z array


def logit_params_from_moment_match(mu, sigma2):
    key = (round(float(mu), 6), round(float(sigma2), 6))
    if key in LOGIT_CACHE:
        return LOGIT_CACHE[key]
    out = solve_logitnormal_params(mu, sigma2)
    if out is None:
        return None
    LOGIT_CACHE[key] = out
    return out


def logit_feasible(mu, sigma2):
    return (0.0 < mu < 1.0) and (0.0 < sigma2 <= 0.25)

# -----------------------------------------------------------------

def solve_s(theta1, gamma, kappa, theta2_nodes, w_nodes):
    """Return s* that maximizes expected utility given θ1 (current), γ, κ, and Eθ2."""
    p1 = 1.0 / (1.0 + theta1)
    if abs(gamma) < 1e-12:  # risk-neutral shortcut
        E_inv = float(np.sum(1.0 / (1.0 + theta2_nodes) * w_nodes))
        return 1.0 if (kappa * E_inv > p1) else 0.0

    def EU_of_s(s):
        inc = (1 - s) * p1 + s * kappa / (1.0 + theta2_nodes)
        u = np.log(np.maximum(inc, 1e-12)) if abs(gamma - 1.0) < 1e-12 else (np.maximum(inc, 1e-12)**(1-gamma) - 1) / (1-gamma)
        return float(np.sum(u * w_nodes))

    return float(golden_max(EU_of_s, a=0.0, b=1.0, tol=5e-5, max_iter=90))


def precompute_cache(kappa, theta1_grid, gamma_grid, theta2_nodes, w_nodes):
    """Precompute s*(θ1, γ) for a single κ over the (θ1, γ) grid."""
    Z = np.zeros((len(theta1_grid), len(gamma_grid)))
    for i, t1 in enumerate(theta1_grid):
        for j, g in enumerate(gamma_grid):
            Z[i, j] = solve_s(t1, g, kappa, theta2_nodes, w_nodes)
    return Z


def interp2_vec(xgrid, ygrid, Z, x, y_arr):
    """Bilinear interpolation at a single x with a vector of y's. Z shape: (len(xgrid), len(ygrid))."""
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
# Core computation for one (κ_base→κ_new, σ²) subplot
# -----------------------------

def compute_curve_new_adopters(kappa_base, kappa_new, sigma2, gap, mu1_grid):
    """
    For a fixed (kappa_base→kappa_new, sigma2, gap), compute % share of *new adopters*
    across the provided feasible μ1 grid. "New adopter" means: s*(κ_base) ≤ threshold
    and s*(κ_new) > threshold. Uses Logit-Normal for θ1, θ2 with exact moment-matching.
    Returns (mu1_vec, pct_share_vec).
    """
    mu1_list, pct_list = [], []

    for mu1 in mu1_grid:
        mu2 = mu1 - gap
        if not logit_feasible(mu1, sigma2) or not logit_feasible(mu2, sigma2):
            continue

        pars1 = logit_params_from_moment_match(mu1, sigma2)
        pars2 = logit_params_from_moment_match(mu2, sigma2)
        if (pars1 is None) or (pars2 is None):
            continue
        mu01, sig01 = pars1
        mu02, sig02 = pars2

        # Quadrature nodes (θ2) for s* expectation: θ2_ppf(u) = logistic(μ02 + σ02 Φ^{-1}(u))
        zq = norm.ppf(x_nodes)
        theta2_nodes = logistic(mu02 + sig02 * zq)
        theta2_nodes = np.clip(theta2_nodes, 1e-9, 1 - 1e-9)

        # Precompute s*(θ1,γ) surfaces for κ_base and κ_new with caching on θ2 params
        scache_key_base = (kappa_base, round(mu02,6), round(sig02,6), len(theta1_grid), len(gamma_grid))
        scache_key_new  = (kappa_new,  round(mu02,6), round(sig02,6), len(theta1_grid), len(gamma_grid))

        if scache_key_base in SCACHE:
            s_cache_base = SCACHE[scache_key_base]
        else:
            s_cache_base = precompute_cache(kappa_base, theta1_grid, gamma_grid, theta2_nodes, w_nodes)
            SCACHE[scache_key_base] = s_cache_base

        if scache_key_new in SCACHE:
            s_cache_new = SCACHE[scache_key_new]
        else:
            s_cache_new = precompute_cache(kappa_new, theta1_grid, gamma_grid, theta2_nodes, w_nodes)
            SCACHE[scache_key_new] = s_cache_new

        # Sample worlds for θ1: θ1 = logistic(μ01 + σ01 * Z)
        if REUSE_WORLDS:
            z1 = rng.normal(loc=0.0, scale=1.0, size=R)
        else:
            z1 = np.random.normal(size=R)
        theta1_worlds = logistic(mu01 + sig01 * z1)

        # Compute share of new adopters per world, average across worlds
        share_sum = 0.0
        for r_idx in range(R):
            t1 = theta1_worlds[r_idx]
            s_base_vec = interp2_vec(theta1_grid, gamma_grid, s_cache_base, t1, gammas)
            s_new_vec  = interp2_vec(theta1_grid, gamma_grid, s_cache_new,  t1, gammas)
            adopters = (s_base_vec <= STORE_THRESHOLD) & (s_new_vec > STORE_THRESHOLD)
            share_sum += float(np.mean(adopters))

        share_pct = 100.0 * share_sum / R
        mu1_list.append(mu1)
        pct_list.append(share_pct)

    return np.array(mu1_list), np.array(pct_list)

# -----------------------------
# Build all data first (for global y-limits), then plot
# -----------------------------

def run_and_plot():
    # Precompute all curves to determine global y-range
    results = {}  # (row_idx, col_idx) → {gap: (mu1, pct)}
    all_pct_values = []

    for r_idx, (k0, k1) in enumerate(kappa_rows):
        for c_idx, sigma2 in enumerate(sigma2_cols):
            col_dict = {}
            for g in gap_list:
                mu1_grid = build_mu1_grid(sigma2, g)
                mu1_vec, pct_vec = compute_curve_new_adopters(k0, k1, sigma2, g, mu1_grid)
                col_dict[g] = (mu1_vec, pct_vec)
                if pct_vec.size > 0:
                    all_pct_values.append(pct_vec)
            results[(r_idx, c_idx)] = col_dict

    # Global y-limits (0–100%) with padding
    if len(all_pct_values) > 0:
        all_pcts = np.concatenate(all_pct_values)
        y_lo, y_hi = max(0.0, float(np.min(all_pcts))), min(100.0, float(np.max(all_pcts)))
        span = max(1e-6, y_hi - y_lo)
        y_lim = (max(0.0, y_lo - 0.05 * span), min(100.0, y_hi + 0.05 * span))
    else:
        y_lim = (0.0, 100.0)  # fallback

    # Plotting
    fig, axes = plt.subplots(len(kappa_rows), len(sigma2_cols), figsize=(13, 10), sharex=False, sharey=True)

    for r_idx, (k0, k1) in enumerate(kappa_rows):
        for c_idx, sigma2 in enumerate(sigma2_cols):
            ax = axes[r_idx, c_idx]
            col_data = results[(r_idx, c_idx)]

            ax.set_xlim(0.05, 0.95)
            ax.set_ylim(*y_lim)

            for g in gap_list:
                mu1_vec, pct_vec = col_data[g]
                if mu1_vec.size == 0:
                    continue
                style = CURVE_STYLE[g]
                ax.plot(mu1_vec, pct_vec,
                        marker=style["marker"], linewidth=style["linewidth"], markersize=style["markersize"],
                        color=style["color"], label=style["label"], alpha=0.95)

            if r_idx == 0:
                ax.set_title(rf"$\sigma^2={sigma2:.2f}$", fontsize=11, pad=6)
            if c_idx == 0:
                ax.set_ylabel(rf"$\kappa: {k0:.2f}\rightarrow{k1:.2f}$" + "\n" + "New adopters (%)", fontsize=10)
            else:
                ax.set_ylabel("")

            ax.set_xlabel(r"$\mu_1$ (mean of $\theta_1$)", fontsize=9)
            ax.grid(True, alpha=0.35, linewidth=0.7)

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

    fig.suptitle(
        "Share of New Adopters (Logit-Normal, exact moment-matching)\n"
        "Rows: κ ∈ {0.80→0.85, 0.80→0.90, 0.80→0.95}; "
        "Cols: Var(θ) = {0.05, 0.10, 0.15, 0.20}",
        fontsize=15, y=0.95
    )

    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.905),
               ncols=3, frameon=True, fontsize=10, handlelength=2.5, columnspacing=1.6, borderpad=0.6)

    footer = (f"N={N} farmers; R={R} worlds; γ~Uniform[0,10]; "
              f"Adoption threshold={STORE_THRESHOLD}; "
              "s* via golden-section; Logit-Normal with moment-matching; "
              "Markers aid B/W reproduction.")
    fig.text(0.5, 0.018, footer, ha="center", va="center", fontsize=10)

    fig.tight_layout(rect=[0.04, 0.06, 0.98, 0.88])
    plt.savefig(FIG_PATH, dpi=DPI, bbox_inches="tight")
    plt.show()
    return FIG_PATH

if __name__ == "__main__":
    out = run_and_plot()
    print(out)
