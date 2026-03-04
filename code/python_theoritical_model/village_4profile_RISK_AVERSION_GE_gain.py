# -*- coding: utf-8 -*-
"""
Village welfare comparison: 4 risk-aversion profiles × 2×4 (κ × σ²) grid

Village profiles (N=100 farmers each):
  V1 — Risk-neutral:    all γ_i = 0
  V2 — Uniform:         γ ~ Unif[0,5],                       mean = 2.5, skew =  0
  V3 — Right-skewed:    γ ~ Beta(1.5, 3.5) scaled to [0,5],  mean ≈ 1.5, skew = +0.6
  V4 — Left-skewed:     γ ~ Beta(3.5, 1.5) scaled to [0,5],  mean ≈ 3.5, skew = -0.6

Each panel: village-average ΔCE (% of no-storage income) vs μ_θ (with μ1=μ2).

Computation (exact, no interpolation):
  - θ1 and θ2 are both integrated via 20-point Gauss–Legendre quadrature
  - For each (θ1 quadrature node, farmer γ), ΔCE is computed individually
  - Village average = mean over N farmers; expectation over θ1 = quadrature dot product
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import beta
from numpy.polynomial.legendre import leggauss

# ----------------------------- Global font settings
FS_BASE    = 22
FS_TITLE   = 24
FS_LABEL   = 22
FS_TICK    = 20
FS_LEGEND  = 21
FS_SUPTITLE = 26
FS_ANNOT      = 20
FS_STRIP_TITLE = 15

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         FS_BASE,
    "axes.titlesize":    FS_TITLE,
    "axes.labelsize":    FS_LABEL,
    "xtick.labelsize":   FS_TICK,
    "ytick.labelsize":   FS_TICK,
    "legend.fontsize":   FS_LEGEND,
    "figure.titlesize":  FS_SUPTITLE,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    1.1,
    "grid.linewidth":    0.7,
    "grid.alpha":        0.35,
    "grid.color":        "#cccccc",
})

# ----------------------------- Paths
try:
    CURRENT_FILE = os.path.abspath(__file__)
    current_dir  = os.path.dirname(CURRENT_FILE)
except NameError:
    current_dir = os.getcwd()

parent_dir      = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
target_dir      = os.path.join(grandparent_dir, "model_figures")
os.makedirs(target_dir, exist_ok=True)

DPI      = 300
FIG_PATH = os.path.join(target_dir, "village_4profile_individual.png")

# ----------------------------- Quadrature (shared across all integrals)
nodes, weights = leggauss(20)
x_nodes = 0.5 * (nodes + 1.0)   # mapped from [-1,1] to [0,1]
w_nodes = 0.5 * weights

# ----------------------------- Grid parameters (top two κ rows only)
kappa_rows  = [0.95, 0.90]
sigma2_cols = [0.02, 0.05, 0.10, 0.15]
mu_grid     = np.linspace(0.10, 0.90, 50)

col_feasible = {
    0.02: (0.05, 0.95),
    0.05: (0.10, 0.90),
    0.10: (0.15, 0.85),
    0.15: (0.20, 0.80),
}

# ----------------------------- Village γ samples (N=100, fixed seed)
N   = 100
rng = np.random.default_rng(42)

gammas_v1 = np.zeros(N)
gammas_v2 = np.sort(rng.uniform(0.0, 5.0, size=N))
gammas_v3 = np.sort(beta.rvs(1.5, 3.5, size=N, random_state=rng) * 5)
gammas_v4 = np.sort(beta.rvs(3.5, 1.5, size=N, random_state=rng) * 5)

village_profiles = [
    (gammas_v1, "#2ca02c", "-",  2.5,
     "V1: Risk-neutral ($\\gamma_i=0$)"),
    (gammas_v2, "#1f77b4", "--", 2.0,
     "V2: Uniform $[0,5]$  (mean $=2.5$)"),
    (gammas_v3, "#ff7f0e", "-.", 2.5,
     "V3: Beta$(1.5,\\,3.5)$ scaled to $[0,5]$  (mean $\\approx1.5$, right-skewed)"),
    (gammas_v4, "#d62728", ":",  2.5,
     "V4: Beta$(3.5,\\,1.5)$ scaled to $[0,5]$  (mean $\\approx3.5$, left-skewed)"),
]

# ----------------------------- Core functions

def get_beta_quad_nodes(mu, sigma2):
    """Gauss-Legendre nodes mapped through Beta(mu, sigma2) quantiles."""
    if sigma2 >= mu * (1.0 - mu) - 1e-12:
        return None
    factor = mu * (1.0 - mu) / sigma2 - 1.0
    a = mu * factor
    b = (1.0 - mu) * factor
    if a <= 0.0 or b <= 0.0:
        return None
    q = beta.ppf(x_nodes, a, b)
    return np.clip(q, 1e-9, 1.0 - 1e-9)


def crra_utility(inc, gamma):
    """CRRA utility, safe against inc <= 0."""
    inc = np.maximum(inc, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(inc)
    return (inc ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def crra_ce(E_u, gamma):
    """Inverse CRRA: certainty equivalent given expected utility E_u."""
    if abs(gamma - 1.0) < 1e-12:
        return np.exp(E_u)
    return np.maximum(E_u * (1.0 - gamma) + 1.0, 1e-30) ** (1.0 / (1.0 - gamma))


def golden_max(f, a=0.0, b=1.0, tol=1e-5, max_iter=80):
    """Golden-section search for the maximum of f on [a, b]."""
    gr   = (np.sqrt(5.0) + 1.0) / 2.0
    c    = b - (b - a) / gr
    d    = a + (b - a) / gr
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


def delta_ce_farmer(theta1, gamma, kappa, t2_nodes):
    """
    ΔCE for one farmer: CE_storage(θ1, γ, κ) − p1.
    θ2 integrated via quadrature (t2_nodes, w_nodes shared globals).
    """
    p1 = 1.0 / (1.0 + theta1)

    if abs(gamma) < 1e-12:
        E_p2   = float(np.dot(1.0 / (1.0 + t2_nodes), w_nodes))
        s_star = 1.0 if kappa * E_p2 > p1 else 0.0
        ce     = (1.0 - s_star) * p1 + s_star * kappa * E_p2
        return ce - p1

    def EU(s):
        inc = (1.0 - s) * p1 + s * kappa / (1.0 + t2_nodes)
        return float(np.dot(crra_utility(inc, gamma), w_nodes))

    s_star = golden_max(EU, a=0.0, b=1.0, tol=1e-5, max_iter=80)
    inc    = (1.0 - s_star) * p1 + s_star * kappa / (1.0 + t2_nodes)
    E_u    = float(np.dot(crra_utility(inc, gamma), w_nodes))
    return crra_ce(E_u, gamma) - p1


def village_avg_dce_pct(mu, sigma2, kappa, gammas):
    """
    Village-average ΔCE as % of mean no-storage income E[p1].
    Outer integral (θ1) via quadrature; N farmers looped individually.
    """
    t_nodes = get_beta_quad_nodes(mu, sigma2)
    if t_nodes is None:
        return np.nan

    mean_p1 = float(np.dot(1.0 / (1.0 + t_nodes), w_nodes))

    dce_by_theta1 = np.zeros(len(t_nodes))
    for k, t1 in enumerate(t_nodes):
        farmer_dce = np.array([
            delta_ce_farmer(t1, gamma_i, kappa, t_nodes)
            for gamma_i in gammas
        ])
        dce_by_theta1[k] = farmer_dce.mean()

    avg_dce = float(np.dot(dce_by_theta1, w_nodes))
    return 100.0 * avg_dce / (mean_p1 + 1e-12)


# ----------------------------- Compute all panels
print("Computing village ΔCE curves (N=100 farmers, individual loops)...")
results = {}

for r_idx, kappa in enumerate(kappa_rows):
    for c_idx, sigma2 in enumerate(sigma2_cols):
        print(f"  κ={kappa}, σ²={sigma2}")
        lo, hi  = col_feasible[sigma2]
        mu_feas = mu_grid[(mu_grid >= lo - 1e-9) & (mu_grid <= hi + 1e-9)]

        panel = {}
        for v_idx, (gammas, *_) in enumerate(village_profiles):
            vals = np.array([
                village_avg_dce_pct(mu, sigma2, kappa, gammas)
                for mu in mu_feas
            ])
            panel[v_idx] = {"mu": mu_feas, "vals": vals}
        results[(r_idx, c_idx)] = panel

# ----------------------------- Global y-limits
all_vals = np.concatenate([
    v["vals"][np.isfinite(v["vals"])]
    for p in results.values()
    for v in p.values()
])
pad  = 0.07 * (all_vals.max() - all_vals.min())
ymin = all_vals.min() - pad
ymax = all_vals.max() + pad

# ----------------------------- Figure layout (2 rows × 4 cols + top strip)
fig      = plt.figure(figsize=(18, 12))
gs_outer = gridspec.GridSpec(
    2, 1,
    height_ratios=[1.1, 5.0],
    hspace=0.45,
    figure=fig
)

# ── Top strip: γ distribution insets ─────────────────────────────────────────
gs_top  = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_outer[0], wspace=0.42)
gamma_x = np.linspace(0, 5, 600)

dist_info = [
    (
        "V1: Risk-neutral\n($\\gamma_i = 0$)",
        "#2ca02c", None, None
    ),
    (
        "V2: Uniform $[0,5]$\n(mean $= 2.5$)",
        "#1f77b4",
        lambda x: np.where((x >= 0) & (x <= 5), 1.0 / 5, 0.0),
        2.5
    ),
    (
        "V3: Beta$(1.5,\\,3.5)\\times 5$\n(mean $\\approx 1.5$, right-skewed)",
        "#ff7f0e",
        lambda x: beta.pdf(x / 5.0, 1.5, 3.5) / 5.0,
        1.5
    ),
    (
        "V4: Beta$(3.5,\\,1.5)\\times 5$\n(mean $\\approx 3.5$, left-skewed)",
        "#d62728",
        lambda x: beta.pdf(x / 5.0, 3.5, 1.5) / 5.0,
        3.5
    ),
]

for v_idx, (title, color, pdf_fn, mean_v) in enumerate(dist_info):
    ax = fig.add_subplot(gs_top[v_idx])
    ax.set_facecolor("#fafafa")

    if pdf_fn is None:
        ax.axvline(0, color=color, linewidth=3.5)
        ax.set_xlim(-0.3, 5.3)
        ax.set_ylim(0, 1.0)
        ax.text(0.20, 0.62, "Point mass\nat $\\gamma = 0$",
                transform=ax.transAxes, fontsize=FS_ANNOT,
                color=color, va="center", ha="left")
    else:
        y = pdf_fn(gamma_x)
        ax.plot(gamma_x, y, color=color, linewidth=2.4)
        ax.fill_between(gamma_x, y, alpha=0.20, color=color)
        ax.axvline(mean_v, color=color, linewidth=1.4,
                   linestyle="--", alpha=0.80)
        # Per-distribution label placement to avoid overlapping curves
        if v_idx == 1:   # Uniform: flat PDF, move label lower
            lbl_x, lbl_y = mean_v + 0.13, y.max() * 0.45
        elif v_idx == 3: # Left-skewed Beta: peak is on right, place label to the left
            lbl_x, lbl_y = mean_v - 2.00, y.max() * 0.88
        else:            # Right-skewed Beta: default right placement
            lbl_x, lbl_y = mean_v + 0.13, y.max() * 0.88
        ax.text(lbl_x, lbl_y, f"$\\bar{{\\gamma}}={mean_v}$",
                fontsize=FS_ANNOT, color=color)
        ax.set_xlim(-0.3, 5.3)
        ax.set_ylim(0, y.max() * 1.25)

    ax.set_title(title, fontsize=FS_STRIP_TITLE, color=color, fontweight="bold", pad=5)
    ax.set_xlabel("$\\gamma$", fontsize=FS_LABEL)
    ax.set_ylabel("Density" if v_idx == 0 else "", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── 2×4 main grid ─────────────────────────────────────────────────────────────
gs_main = gridspec.GridSpecFromSubplotSpec(
    len(kappa_rows), len(sigma2_cols),
    subplot_spec=gs_outer[1],
    hspace=0.52, wspace=0.30
)

for r_idx, kappa in enumerate(kappa_rows):
    for c_idx, sigma2 in enumerate(sigma2_cols):
        ax    = fig.add_subplot(gs_main[r_idx, c_idx])
        panel = results[(r_idx, c_idx)]
        ax.set_facecolor("#fafafa")

        ax.axhline(0, color="#888888", linewidth=1.0, linestyle="--", zorder=1)

        for v_idx, (gammas, color, ls, lw, label) in enumerate(village_profiles):
            ax.plot(panel[v_idx]["mu"], panel[v_idx]["vals"],
                    color=color, linestyle=ls, linewidth=lw, zorder=3)

        ax.set_xlim(0.08, 0.92)
        ax.set_ylim(ymin, ymax)

        if r_idx == 0:
            ax.set_title(rf"$\sigma^2_\theta = {sigma2}$",
                         fontsize=FS_TITLE, pad=6)
        if c_idx == 0:
            ax.set_ylabel(
                rf"$\kappa = {kappa}$" + "\n$\\Delta$CE vs no-storage (%)",
                fontsize=FS_LABEL
            )
        else:
            ax.set_ylabel("")

        if r_idx == len(kappa_rows) - 1:
            ax.set_xlabel(r"Mean buyer power  $\mu_\theta$", fontsize=FS_LABEL)
        else:
            ax.set_xlabel("")

        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "Village-Average Certainty-Equivalent Gain from Optimal Storage"
    r"  ($\mu_{\theta_1} = \mu_{\theta_2}$)" + "\n"
    r"Four village risk-aversion profiles  $|$  "
    r"Rows: storage efficiency $\kappa$  $|$  "
    r"Cols: buyer-power variance $\sigma^2_\theta$",
    fontsize=FS_SUPTITLE,
    y=1.01
)

fig.tight_layout(rect=[0.0, 0.02, 1.0, 1.0])
plt.savefig(FIG_PATH, dpi=DPI, bbox_inches="tight")
print(f"\nSaved → {FIG_PATH}")

fig.show()
