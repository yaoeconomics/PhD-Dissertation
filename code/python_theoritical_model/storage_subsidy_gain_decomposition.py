import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from numpy.polynomial.legendre import leggauss

# Global font settings
plt.rcParams.update({
    "axes.titlesize": 18,      # panel titles
    "axes.labelsize": 16,      # axis labels
    "xtick.labelsize": 14,     # x tick labels
    "ytick.labelsize": 14,     # y tick labels
    "legend.fontsize": 13,     # legend entries
    "legend.title_fontsize": 14,
    "figure.titlesize": 20     # suptitle
})

# -----------------------------
# Switch
# -----------------------------
REUSE_WORLDS = True   # <<==== toggle here

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
# Config
# -----------------------------
rng = np.random.default_rng(314)  # reproducible


mu_grid = np.arange(0.15, 0.86, 0.05)
sigma2 = 0.10
N = 100
gammas = rng.uniform(0, 5, size=N); gammas.sort()
# gammas = 10.0 * rng.beta(1, 5, size=N); gammas.sort()
quintiles = pd.qcut(gammas, 5, labels=False)  # 0..9
kappa_scenarios = [0.75, 0.80, 0.85, 0.90, 0.95]
kappa_pairs = [(0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95)]
R = 5000  # worlds per μ
# 12-point Gauss-Legendre for accurate inner expectations
nodes, weights = leggauss(12)
x_nodes = 0.5*(nodes+1); w_nodes = 0.5*weights

# -----------------------------
# Utility and helpers
# -----------------------------
def crra_u(y, gamma):
    y = np.maximum(y, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(y)
    return (y**(1-gamma) - 1) / (1-gamma)

def crra_u_inv(u, gamma):
    if abs(gamma - 1.0) < 1e-12:
        return np.exp(u)
    return np.maximum(u*(1-gamma)+1, 1e-12)**(1/(1-gamma))

def alpha_beta_from(mu, sigma2):
    # Var(theta) = mu(1-mu)/(alpha+beta+1)  =>  alpha+beta = mu(1-mu)/sigma2 - 1
    factor = mu*(1-mu)/sigma2 - 1
    return mu*factor, (1-mu)*factor

def solve_s(theta1, gamma, kappa, theta2_nodes, w_nodes, E_inv1p_theta2):
    # Optimal s* given realized theta1, risk aversion gamma, and κ
    p1 = 1.0/(1.0 + theta1)
    if abs(gamma) < 1e-12:  # risk-neutral bang-bang rule
        return 1.0 if (kappa*E_inv1p_theta2 > p1) else 0.0
    # coarse-to-fine grid search over s in [0,1]
    s_grid = np.linspace(0, 1, 33)
    EU = []
    for s in s_grid:
        incomes = (1 - s)*p1 + s*kappa/(1.0 + theta2_nodes)
        EU.append(np.sum(crra_u(incomes, gamma)*w_nodes))
    s0 = s_grid[int(np.argmax(EU))]
    lo, hi = max(0.0, s0 - 0.1), min(1.0, s0 + 0.1)
    s_fine = np.linspace(lo, hi, 33)
    EUf = []
    for s in s_fine:
        incomes = (1 - s)*p1 + s*kappa/(1.0 + theta2_nodes)
        EUf.append(np.sum(crra_u(incomes, gamma)*w_nodes))
    return float(s_fine[int(np.argmax(EUf))])

def precompute_cache(kappas, theta1_grid, gamma_grid, theta2_nodes, w_nodes, E_inv1p_theta2):
    # Precompute s*(theta1, gamma | κ) on a coarse grid for fast interpolation
    cache = {k: np.zeros((len(theta1_grid), len(gamma_grid))) for k in kappas}
    for k in kappas:
        for i, t1 in enumerate(theta1_grid):
            for j, g in enumerate(gamma_grid):
                cache[k][i, j] = solve_s(t1, g, k, theta2_nodes, w_nodes, E_inv1p_theta2)
    return cache

def interp2_vec(xgrid, ygrid, Z, x, y_arr):
    # Bilinear interpolation of Z over (xgrid, ygrid) at (x, y_arr), vectorized in y_arr
    xi = np.searchsorted(xgrid, x) - 1
    xi = np.clip(xi, 0, len(xgrid)-2)
    x0, x1 = xgrid[xi], xgrid[xi+1]
    tx = (x - x0) / (x1 - x0 + 1e-12)

    yi = np.searchsorted(ygrid, y_arr) - 1
    yi = np.clip(yi, 0, len(ygrid)-2)
    y0 = ygrid[yi]; y1 = ygrid[yi+1]
    ty = (y_arr - y0) / (y1 - y0 + 1e-12)

    z00 = Z[xi, yi];   z01 = Z[xi, yi+1]
    z10 = Z[xi+1, yi]; z11 = Z[xi+1, yi+1]
    return (1-tx)*(1-ty)*z00 + (1-tx)*ty*z01 + tx*(1-ty)*z10 + tx*ty*z11

# -----------------------------
# Precompute per-μ objects (nodes, expectations, optional worlds)
# -----------------------------
nodes_cache = {}   # key: rounded μ -> dict(theta2_nodes=..., E_inv1p=..., alpha=..., beta=...)
worlds_cache = {}  # key: rounded μ -> tuple(theta1_worlds, theta2_worlds)

def mu_key(mu):  # avoid float-key surprises
    return round(float(mu), 3)

for mu in mu_grid:
    ak, bk = alpha_beta_from(mu, sigma2)
    theta2_nodes = beta.ppf(x_nodes, ak, bk)
    theta2_nodes = np.clip(theta2_nodes, 1e-9, 1-1e-9)
    E_inv1p = float(np.sum(1.0/(1.0 + theta2_nodes) * w_nodes))
    nodes_cache[mu_key(mu)] = dict(theta2_nodes=theta2_nodes, E_inv1p=E_inv1p, alpha=ak, beta=bk)

    if REUSE_WORLDS:
        theta1_worlds = beta.rvs(ak, bk, size=R, random_state=rng)
        theta2_worlds = beta.rvs(ak, bk, size=R, random_state=rng)
        worlds_cache[mu_key(mu)] = (theta1_worlds, theta2_worlds)

# -----------------------------
# (1) Summary surfaces across κ (optionally reusing worlds per μ)
# -----------------------------
summary_rows = []
rn_rows = []

for mu in mu_grid:
    key = mu_key(mu)
    ak = nodes_cache[key]["alpha"]; bk = nodes_cache[key]["beta"]
    theta2_nodes = nodes_cache[key]["theta2_nodes"]
    E_inv1p_theta2 = nodes_cache[key]["E_inv1p"]

    # worlds: reuse if available, else draw
    if REUSE_WORLDS:
        theta1_worlds, theta2_worlds = worlds_cache[key]
    else:
        theta1_worlds = beta.rvs(ak, bk, size=R, random_state=rng)
        theta2_worlds = beta.rvs(ak, bk, size=R, random_state=rng)

    # Grids for precompute
    theta1_grid = np.linspace(0.001, 0.999, 15)
    gamma_grid  = np.linspace(0.0, 5.0, 15)

    # Precompute s*(θ1,γ | κ) for all κ scenarios
    s_cache = precompute_cache(kappa_scenarios, theta1_grid, gamma_grid,
                               theta2_nodes, w_nodes, E_inv1p_theta2)

    # Risk-neutral storage probabilities (analytic + MC)
    for kappa in kappa_scenarios:
        tau = 1.0/(kappa * E_inv1p_theta2) - 1.0
        if tau <= 0:
            prob_analytic = 1.0
        elif tau >= 1:
            prob_analytic = 0.0
        else:
            prob_analytic = 1.0 - beta.cdf(tau, ak, bk)
        p1s = 1.0/(1.0 + theta1_worlds)
        rn_mc = float(np.mean(kappa * E_inv1p_theta2 > p1s))
        rn_rows.append({
            "mu": mu, "kappa": kappa, "tau_threshold": tau,
            "RN_store_prob_analytic": prob_analytic,
            "RN_store_prob_MC": rn_mc
        })

    # CRRA population summaries for each κ
    for kappa in kappa_scenarios:
        s_bar = np.zeros(N); Y_bar = np.zeros(N); EU_bar = np.zeros(N)
        store_count = 0; eps = 1e-8
        for r in range(R):
            t1 = theta1_worlds[r]; t2 = theta2_worlds[r]
            p1 = 1.0/(1.0 + t1); p2 = 1.0/(1.0 + t2)
            s_star = interp2_vec(theta1_grid, gamma_grid, s_cache[kappa], t1, gammas)
            store_count += np.sum(s_star > eps)
            incomes = (1 - s_star)*p1 + s_star*kappa*p2
            Uvals = np.array([crra_u(incomes[i], gammas[i]) for i in range(N)])
            # online averages
            Y_bar += (incomes - Y_bar)/(r+1)
            EU_bar += (Uvals  - EU_bar)/(r+1)
            s_bar += (s_star  - s_bar)/(r+1)
        CE_bar = np.array([crra_u_inv(EU_bar[i], gammas[i]) for i in range(N)])
        RS_bar = Y_bar - CE_bar
        summary_rows.append({
            "mu": mu, "kappa": kappa,
            "mean_income": Y_bar.mean(),
            "median_income": np.median(Y_bar),
            "mean_CE": CE_bar.mean(),
            "mean_RS": RS_bar.mean(),
            "share_storing": store_count/(N*R),
            "mean_s": s_bar.mean()
        })

summary_df = pd.DataFrame(summary_rows)
rn_df = pd.DataFrame(rn_rows)



# -----------------------------
# (1) quintile-by-γ decomposition across κ-contrasts
# -----------------------------
def run_mu_contrast(mu, k0, k1, nodes_cache, worlds_cache, reuse_worlds):
    key = mu_key(mu)
    ak = nodes_cache[key]["alpha"]; bk = nodes_cache[key]["beta"]
    theta2_nodes = nodes_cache[key]["theta2_nodes"]
    E_inv1p_theta2 = nodes_cache[key]["E_inv1p"]

    # worlds: reuse if flag set, else draw fresh
    if reuse_worlds:
        theta1_worlds, theta2_worlds = worlds_cache[key]
    else:
        theta1_worlds = beta.rvs(ak, bk, size=R, random_state=rng)
        theta2_worlds = beta.rvs(ak, bk, size=R, random_state=rng)

    theta1_grid = np.linspace(0.001, 0.999, 15)
    gamma_grid  = np.linspace(0.0, 5.0, 15)

    s_cache = precompute_cache([k0, k1], theta1_grid, gamma_grid,
                               theta2_nodes, w_nodes, E_inv1p_theta2)
    stats = {}
    for kappa in [k0, k1]:
        Y_bar = np.zeros(N); EU_bar = np.zeros(N)
        for r in range(R):
            t1 = theta1_worlds[r]; t2 = theta2_worlds[r]
            p1 = 1.0/(1.0 + t1); p2 = 1.0/(1.0 + t2)
            s_star = interp2_vec(theta1_grid, gamma_grid, s_cache[kappa], t1, gammas)
            inc = (1 - s_star)*p1 + s_star*kappa*p2
            Uvals = np.array([crra_u(inc[i], gammas[i]) for i in range(N)])
            Y_bar += (inc - Y_bar)/(r+1)
            EU_bar += (Uvals - EU_bar)/(r+1)
        CE_bar = np.array([crra_u_inv(EU_bar[i], gammas[i]) for i in range(N)])
        RS_bar = Y_bar - CE_bar
        stats[kappa] = {"Y": Y_bar, "CE": CE_bar, "RS": RS_bar}

    df0 = pd.DataFrame({
        "gamma": gammas, "quintile": quintiles,
        "Y0": stats[k0]["Y"], "CE0": stats[k0]["CE"], "RS0": stats[k0]["RS"]
    })
    df1 = pd.DataFrame({
        "gamma": gammas, "quintile": quintiles,
        "Y1": stats[k1]["Y"], "CE1": stats[k1]["CE"], "RS1": stats[k1]["RS"]
    })
    df = df0.merge(df1[["Y1", "CE1", "RS1"]], left_index=True, right_index=True)
    df["dY"]  = df["Y1"]  - df["Y0"]
    df["dCE"] = df["CE1"] - df["CE0"]
    df["dRS"] = df["RS1"] - df["RS0"]

    g = df.groupby("quintile").agg(
        gamma_lo=("gamma", "min"),
        gamma_hi=("gamma", "max"),
        Δy=("dY", "mean"),
        ΔCE=("dCE", "mean"),
        ΔRS=("dRS", "mean")
    ).reset_index()
    g["μ"] = mu
    g["contrast"] = f"{k0:.2f}→{k1:.2f}"
    return g

# Build panel across μ and all κ-contrasts
records = []
for mu in mu_grid:
    for (k0, k1) in kappa_pairs:
        records.append(run_mu_contrast(mu, k0, k1, nodes_cache, worlds_cache, REUSE_WORLDS))
panel = pd.concat(records, ignore_index=True)

# Average across μ to show a clean quintile-by-γ summary per contrast
avg_panel = panel.groupby(["contrast", "quintile"]).agg(
    gamma_lo=("gamma_lo", "mean"),
    gamma_hi=("gamma_hi", "mean"),
    Δy=("Δy", "mean"),
    ΔCE=("ΔCE", "mean"),
    ΔRS=("ΔRS", "mean")
).reset_index()
avg_panel["−ΔRS (risk spreading)"] = -avg_panel["ΔRS"]

# ---- Plot for (2): 2×2 grid, one subplot per κ-contrast ----
fig, axes = plt.subplots(1, 4, figsize=(16, 8), sharey=True)
axes = axes.ravel()
for ax, contrast in zip(axes, avg_panel["contrast"].unique()):
    sub = avg_panel[avg_panel["contrast"] == contrast].sort_values("quintile")
    x = sub["quintile"] + 1
    width = 0.35
    ax.bar(x - width/2, sub["Δy"], width=width, label="Δ[y] (income growth)")
    ax.bar(x + width/2, sub["−ΔRS (risk spreading)"], width=width, label="−ΔRS (risk spreading)")
    ax.plot(x, sub["ΔCE"], marker="o", linestyle="-", label="ΔCE (total)")
    ax.set_title(f"κ {contrast}")
    ax.set_xlabel("γ quintile")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
axes[0].set_ylabel("Gain (price units)")
axes[2].set_ylabel("Gain (price units)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.12))   # move legend further down

plt.subplots_adjust(bottom=0.10)          # create room for it
plt.tight_layout()
plt.savefig(os.path.join(target_dir, "storage_subsidy_gain_decomposition.png"),
            dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------
# (2) Storage regime composition: none / partial / full by γ quintile
#     Stacked bars — two per quintile (low κ vs high κ) stacking to 1.0
# -----------------------------
def run_regime_shares(mu, k0, k1, nodes_cache, worlds_cache, reuse_worlds):
    """Classify each farmer–world s* into none/partial/full.
    Return quintile-level shares for both κ values."""
    key = mu_key(mu)
    ak = nodes_cache[key]["alpha"]; bk = nodes_cache[key]["beta"]
    theta2_nodes = nodes_cache[key]["theta2_nodes"]
    E_inv1p_theta2 = nodes_cache[key]["E_inv1p"]

    if reuse_worlds:
        theta1_worlds, theta2_worlds = worlds_cache[key]
    else:
        theta1_worlds = beta.rvs(ak, bk, size=R, random_state=rng)
        theta2_worlds = beta.rvs(ak, bk, size=R, random_state=rng)

    theta1_grid = np.linspace(0.001, 0.999, 15)
    gamma_grid  = np.linspace(0.0, 5.0, 15)

    s_cache_local = precompute_cache([k0, k1], theta1_grid, gamma_grid,
                                     theta2_nodes, w_nodes, E_inv1p_theta2)

    eps_lo = 1e-4
    eps_hi = 1 - 1e-4

    counts = {}
    for kappa in [k0, k1]:
        none_c = np.zeros(N); part_c = np.zeros(N); full_c = np.zeros(N)
        for r in range(R):
            t1 = theta1_worlds[r]
            s_star = interp2_vec(theta1_grid, gamma_grid, s_cache_local[kappa], t1, gammas)
            none_c += (s_star <= eps_lo)
            full_c += (s_star >= eps_hi)
            part_c += ((s_star > eps_lo) & (s_star < eps_hi))
        counts[kappa] = {
            "none": none_c / R, "partial": part_c / R, "full": full_c / R
        }

    rows = []
    for kappa in [k0, k1]:
        df_tmp = pd.DataFrame({
            "quintile": quintiles,
            "none": counts[kappa]["none"],
            "partial": counts[kappa]["partial"],
            "full": counts[kappa]["full"]
        })
        g = df_tmp.groupby("quintile").mean().reset_index()
        g["kappa"] = kappa
        g["mu"] = mu
        g["contrast"] = f"{k0:.2f}→{k1:.2f}"
        rows.append(g)
    return pd.concat(rows, ignore_index=True)

regime_records = []
for mu in mu_grid:
    for (k0, k1) in kappa_pairs:
        regime_records.append(
            run_regime_shares(mu, k0, k1, nodes_cache, worlds_cache, REUSE_WORLDS)
        )
regime_panel = pd.concat(regime_records, ignore_index=True)

regime_avg = regime_panel.groupby(["contrast", "kappa", "quintile"]).agg(
    none=("none", "mean"),
    partial=("partial", "mean"),
    full=("full", "mean")
).reset_index()

# ---- Plot (3): stacked bars, two per quintile ----
fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=True)

color_none    = "#d9d9d9"   # light grey
color_partial = "#fdae6b"   # warm orange
color_full    = "#e6550d"   # deep orange

bar_width = 0.32

for ax, contrast in zip(axes, regime_avg["contrast"].unique()):
    sub = regime_avg[regime_avg["contrast"] == contrast]
    k0_val = sub["kappa"].min()
    k1_val = sub["kappa"].max()
    sub_lo = sub[sub["kappa"] == k0_val].sort_values("quintile")
    sub_hi = sub[sub["kappa"] == k1_val].sort_values("quintile")
    x = np.arange(1, 6)

    # Low κ stacked bars (left of center)
    ax.bar(x - bar_width/2, sub_lo["none"].values,    width=bar_width,
           color=color_none, edgecolor="white", linewidth=0.8)
    ax.bar(x - bar_width/2, sub_lo["partial"].values,  width=bar_width,
           bottom=sub_lo["none"].values,
           color=color_partial, edgecolor="white", linewidth=0.8)
    ax.bar(x - bar_width/2, sub_lo["full"].values,     width=bar_width,
           bottom=sub_lo["none"].values + sub_lo["partial"].values,
           color=color_full, edgecolor="white", linewidth=0.8)

    # High κ stacked bars (right of center)
    ax.bar(x + bar_width/2, sub_hi["none"].values,    width=bar_width,
           color=color_none, edgecolor="white", linewidth=0.8,
           hatch="//", alpha=0.9)
    ax.bar(x + bar_width/2, sub_hi["partial"].values,  width=bar_width,
           bottom=sub_hi["none"].values,
           color=color_partial, edgecolor="white", linewidth=0.8,
           hatch="//", alpha=0.9)
    ax.bar(x + bar_width/2, sub_hi["full"].values,     width=bar_width,
           bottom=sub_hi["none"].values + sub_hi["partial"].values,
           color=color_full, edgecolor="white", linewidth=0.8,
           hatch="//", alpha=0.9)

    # Quintile labels at bottom
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i}" for i in x])
    ax.set_title(f"κ: {k0_val:.2f} → {k1_val:.2f}")
    ax.set_xlabel("γ quintile (low → high risk aversion)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

axes[0].set_ylabel("Share of farmer–worlds")

# Build legend manually
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=color_none,    edgecolor="grey",  label="No storage (s*≈0)"),
    Patch(facecolor=color_partial, edgecolor="grey",  label="Partial storage (0<s*<1)"),
    Patch(facecolor=color_full,    edgecolor="grey",  label="Full storage (s*≈1)"),
    Patch(facecolor="white", edgecolor="grey", label=f"Solid = low κ"),
    Patch(facecolor="white", edgecolor="grey", hatch="//", label=f"Hatched = high κ"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.08), fontsize=12,
           frameon=True, fancybox=True, shadow=False)

fig.suptitle("Storage Regime Composition by Risk Aversion and Efficiency Gain",
             y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(target_dir, "storage_subsidy_regime_composition.png"),
            dpi=300, bbox_inches="tight")
plt.show()

print("Saved figures and CSVs to:", target_dir,
      "\nREUSE_WORLDS =", REUSE_WORLDS)