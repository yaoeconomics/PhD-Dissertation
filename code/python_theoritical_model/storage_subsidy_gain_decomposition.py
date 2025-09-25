
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from numpy.polynomial.legendre import leggauss

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


mu_grid = np.arange(0.10, 0.91, 0.05)
sigma2 = 0.02
N = 100
gammas = rng.uniform(0, 10, size=N); gammas.sort()
deciles = pd.qcut(gammas, 10, labels=False)  # 0..9
kappa_scenarios = [0.75, 0.80, 0.85, 0.90, 0.95]
kappa_pairs = [(0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95)]
R = 500  # worlds per μ
# 12-point Gauss-Legendre for accurate inner expectations
nodes, weights = leggauss(12)
x_nodes = 0.5*(nodes+1); w_nodes = 0.5*weights

# -----------------------------
# Utility and helpers
# -----------------------------
def crra_u(pi, gamma):
    pi = np.maximum(pi, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(pi)
    return (pi**(1-gamma) - 1) / (1-gamma)

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
    gamma_grid  = np.linspace(0.0, 10.0, 15)

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
        RP_bar = Y_bar - CE_bar
        summary_rows.append({
            "mu": mu, "kappa": kappa,
            "mean_income": Y_bar.mean(),
            "median_income": np.median(Y_bar),
            "mean_CE": CE_bar.mean(),
            "mean_RP": RP_bar.mean(),
            "share_storing": store_count/(N*R),
            "mean_s": s_bar.mean()
        })

summary_df = pd.DataFrame(summary_rows)
rn_df = pd.DataFrame(rn_rows)

# ---- Plots for (1) ----
plt.figure(figsize=(10, 6))
for kappa in kappa_scenarios:
    d = summary_df[summary_df["kappa"] == kappa].sort_values("mu")
    plt.plot(d["mu"], d["share_storing"], marker="o", label=f"κ={kappa}")
plt.xlabel("Belief about buyer power mean μ")
plt.ylabel("Share storing (farmer–world)")
plt.title("Share of Storing vs. Beliefs μ, by Storage Efficiency κ")
plt.grid(True)
plt.legend(ncols=3)
plt.tight_layout()
plt.savefig(os.path.join(target_dir, "storage_subsidy_share_of_storage.png"),
            dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 6))
for kappa in kappa_scenarios:
    d = summary_df[summary_df["kappa"] == kappa].sort_values("mu")
    plt.plot(d["mu"], d["mean_s"], marker="o", label=f"κ={kappa}")
plt.xlabel("Belief about buyer power mean μ")
plt.ylabel("Mean storage share s*")
plt.title("Mean s* vs. Beliefs μ, by Storage Efficiency κ")
plt.grid(True)
plt.legend(ncols=3)
plt.tight_layout()
plt.savefig(os.path.join(target_dir, "storage_subsidy_mean_s.png"),
            dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# (2) Decile-by-γ decomposition across κ-contrasts
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
    gamma_grid  = np.linspace(0.0, 10.0, 15)

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
        RP_bar = Y_bar - CE_bar
        stats[kappa] = {"Y": Y_bar, "CE": CE_bar, "RP": RP_bar}

    df0 = pd.DataFrame({
        "gamma": gammas, "decile": deciles,
        "Y0": stats[k0]["Y"], "CE0": stats[k0]["CE"], "RP0": stats[k0]["RP"]
    })
    df1 = pd.DataFrame({
        "gamma": gammas, "decile": deciles,
        "Y1": stats[k1]["Y"], "CE1": stats[k1]["CE"], "RP1": stats[k1]["RP"]
    })
    df = df0.merge(df1[["Y1", "CE1", "RP1"]], left_index=True, right_index=True)
    df["dY"]  = df["Y1"]  - df["Y0"]
    df["dCE"] = df["CE1"] - df["CE0"]
    df["dRP"] = df["RP1"] - df["RP0"]

    g = df.groupby("decile").agg(
        gamma_lo=("gamma", "min"),
        gamma_hi=("gamma", "max"),
        ΔEπ=("dY", "mean"),
        ΔCE=("dCE", "mean"),
        ΔRP=("dRP", "mean")
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

# Average across μ to show a clean decile-by-γ summary per contrast
avg_panel = panel.groupby(["contrast", "decile"]).agg(
    gamma_lo=("gamma_lo", "mean"),
    gamma_hi=("gamma_hi", "mean"),
    ΔEπ=("ΔEπ", "mean"),
    ΔCE=("ΔCE", "mean"),
    ΔRP=("ΔRP", "mean")
).reset_index()
avg_panel["−ΔRP (insurance gain)"] = -avg_panel["ΔRP"]

# ---- Plot for (2): 2×2 grid, one subplot per κ-contrast ----
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
axes = axes.ravel()
for ax, contrast in zip(axes, avg_panel["contrast"].unique()):
    sub = avg_panel[avg_panel["contrast"] == contrast].sort_values("decile")
    x = sub["decile"] + 1
    width = 0.35
    ax.bar(x - width/2, sub["ΔEπ"], width=width, label="ΔE[π] (growth)")
    ax.bar(x + width/2, sub["−ΔRP (insurance gain)"], width=width, label="−ΔRP (insurance)")
    ax.plot(x, sub["ΔCE"], marker="o", linestyle="-", label="ΔCE (total)")
    ax.set_title(f"κ {contrast}")
    ax.set_xlabel("γ decile (low → high risk aversion)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
axes[0].set_ylabel("Gain (price units)")
axes[2].set_ylabel("Gain (price units)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig(os.path.join(target_dir, "storage_subsidy_gain_decomposition.png"),
            dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# Export core tables
# -----------------------------
summary_df.to_csv(os.path.join(target_dir, "summary_df.csv"), index=False)
rn_df.to_csv(os.path.join(target_dir, "rn_df.csv"), index=False)
avg_panel.to_csv(os.path.join(target_dir, "avg_panel.csv"), index=False)

print("Saved figures and CSVs to:", target_dir,
      "\nREUSE_WORLDS =", REUSE_WORLDS)
