# Re-run after reset

import os

# 当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 上一级目录
parent_dir = os.path.dirname(current_dir)
# 上两级目录
grandparent_dir = os.path.dirname(parent_dir)
# 上两级平行目录（例如 results）
target_dir = os.path.join(grandparent_dir, "model_figures")

# 如果目标文件夹不存在就新建
os.makedirs(target_dir, exist_ok=True)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from numpy.polynomial.legendre import leggauss

rng = np.random.default_rng(314)

mu_grid = np.arange(0.05, 0.951, 0.10)
sigma2 = 0.02
N = 100
gammas = rng.uniform(0, 10, size=N); gammas.sort()
deciles = pd.qcut(gammas, 10, labels=False)
kappa_pairs = [(0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95)]
R = 280
nodes, weights = leggauss(12)
x_nodes = 0.5*(nodes+1); w_nodes = 0.5*weights

def crra_u(pi, gamma):
    pi = np.maximum(pi, 1e-12)
    if abs(gamma-1.0) < 1e-12: return np.log(pi)
    return (pi**(1-gamma)-1)/(1-gamma)

def crra_u_inv(u, gamma):
    if abs(gamma-1.0) < 1e-12: return np.exp(u)
    return np.maximum(u*(1-gamma)+1, 1e-12)**(1/(1-gamma))

def alpha_beta_from(mu, sigma2):
    factor = mu*(1-mu)/sigma2 - 1
    return mu*factor, (1-mu)*factor

def solve_s(theta1, gamma, kappa, theta2_nodes, w_nodes, E_inv1p_theta2):
    p1 = 1.0/(1.0+theta1)
    if abs(gamma) < 1e-12:
        return 1.0 if (kappa*E_inv1p_theta2 > p1) else 0.0
    s_grid = np.linspace(0,1,33)
    EU = []
    for s in s_grid:
        incomes = (1-s)*p1 + s*kappa/(1.0+theta2_nodes)
        EU.append(np.sum(crra_u(incomes, gamma)*w_nodes))
    s0 = s_grid[int(np.argmax(EU))]
    lo, hi = max(0.0, s0-0.1), min(1.0, s0+0.1)
    s_fine = np.linspace(lo, hi, 33)
    EUf = []
    for s in s_fine:
        incomes = (1-s)*p1 + s*kappa/(1.0+theta2_nodes)
        EUf.append(np.sum(crra_u(incomes, gamma)*w_nodes))
    return float(s_fine[int(np.argmax(EUf))])

def precompute_cache(kappas, theta1_grid, gamma_grid, theta2_nodes, w_nodes, E_inv1p_theta2):
    cache = {k: np.zeros((len(theta1_grid), len(gamma_grid))) for k in kappas}
    for k in kappas:
        for i, t1 in enumerate(theta1_grid):
            for j, g in enumerate(gamma_grid):
                cache[k][i,j] = solve_s(t1, g, k, theta2_nodes, w_nodes, E_inv1p_theta2)
    return cache

def interp2_vec(xgrid, ygrid, Z, x, y_arr):
    xi = np.searchsorted(xgrid, x) - 1
    xi = np.clip(xi, 0, len(xgrid)-2)
    x0, x1 = xgrid[xi], xgrid[xi+1]
    tx = (x - x0) / (x1 - x0 + 1e-12)
    yi = np.searchsorted(ygrid, y_arr) - 1
    yi = np.clip(yi, 0, len(ygrid)-2)
    y0 = ygrid[yi]; y1 = ygrid[yi+1]
    ty = (y_arr - y0) / (y1 - y0 + 1e-12)
    z00 = Z[xi, yi]; z01 = Z[xi, yi+1]; z10 = Z[xi+1, yi]; z11 = Z[xi+1, yi+1]
    return (1-tx)*(1-ty)*z00 + (1-tx)*ty*z01 + tx*(1-ty)*z10 + tx*ty*z11

def run_mu_contrast(mu, k0, k1):
    alpha, beta_param = alpha_beta_from(mu, sigma2)
    theta1_worlds = beta.rvs(alpha, beta_param, size=R, random_state=rng)
    theta2_worlds = beta.rvs(alpha, beta_param, size=R, random_state=rng)
    theta2_nodes = beta.ppf(x_nodes, alpha, beta_param)
    theta2_nodes = np.clip(theta2_nodes, 1e-9, 1-1e-9)
    E_inv1p_theta2 = float(np.sum(1.0/(1.0+theta2_nodes)*w_nodes))
    theta1_grid = np.linspace(0.001, 0.999, 15)
    gamma_grid = np.linspace(0.0, 10.0, 15)
    s_cache = precompute_cache([k0, k1], theta1_grid, gamma_grid, theta2_nodes, w_nodes, E_inv1p_theta2)
    stats = {}
    for kappa in [k0, k1]:
        Y_bar = np.zeros(N); EU_bar = np.zeros(N)
        for r in range(R):
            t1 = theta1_worlds[r]; t2 = theta2_worlds[r]
            p1 = 1.0/(1.0+t1); p2 = 1.0/(1.0+t2)
            s_star = interp2_vec(theta1_grid, gamma_grid, s_cache[kappa], t1, gammas)
            inc = (1 - s_star)*p1 + s_star*kappa*p2
            Uvals = np.array([crra_u(inc[i], gammas[i]) for i in range(N)])
            Y_bar += (inc - Y_bar)/(r+1)
            EU_bar += (Uvals - EU_bar)/(r+1)
        CE_bar = np.array([crra_u_inv(EU_bar[i], gammas[i]) for i in range(N)])
        RP_bar = Y_bar - CE_bar
        stats[kappa] = {"Y": Y_bar, "CE": CE_bar, "RP": RP_bar}
    df0 = pd.DataFrame({"gamma": gammas, "decile": deciles,
                        "Y0": stats[k0]["Y"], "CE0": stats[k0]["CE"], "RP0": stats[k0]["RP"]})
    df1 = pd.DataFrame({"gamma": gammas, "decile": deciles,
                        "Y1": stats[k1]["Y"], "CE1": stats[k1]["CE"], "RP1": stats[k1]["RP"]})
    df = df0.merge(df1[["Y1","CE1","RP1"]], left_index=True, right_index=True)
    df["dY"] = df["Y1"] - df["Y0"]
    df["dCE"] = df["CE1"] - df["CE0"]
    df["dRP"] = df["RP1"] - df["RP0"]
    g = df.groupby("decile").agg(
        gamma_lo=("gamma","min"),
        gamma_hi=("gamma","max"),
        ΔEπ=("dY","mean"),
        ΔCE=("dCE","mean"),
        ΔRP=("dRP","mean")
    ).reset_index()
    g["μ"] = mu; g["contrast"] = f"{k0:.2f}→{k1:.2f}"
    return g

records = []
for mu in mu_grid:
    for (k0,k1) in kappa_pairs:
        records.append(run_mu_contrast(mu, k0, k1))
panel = pd.concat(records, ignore_index=True)

avg_panel = panel.groupby(["contrast","decile"]).agg(
    gamma_lo=("gamma_lo","mean"),
    gamma_hi=("gamma_hi","mean"),
    ΔEπ=("ΔEπ","mean"),
    ΔCE=("ΔCE","mean"),
    ΔRP=("ΔRP","mean")
).reset_index()
avg_panel["−ΔRP (insurance gain)"] = -avg_panel["ΔRP"]

fig, axes = plt.subplots(2, 2, figsize=(12,8), sharey=True)
axes = axes.ravel()
for ax, contrast in zip(axes, avg_panel["contrast"].unique()):
    sub = avg_panel[avg_panel["contrast"]==contrast].sort_values("decile")
    x = sub["decile"] + 1
    width = 0.35
    ax.bar(x - width/2, sub["ΔEπ"], width=width, label="ΔE[π] (growth)")
    ax.bar(x + width/2, sub["−ΔRP (insurance gain)"], width=width, label="−ΔRP (insurance)")
    ax.plot(x, sub["ΔCE"], marker="o", linestyle="-", label="ΔCE (total)")
    ax.set_title(f"κ {contrast}")
    ax.set_xlabel("γ (risk aversion)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
axes[0].set_ylabel("Gain (price units)")
axes[2].set_ylabel("Gain (price units)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()

plt.savefig(os.path.join(target_dir, "storage_subsidy_gain_decomposition.png"), dpi=300, bbox_inches="tight")
plt.show()
