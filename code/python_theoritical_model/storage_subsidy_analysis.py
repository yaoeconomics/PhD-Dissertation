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


# Re-run after reset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from numpy.polynomial.legendre import leggauss

rng = np.random.default_rng(123)

mu_grid = np.arange(0.05, 0.951, 0.10)
sigma2 = 0.02
N = 100
gammas = rng.uniform(0, 10, size=N); gammas.sort()
kappa_scenarios = [0.75, 0.80, 0.85, 0.90, 0.95]
R = 250
nodes, weights = leggauss(10)
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
    y0 = ygrid[yi]
    y1 = ygrid[yi+1]
    ty = (y_arr - y0) / (y1 - y0 + 1e-12)

    z00 = Z[xi, yi]
    z01 = Z[xi, yi+1]
    z10 = Z[xi+1, yi]
    z11 = Z[xi+1, yi+1]

    return (1-tx)*(1-ty)*z00 + (1-tx)*ty*z01 + tx*(1-ty)*z10 + tx*ty*z11

summary_rows = []
rn_rows = []

for mu in mu_grid:
    alpha, beta_param = alpha_beta_from(mu, sigma2)
    theta1_worlds = beta.rvs(alpha, beta_param, size=R, random_state=rng)
    theta2_worlds = beta.rvs(alpha, beta_param, size=R, random_state=rng)
    theta2_nodes = beta.ppf(x_nodes, alpha, beta_param)
    theta2_nodes = np.clip(theta2_nodes, 1e-9, 1-1e-9)
    E_inv1p_theta2 = float(np.sum(1.0/(1.0+theta2_nodes)*w_nodes))
    theta1_grid = np.linspace(0.001, 0.999, 15)
    gamma_grid = np.linspace(0.0, 10.0, 15)
    s_cache = precompute_cache(kappa_scenarios, theta1_grid, gamma_grid, theta2_nodes, w_nodes, E_inv1p_theta2)
    for kappa in kappa_scenarios:
        tau = 1.0/(kappa*E_inv1p_theta2) - 1.0
        if tau <= 0: prob = 1.0
        elif tau >= 1: prob = 0.0
        else: prob = 1.0 - beta.cdf(tau, alpha, beta_param)
        p1s = 1.0/(1.0+theta1_worlds)
        rn_mc = float(np.mean(kappa*E_inv1p_theta2 > p1s))
        rn_rows.append({"mu": mu, "kappa": kappa, "tau threshold": tau,
                        "RN store prob (analytical)": prob, "RN store prob (MC)": rn_mc})
    for kappa in kappa_scenarios:
        s_bar = np.zeros(N); Y_bar = np.zeros(N); EU_bar = np.zeros(N)
        store_count = 0; eps = 1e-8
        for r in range(R):
            t1 = theta1_worlds[r]; t2 = theta2_worlds[r]
            p1 = 1.0/(1.0+t1); p2 = 1.0/(1.0+t2)
            s_star = interp2_vec(theta1_grid, gamma_grid, s_cache[kappa], t1, gammas)
            store_count += np.sum(s_star > eps)
            incomes = (1 - s_star)*p1 + s_star*kappa*p2
            Uvals = np.array([crra_u(incomes[i], gammas[i]) for i in range(N)])
            s_bar += (s_star - s_bar)/(r+1)
            Y_bar += (incomes - Y_bar)/(r+1)
            EU_bar += (Uvals - EU_bar)/(r+1)
        CE_bar = np.array([crra_u_inv(EU_bar[i], gammas[i]) for i in range(N)])
        RP_bar = Y_bar - CE_bar
        summary_rows.append({"mu": mu, "kappa": kappa,
                             "mean_income": Y_bar.mean(), "median_income": np.median(Y_bar),
                             "mean_CE": CE_bar.mean(), "mean_RP": RP_bar.mean(),
                             "share_storing": store_count/(N*R), "mean_s": s_bar.mean()})

summary_df = pd.DataFrame(summary_rows)
rn_df = pd.DataFrame(rn_rows)

plt.figure(figsize=(10,6))
for kappa in kappa_scenarios:
    d = summary_df[summary_df["kappa"]==kappa].sort_values("mu")
    plt.plot(d["mu"], d["share_storing"], marker="o", label=f"κ={kappa}")
plt.xlabel("Belief about buyer power mean μ")
plt.ylabel("Share storing (farmer–world)")
plt.title("Share of Storing Farmers vs Beliefs on Buyer Power, by storage efficiency κ")
plt.grid(True)
plt.legend(ncols=3)
plt.tight_layout()
plt.savefig(os.path.join(target_dir, "storage_subsidy_share_of_storage.png"), dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10,6))
for kappa in kappa_scenarios:
    d = summary_df[summary_df["kappa"]==kappa].sort_values("mu")
    plt.plot(d["mu"], d["mean_s"], marker="o", label=f"κ={kappa}")
plt.xlabel("Belief about buyer power mean μ")
plt.ylabel("Mean storage share s*")
plt.title("Mean Storage Share vs Beliefs on Buyer Power, by Storage Efficiency κ")
plt.grid(True)
plt.legend(ncols=3)
plt.tight_layout()
plt.savefig(os.path.join(target_dir, "storage_subsidy_mean_s.png"), dpi=300, bbox_inches="tight")
plt.show()