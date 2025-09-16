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



# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from pathlib import Path
from matplotlib.lines import Line2D

# ---------------------- Primitives & Grids ----------------------
theta1, kappa, delta = 0.5, 0.9, 1.0
p1 = 1 / (1 + theta1)
variance = 0.02
mu_values = np.arange(0.05, 0.951, 0.01)   # E[theta2]
gammas    = [0.5, 2, 4, 7]                 # risk-averse levels
num_draws = 5000
s_grid    = np.linspace(0, 1, 25)

# Output directory (…/model_figures relative to this file if possible)
try:
    SAVE_DIR = (Path(__file__).resolve().parents[2] / "model_figures")
except NameError:  # e.g., notebook
    SAVE_DIR = Path.cwd() / "model_figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- Helpers ----------------------
def beta_params(mu, var):
    f = mu * (1 - mu) / var - 1
    return mu * f, (1 - mu) * f

def U_crra(pi, gamma):
    pi = np.maximum(pi, 1e-10)
    return np.log(pi) if gamma == 1 else (pi**(1 - gamma) - 1) / (1 - gamma)

def s_star_RN(v_mean):
    return 1.0 if delta * kappa * v_mean > p1 else 0.0

def s_star_RA(v_draws, gamma):
    # Broadcast over s_grid × draws
    incomes = (1 - s_grid)[:, None] * p1 + (s_grid[:, None] * kappa) * v_draws[None, :]
    util = U_crra(incomes, gamma).mean(axis=1)
    return s_grid[util.argmax()]

def E_income_from_v(s, v_mean):
    return (1 - s) * p1 + delta * s * kappa * v_mean

# ---------------------- Simulation (single pass) ----------------------
G = len(gammas); M = len(mu_values)
s_RA   = np.zeros((G, M))           # risk-averse s*
s_RN   = np.zeros(M)                # risk-neutral s*
gap_abs = np.zeros((G, M))          # abs gap
gap_pct = np.zeros((G, M))          # % gap

for j, mu in enumerate(mu_values):
    a, b = beta_params(mu, variance)
    th2  = beta.rvs(a, b, size=num_draws)
    v    = 1.0 / (1.0 + th2)
    vbar = v.mean()

    s_rn = s_star_RN(vbar)
    s_RN[j] = s_rn
    Ei_rn = E_income_from_v(s_rn, vbar)

    for i, g in enumerate(gammas):
        s_ra = s_star_RA(v, g)
        s_RA[i, j] = s_ra
        Ei_ra = E_income_from_v(s_ra, vbar)
        diff  = Ei_rn - Ei_ra
        gap_abs[i, j] = diff
        gap_pct[i, j] = 0.0 if Ei_rn <= 1e-12 else 100.0 * diff / Ei_rn

# Risk-neutral switch (approximate where s_RN changes)
switch_idx = np.where(np.diff(s_RN) != 0)[0]
mu_star = mu_values[switch_idx[0] + 1] if switch_idx.size > 0 else None

# ---------------------- Figure 1: Unified plot ----------------------
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(gammas)))

for i, g in enumerate(gammas):
    ax1.plot(mu_values, gap_abs[i], color=colors[i], lw=2)            # absolute (solid)
    ax2.plot(mu_values, gap_pct[i], color=colors[i], lw=2, ls='--')   # relative % (dashed)

if mu_star is not None:
    ax1.axvline(mu_star, color='k', ls=':', lw=1)
    ax2.axvline(mu_star, color='k', ls=':', lw=1)

ax1.set_xlabel(r"Expected second-period buyer power $\mathbb{E}[\theta_2]=\mu$")
ax1.set_ylabel(r"Absolute income gap  $\mathbb{E}[\pi(s^*_{RN})]-\mathbb{E}[\pi(s^*_{RA})]$")
ax2.set_ylabel(r"Relative gap  $\frac{\mathbb{E}[\pi(s^*_{RN})]-\mathbb{E}[\pi(s^*_{RA})]}{\mathbb{E}[\pi(s^*_{RN})]}\times100\%$")
ax1.set_title("Risk Aversion Cost: Absolute vs Relative Gaps (Unified View)")
ax1.grid(True, alpha=0.3)

legend1 = ax1.legend([Line2D([0], [0], color=colors[i], lw=2) for i in range(len(gammas))],
                     [f"γ = {g}" for g in gammas], title="Risk aversion (color)", loc="upper left")
legend2 = ax1.legend([Line2D([0], [0], color='grey', lw=2, ls='-'),
                      Line2D([0], [0], color='grey', lw=2, ls='--')],
                     ["Absolute (left axis)", "Relative % (right axis)"],
                     title="Line style", loc="upper right")
ax1.add_artist(legend1)

fig.tight_layout()
plt.savefig(os.path.join(target_dir, "income_gap_vs_mu_(unified).png"), dpi=300, bbox_inches="tight")
plt.show()

# ---------------------- Figure 2: Sensitivity to θ2 (s* vs μ) ----------------------
plt.figure(figsize=(10, 6))
for i, g in enumerate(gammas):
    plt.plot(mu_values, s_RA[i], color=colors[i], lw=2, label=rf"$\gamma={g}$")
plt.plot(mu_values, s_RN, 'k--', lw=2, label=r"Risk-neutral $s^*$")

if mu_star is not None:
    plt.axvline(mu_star, color='k', ls=':', lw=1)

plt.xlabel(r"Expected second-period buyer power $\mathbb{E}[\theta_2]=\mu$")
plt.ylabel(r"Optimal storage share $s^*$")
plt.title("Chosen Storage Shares: Risk-Averse vs. Risk-Neutral")
plt.grid(True, alpha=0.3)
plt.legend(title="Risk aversion")
plt.tight_layout()
plt.savefig(os.path.join(target_dir, "sensitivity_to_theta_2.png"), dpi=300, bbox_inches="tight")
plt.show()
