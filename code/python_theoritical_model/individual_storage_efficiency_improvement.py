# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from pathlib import Path
from matplotlib.lines import Line2D

# ============================================================
# Directory setup
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
target_dir = os.path.join(grandparent_dir, "model_figures")
os.makedirs(target_dir, exist_ok=True)

# ============================================================
# Primitives & Grids
# ============================================================
theta1, kappa, delta = 0.5, 0.9, 1.0
p1 = 1 / (1 + theta1)
variance = 0.02
mu_values = np.arange(0.05, 0.951, 0.01)   # E[theta2]
gammas    = [0.5, 2, 4, 7]                 # risk aversion levels
num_draws = 5000
s_grid    = np.linspace(0, 1, 25)

# Output directory (in case running from notebook)
try:
    SAVE_DIR = (Path(__file__).resolve().parents[2] / "model_figures")
except NameError:
    SAVE_DIR = Path.cwd() / "model_figures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Reproducibility: fixed random generator
# ============================================================
rng = np.random.default_rng(seed=42)

# ============================================================
# Helper functions
# ============================================================
def beta_params(mu, var):
    """Convert mean and variance to (α, β) parameters of Beta distribution."""
    f = mu * (1 - mu) / var - 1
    return mu * f, (1 - mu) * f

def U_crra(pi, gamma):
    pi = np.maximum(pi, 1e-10)
    return np.log(pi) if gamma == 1 else (pi**(1 - gamma) - 1) / (1 - gamma)

def s_star_RN(v_mean):
    """Risk-neutral optimal storage share."""
    return 1.0 if delta * kappa * v_mean > p1 else 0.0

def s_star_RA(v_draws, gamma):
    """Risk-averse optimal storage share via expected utility maximization."""
    incomes = (1 - s_grid)[:, None] * p1 + (s_grid[:, None] * kappa) * v_draws[None, :]
    util = U_crra(incomes, gamma).mean(axis=1)
    return s_grid[util.argmax()]

def E_income_from_v(s, v_mean):
    """Expected income given s and mean future price."""
    return (1 - s) * p1 + delta * s * kappa * v_mean

# ============================================================
# Simulation
# ============================================================
G = len(gammas)
M = len(mu_values)
s_RA   = np.zeros((G, M))           # risk-averse s*
s_RN   = np.zeros(M)                # risk-neutral s*
gap_abs = np.zeros((G, M))          # absolute income gap
gap_pct = np.zeros((G, M))          # relative gap (%)

for j, mu in enumerate(mu_values):
    a, b = beta_params(mu, variance)
    th2  = rng.beta(a, b, size=num_draws)
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

# ============================================================
# Figure 1: Risk-aversion cost (absolute vs relative)
# ============================================================
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(gammas)))

for i, g in enumerate(gammas):
    ax1.plot(mu_values, gap_abs[i], color=colors[i], lw=2)
    ax2.plot(mu_values, gap_pct[i], color=colors[i], lw=2, ls='--')

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

# ============================================================
# Figure 2: Sensitivity of s* to E[θ2]
# ============================================================
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

# ============================================================
# Figure 3: Expected Price Gain vs μ for θ1 = {0.25, 0.5, 0.75} (highlight θ1=0.5)
# ============================================================
E_p2 = np.zeros(M)
for j, mu in enumerate(mu_values):
    a, b = beta_params(mu, variance)
    th2  = rng.beta(a, b, size=num_draws)
    p2   = 1.0 / (1.0 + th2)
    E_p2[j] = p2.mean()

theta1_vals = [0.25, 0.50, 0.75]
p1_vals = [1.0 / (1.0 + t1) for t1 in theta1_vals]
gains = [kappa * E_p2 - p1 for p1 in p1_vals]

# Color scheme: muted gray for outer cases, strong blue for θ1=0.5
colors = {
    0.25: "lightgray",
    0.50: "darkblue",
    0.75: "lightgray"
}
linestyles = {
    0.25: "--",
    0.50: "-",
    0.75: "--"
}

plt.figure(figsize=(10, 6))
for t1, g in zip(theta1_vals, gains):
    plt.plot(mu_values, g, lw=3 if t1 == 0.5 else 2,
             color=colors[t1], ls=linestyles[t1],
             label=f"θ₁ = {t1:.2f}  (p₁={1/(1+t1):.3f})")

# Highlight y = 0 line (risk-neutral indifference)
plt.axhline(0.0, lw=2.5, color="darkred", alpha=0.7)
plt.text(mu_values[-1]*0.98, 0.005, r"Indifference: $\kappa\,\mathbb{E}[p_2]=p_1$",
         fontsize=11, color="red", ha="right", va="bottom")

plt.xlabel(r"Expected second-period buyer power $\mathbb{E}[\theta_2]=\mu$")
plt.ylabel(r"Expected Price Gain  $=\kappa\,\mathbb{E}[p_2]-p_1$")
plt.title(r"Expected Price Gain vs. $\mu$  ($\kappa=0.9$)")
plt.grid(True, alpha=0.3)
plt.legend(title="First-period buyer power", frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(target_dir, "expected_price_gain_vs_mu.png"), dpi=300, bbox_inches="tight")
plt.show()
