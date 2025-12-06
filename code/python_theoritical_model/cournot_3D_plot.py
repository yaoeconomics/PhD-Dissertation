# Re-import required packages and redefine everything due to state reset
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    "axes.titlesize": 18,      # subplot titles
    "axes.labelsize": 16,      # axis labels
    "xtick.labelsize": 14,     # x tick labels
    "ytick.labelsize": 14,     # y tick labels
    "legend.fontsize": 14,     # legend text
    "figure.titlesize": 20     # suptitle
})

# --------------------------------
# Utility and pricing
# --------------------------------
def crra_utility(pi, gamma):
    with np.errstate(divide='ignore', invalid='ignore'):
        if gamma == 1:
            return np.log(pi)
        else:
            return (np.power(pi, 1 - gamma) - 1) / (1 - gamma)

def effective_price(N):
    """
    Cournot price with N buyers, no outside option:
    p_t = N_t / (N_t + 1)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return N / (N + 1)

# --------------------------------
# Optimal storage functions
# --------------------------------
def optimal_s(N2_samples, gamma, kappa):
    """
    Compute optimal storage share s* on a grid for given N2_samples, gamma, kappa.
    Here N1 = 2 so p1 = 2/3.
    """
    s_grid = np.linspace(0, 1, 25)
    expected_utils = []
    prices2 = effective_price(N2_samples)

    p1 = 2.0 / 3.0  # from N1 = 2

    for s in s_grid:
        income1 = (1 - s) * p1
        income2 = s * kappa * prices2
        income_total = income1 + income2
        EU = np.mean(crra_utility(income_total, gamma))
        expected_utils.append(EU)

    return s_grid[np.argmax(expected_utils)]

def closed_form_s(N2_samples, kappa):
    """
    Risk-neutral benchmark (gamma = 0).
    Store all (s = 1) if expected discounted future income exceeds current income.
    """
    p1 = 2.0 / 3.0  # from N1 = 2
    income1 = p1
    prices2 = effective_price(N2_samples)
    income2 = kappa * np.mean(prices2)
    return 1.0 if income2 > income1 else 0.0

# --------------------------------
# Parameters
# --------------------------------
mu_values = [8, 6, 3, 1]
gamma_grid = np.linspace(0, 10, 30)
kappa_grid = np.linspace(0.6, 1.0, 20)
s_min, s_max = 0.0, 1.0

# --------------------------------
# Figure: 2x4 grid with Poisson truncated to [1, 9]
# --------------------------------
fig = plt.figure(figsize=(20, 8))

for mu_idx, mu in enumerate(mu_values):
    # Simulate Poisson samples truncated to [1, 9]
    N2_samples = np.random.poisson(mu, 5000)
    N2_samples = np.clip(N2_samples, 1, 9)

    # Plot truncated PMF as curve (row 1)
    x_vals = np.arange(1, 10)
    pmf_vals = poisson.pmf(x_vals, mu)
    pmf_vals /= pmf_vals.sum()  # renormalize over [1, 9]

    ax = fig.add_subplot(2, 4, mu_idx + 1)
    ax.plot(x_vals, pmf_vals, marker='o', linestyle='-')
    ax.set_title(f"$\\mu$={mu} (Poisson, Truncated [1,9])")
    ax.set_xlim([1, 9])
    ax.set_ylabel("PMF")
    ax.set_xlabel("$N_2$")

    # Compute optimal s* surface
    s_star_surface = np.zeros((len(gamma_grid), len(kappa_grid)))
    for i, gamma in enumerate(gamma_grid):
        for j, kappa in enumerate(kappa_grid):
            if gamma == 0:
                s_star_surface[i, j] = closed_form_s(N2_samples, kappa)
            else:
                s_star_surface[i, j] = optimal_s(N2_samples, gamma, kappa)

    # Plot 3D surface (row 2)
    ax3d = fig.add_subplot(2, 4, mu_idx + 5, projection='3d')
    K, G = np.meshgrid(kappa_grid, gamma_grid)
    surf = ax3d.plot_surface(K, G, s_star_surface,
                             cmap='viridis',
                             edgecolor='none',
                             vmin=s_min, vmax=s_max)
    ax3d.set_title(f"$s^*(\\kappa, \\gamma)$ | $\\mu$={mu}")
    ax3d.set_xlabel("$\\kappa$")
    ax3d.set_ylabel("$\\gamma$")
    ax3d.set_zlabel("$s^*$")
    ax3d.set_xlim(kappa_grid.min(), kappa_grid.max())
    ax3d.set_ylim(gamma_grid.min(), gamma_grid.max())
    ax3d.set_zlim(s_min, s_max)
    ax3d.view_init(elev=30, azim=-135)

plt.tight_layout()
plt.suptitle("Optimal Storage under Cournot (Poisson Truncated [1,9])",
             fontsize=18, y=1.03)
plt.subplots_adjust(top=0.9, bottom=0.1)

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

# Save the figure
plt.savefig(os.path.join(target_dir, "3D_cournot.png"),
            dpi=300, bbox_inches="tight")

plt.show()
