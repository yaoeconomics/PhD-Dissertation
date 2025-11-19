
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
# Global font settings
plt.rcParams.update({
    "axes.titlesize": 14,      # panel titles
    "axes.labelsize": 14,      # axis labels
    "xtick.labelsize": 14,     # x tick labels
    "ytick.labelsize": 14,     # y tick labels
    "legend.fontsize": 14,     # legend entries
    "legend.title_fontsize": 14,
    "figure.titlesize": 20     # suptitle
})

# Constants
theta1 = 0.5
p1 = 1 / (1 + theta1)
delta = 1.0
s_grid = np.linspace(0, 1, 100)
gamma_grid = np.linspace(0, 10, 50)
kappa_grid = np.linspace(0.6, 1.0, 50)
num_draws = 5000

def compute_beta_params(mu, sigma2):
    factor = mu * (1 - mu) / sigma2 - 1
    alpha = mu * factor
    beta_param = (1 - mu) * factor
    return alpha, beta_param

def crra_utility(pi, gamma):
    pi = np.maximum(pi, 1e-8)
    if gamma == 1:
        return np.log(pi)
    else:
        return (pi**(1 - gamma) - 1) / (1 - gamma)

def optimize_storage_share(theta2_draws, gamma, kappa):
    if gamma == 0:
        expected_p2 = np.mean(1 / (1 + theta2_draws))
        return 1.0 if delta * kappa * expected_p2 > p1 else 0.0
    utilities = []
    for s in s_grid:
        income = (1 - s) * p1 + delta * s * kappa / (1 + theta2_draws)
        util = np.mean(crra_utility(income, gamma))
        utilities.append(util)
    return s_grid[np.argmax(utilities)]

# Modified mean values
means_modified = [0.2, 0.4, 0.5, 0.8]
variances = [0.02, 0.05]

pdf_plots = []
surface_plots_low = []
surface_plots_high = []

for var in variances:
    for mu in means_modified:
        alpha, beta_param = compute_beta_params(mu, var)
        theta2_draws = beta.rvs(alpha, beta_param, size=num_draws)
        surface = np.zeros((len(gamma_grid), len(kappa_grid)))
        for i, gamma in enumerate(gamma_grid):
            for j, kappa in enumerate(kappa_grid):
                s_star = optimize_storage_share(theta2_draws, gamma, kappa)
                surface[i, j] = s_star
        if var == 0.02:
            surface_plots_low.append(surface)
        else:
            surface_plots_high.append(surface)
        x = np.linspace(0, 1, 1000)
        y = beta.pdf(x, alpha, beta_param)
        pdf_plots.append((x, y, mu, var, alpha, beta_param))

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(4, 4, height_ratios=[1, 3, 3, 1])

for i in range(4):
    ax = fig.add_subplot(gs[0, i])
    x, y, mu, var, alpha, beta_param = pdf_plots[i]
    ax.plot(x, y)
    ax.set_title(f"$\\mu={mu}$, $\\sigma^2={var}$\\n$\\alpha={alpha:.2f}, \\beta={beta_param:.2f}$")
    ax.set_xlim(0, 1)
    ax.grid(True)
    if i == 0:
        ax.set_ylabel("PDF")
    ax.set_xlabel("$\\theta_2$")

    ax = fig.add_subplot(gs[1, i], projection='3d')
    X, Y = np.meshgrid(kappa_grid, gamma_grid)
    Z = surface_plots_low[i]
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor='none')
    ax.set_title(f"$s^*$ (Low Variance)")
    ax.set_xlabel("$\\kappa$")
    ax.set_ylabel("$\\gamma$")
    ax.set_zlabel("$s^*$")
    ax.view_init(elev=30, azim=-135)
    ax.set_zlim(0, 1)

    ax = fig.add_subplot(gs[2, i], projection='3d')
    Z = surface_plots_high[i]
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor='none')
    ax.set_title(f"$s^*$ (High Variance)")
    ax.set_xlabel("$\\kappa$")
    ax.set_ylabel("$\\gamma$")
    ax.set_zlabel("$s^*$")
    ax.view_init(elev=30, azim=-135)
    ax.set_zlim(0, 1)

    ax = fig.add_subplot(gs[3, i])
    x, y, mu, var, alpha, beta_param = pdf_plots[i + 4]
    ax.plot(x, y)
    ax.set_title(f"$\\mu={mu}$, $\\sigma^2={var}$\\n$\\alpha={alpha:.2f}, \\beta={beta_param:.2f}$")
    ax.set_xlim(0, 1)
    ax.grid(True)
    if i == 0:
        ax.set_ylabel("PDF")
    ax.set_xlabel("$\\theta_2$")


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

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.savefig(os.path.join(target_dir, "3D_formulation.png"), dpi=300, bbox_inches="tight")
plt.show()
