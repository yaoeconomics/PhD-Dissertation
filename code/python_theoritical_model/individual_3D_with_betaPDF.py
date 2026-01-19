import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection in some environments)
import matplotlib.gridspec as gridspec
import os

# =========================
# Global font + style setup
# =========================
plt.rcParams.update({
    "axes.titlesize": 20,       # panel titles
    "axes.labelsize": 22,       # axis labels (bigger)
    "xtick.labelsize": 18,      # x tick labels
    "ytick.labelsize": 18,      # y tick labels
    "legend.fontsize": 18,
    "legend.title_fontsize": 18,
    "figure.titlesize": 24
})

# =========
# Constants
# =========
theta1 = 0.5
p1 = 1 / (1 + theta1)
delta = 1.0

s_grid = np.linspace(0, 1, 100)
gamma_grid = np.linspace(0, 10, 50)
kappa_grid = np.linspace(0.6, 1.0, 50)
num_draws = 5000

# Target case
mu = 0.2
sigma2 = 0.02

# =================
# Helper functions
# =================
def compute_beta_params(mu, sigma2):
    # Validity check: need 0 < sigma2 < mu(1-mu)
    max_var = mu * (1 - mu)
    if not (0 < sigma2 < max_var):
        raise ValueError(
            f"Invalid variance for Beta: need 0 < sigma2 < mu(1-mu) = {max_var:.4f}, "
            f"but got sigma2={sigma2}."
        )
    factor = mu * (1 - mu) / sigma2 - 1
    alpha = mu * factor
    beta_param = (1 - mu) * factor
    return alpha, beta_param

def crra_utility(pi, gamma):
    pi = np.maximum(pi, 1e-8)
    if gamma == 1:
        return np.log(pi)
    return (pi**(1 - gamma) - 1) / (1 - gamma)

def optimize_storage_share(theta2_draws, gamma, kappa):
    # Risk-neutral shortcut
    if gamma == 0:
        expected_p2 = np.mean(1 / (1 + theta2_draws))
        return 1.0 if delta * kappa * expected_p2 > p1 else 0.0

    utilities = []
    for s in s_grid:
        income = (1 - s) * p1 + delta * s * kappa / (1 + theta2_draws)
        util = np.mean(crra_utility(income, gamma))
        utilities.append(util)

    return s_grid[int(np.argmax(utilities))]

# ==========================
# Simulate theta2 and surface
# ==========================
alpha, beta_param = compute_beta_params(mu, sigma2)
theta2_draws = beta.rvs(alpha, beta_param, size=num_draws)

surface = np.zeros((len(gamma_grid), len(kappa_grid)))
for i, gamma in enumerate(gamma_grid):
    for j, kappa in enumerate(kappa_grid):
        surface[i, j] = optimize_storage_share(theta2_draws, gamma, kappa)

# PDF for theta2
x_pdf = np.linspace(0, 1, 1000)
y_pdf = beta.pdf(x_pdf, alpha, beta_param)

# ======================
# Plot: PDF + 3D surface
# ======================
fig = plt.figure(figsize=(16, 12), constrained_layout=True)
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], figure=fig)

# Top: PDF
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x_pdf, y_pdf)
ax1.set_title(rf"$\theta_2 \sim \mathrm{{Beta}}(\alpha,\beta)$  with  "
              rf"$\mu={mu}$, $\sigma^2={sigma2}$  "
              rf"($\alpha={alpha:.2f}$, $\beta={beta_param:.2f}$)")
ax1.set_xlim(0, 1)
ax1.grid(True)
ax1.set_ylabel("PDF")
ax1.set_xlabel(r"$\theta_2$")

# Bottom: 3D surface
ax2 = fig.add_subplot(gs[1, 0], projection="3d")
X, Y = np.meshgrid(kappa_grid, gamma_grid)
Z = surface

surf = ax2.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
ax2.set_title(r"$s^*$ surface (mean $0.2$, variance $0.02$)")
ax2.set_xlabel(r"$\kappa$ (Storage Efficiency)", labelpad=14)
ax2.set_ylabel(r"$\gamma$ (Risk Aversion)", labelpad=14)
ax2.set_zlabel(r"$s^*$ (Optimal Storage Share)", labelpad=14)

ax2.view_init(elev=30, azim=-135)
ax2.set_zlim(0, 1)

# ===== FORCE SAME WIDTH =====
pos2 = ax2.get_position()
pos1 = ax1.get_position()
ax1.set_position([pos2.x0, pos1.y0, pos2.width, pos1.height])


# ==========================
# Save to model_figures dir
# ==========================
# Robust save path: if __file__ doesn't exist (e.g., notebook), fall back to cwd.
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
target_dir = os.path.join(grandparent_dir, "model_figures")
os.makedirs(target_dir, exist_ok=True)

outpath = os.path.join(target_dir, "3D_formulation_mu0p2_var0p02_with_betaPDF.png")
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()
